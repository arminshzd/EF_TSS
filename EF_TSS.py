import json
import os
import subprocess
from collections import defaultdict
import numpy as np
from pathlib import Path
from scipy.optimize import fsolve

class EF_TSS:
    """
    """
    def __init__(self, settings_json, initial_structure) -> None: # NOT DONE
        # Read the setting from the setting.json file
        with open(settings_json) as f:
            settings_dict_in = json.load(f)
        
        # Turn the setting dict into a default dict to prevent exceptions
        settings_dict = defaultdict(str)
        for key, val in settings_dict_in.items():
            settings_dict[key] = val
        
        self.N = settings_dict['N']
        self.charge = settings_dict['charge'] if not (settings_dict['charge'] == '') else 0
        self.spin = settings_dict['spin'] if not (settings_dict['spin'] == '') else 1
        self.N_procs = settings_dict['N-procs'] if not (settings_dict['N-procs'] == '') else 8
        self.R_trust = settings_dict['trust-radius'] if not (settings_dict['trust-radius'] == '') else 1
        self.R_conv = settings_dict['conv-radius'] if not (settings_dict['conv-radius'] == '') else 1e-6
        self.G_conv = settings_dict['conv-grad'] if not (settings_dict['conv-grad'] == '') else 1e-6
        self.max_iter = settings_dict['max-iter'] if not (settings_dict['conv-grad'] == '') else 1e-6

        self.basis_dir = Path(settings_dict['working-dir']) / (settings_dict['basis-f-name']) if not (settings_dict['basis-f-name'] == '') else ''
        self.hist_file = Path(settings_dict['working-dir']) / ((settings_dict['history-f-name'] + '.xyz') if not (settings_dict['history-f-name'] == '') else 'history.xyz')
        self.gjf_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.gjf') if not (settings_dict['gaussian-f-name'] == '') else 'in.gjf')
        self.log_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.log') if not (settings_dict['gaussian-f-name'] == '') else 'in.log')
        self.chk_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.chk') if not (settings_dict['gaussian-f-name'] == '') else 'in.chk')
        self.fchk_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.fchk') if not (settings_dict['gaussian-f-name'] == '') else 'in.fchk')
        self.submit_dir = settings_dict['submit-f-dir']

        self.force_calc_header = settings_dict['force-header-calc'] if not (settings_dict['force-header-calc'] == '') else "#P wB97XD/6-31G** nosymm" 
        self.hess_calc_header = settings_dict['hess-header-calc'] if not (settings_dict['force-header-calc'] == '') else "#P wB97XD/6-31G** nosymm freq"

        self.init_coords, self.atom_types, self.periphery = self._read_coords(initial_structure)

        self.atom_dict = {'H': 1, 'O': 8, 'Al':13, 'F': 9}
        self.atom_dict_r = {1: 'H', 8: 'O', 13: 'Al', 9: 'F'}
        self.atom_types_name = [self.atom_dict_r[i] for i in self.atom_types.squeeze()]
        

        self.H = np.zeros((3*self.N, 3*self.N))
        self.H_old = np.zeros((3*self.N, 3*self.N))
        self.G = np.zeros((1, 3*self.N))
        self.G_old = np.zeros((1, 3*self.N))
        self.evec_mem = []
    
    def _read_coords(self, coord_f): # Tested: Works
        """
        Reads geometric data from .inp file with exactly self.N lines with format:
        atom_type   periphery(0 or -1)   x   y   z
        """
        init_coords = np.zeros((self.N, 3))
        periphery = np.zeros((self.N, ), dtype='int8')
        atom_types = np.zeros((self.N, ), dtype='int8')
        atom_ind = 0
        with open(coord_f) as f:
            for line in f:
                line_s = line.split()
                atom_types[atom_ind] = line_s[0]
                periphery[atom_ind] = line_s[1]
                init_coords[atom_ind, 0] = line_s[2]
                init_coords[atom_ind, 1] = line_s[3]
                init_coords[atom_ind, 2] = line_s[4]
                atom_ind +=1
            
        return init_coords, atom_types, periphery

    def _get_grad(self) -> None: # Tested: Works
        with open(self.fchk_dir) as f:
            f_cnt = f.readlines()
            for ind, line in enumerate(f_cnt):
                line_s = line.split()
                if all(i in line_s for i in ['Cartesian', 'Gradient']):
                    start_ind = ind+1
                    break
            
            if (3*self.N)%5 == 0:
                end_ind = start_ind + (3*self.N)//5
            else:
                end_ind = start_ind + (3*self.N)//5 + 1

            G_raw = f_cnt[start_ind:end_ind]

        G_ind = 0
        for line in G_raw:
            line_list = line.split()
            for num in line_list:
                self.G[0, G_ind] = float(num)
                G_ind += 1
        
        return

    def _get_hessian(self) -> None: # Tested: Works
        with open(self.fchk_dir) as f:
            f_cnt = f.readlines()
            for ind, line in enumerate(f_cnt):
                line_s = line.split()
                if all(i in line_s for i in ['Cartesian', 'Force', 'Constants']):
                    start_ind = ind+1
                    break

            H_tot_size = int(3*self.N * (3*self.N + 1) / 2)
            if H_tot_size%5 == 0:
                end_ind = start_ind + H_tot_size//5
            else:
                end_ind = start_ind + H_tot_size//5 + 1

            H_raw = f_cnt[start_ind:end_ind]

        H_list = []
        for line in H_raw:
            line_list = line.split()
            for num in line_list:
                H_list.append(float(num))
                
        list_cntr = 0
        for i in range(3*self.N):
            for j in range(0, i+1):
                self.H[i, j] = H_list[list_cntr]
                self.H[j, i] = H_list[list_cntr]
                list_cntr += 1

        return
            
    def _get_lamda_bath(self, gamma, evals, mode) -> float: # Tested: Looks fine for now
        """
        Find lambda_BATH. Returns smallest root of equation 8.3.1 from Reaction Rate Theory and Rare Events.
        Gamma: Grad in eigenvector basis. 1xN vector.
        evals: vector of eigenvalues. 1XN vector.
        mode: selected mode to follow.
        """
        gamma_no_k = [i for i in gamma]
        gamma_no_k.pop(mode)
        evals_no_k = [i for i in evals]
        evals_no_k.pop(mode)
        # Instead of solving for all roots, just start with the minimum guess?
        #init_guess = [1.1*i for i in evals_no_k]
        #init_guess.insert(0, 0)
        init_guess = [min([0, 1.1*min(evals_no_k)])]
        res = []

        def _func(x):
            return x - sum([i**2/(x-j) for i,j in zip(gamma_no_k, evals_no_k)])
        
        for guess in init_guess:
            res.append(fsolve(_func, guess))

        return min(res)

    def _get_lambda_RC(self, gamma_k, evals_k) -> float:  # Tested: Looks fine for now 
        """
        Find lambda_RC. Returns biggest root of equation 8.3.2 from Reaction Rate Theory and Rare Events.
        Gamma: Grad in eigenvector basis. 1xN vector.
        evals: vector of eigenvalues. 1XN vector.
        mode: selected mode to follow.
        """
        init_guess = [0, 1.1*evals_k]
        res = []

        def _func(x):
            return x**2 - evals_k*x - gamma_k**2
        for guess in init_guess:
            res.append(fsolve(_func, guess))
            
        return max(res)

    def _get_ksi(self, evals, gamma, mode) -> np.array: # Tested: Looks fine for now
        """
        Get ksi vector as defined by equation 8.3.3 from Reaction Rate Theory and Rare Events.
        evals: eigenvalues of the Hessian. 1xN vector
        gamma:  Grad in eigenvector basis. 1xN vector
        mode: eigenmode to be followed. int
        """
        l_bath = self._get_lamda_bath(gamma, evals, mode)
        l_RC = self._get_lambda_RC(gamma[mode], evals[mode])

        ksi = np.zeros((1, len(gamma)))
        for i in range(len(gamma)):
            if i != mode:
                ksi[0, i] = -gamma[i]/(evals[i]-l_bath)
            else:
                ksi[0, i] = -gamma[i]/(evals[i]-l_RC)
        
        return ksi

    def _get_H_estim(self) -> np.array:
        pass

    def _get_g_estim(self) -> np.array:
        pass

    def _sub_gaussian(self) -> bool:
        """
        Submit a gaussian job for input file self.gjf_dir. Returns 0 for successful and 1 for failed jobs.
        """
        subprocess.run('{} {} {}'.format(self.submit_dir ,self.gjf_dir, self.log_dir), shell=True, check=True)
        with open(self.log_dir) as f:
            f_cnt = f.readlines()
            if all(i in f_cnt[-1].split() for i in ['Normal', 'Termination']):
                return 0
            else:
                return 1
    
    def _write_gaussian(self, struct, c_type='h') -> None: # Tested: works
        """
        write a freq calculation g16 input file with struct coords and self.basis for basis.
        """
        str_list = ["%NProcShared={}\n".format(self.N_procs),
        "%chk={}\n".format(self.chk_dir),
        "{}\n".format(self.hess_calc_header if c_type == 'h' else self.force_calc_header),
        "\n",
        "EF-TSS-calc-{}\n".format(c_type),
        "\n",
        "{} {}\n".format(self.charge, self.spin)
        ]

        for i in range(self.N):
            str_list.append("{}\t{}\t{}\t{}\t{}\n". format(self.atom_types[i], self.periphery[i], struct[i][0], struct[i][1], struct[i][2]))
        
        str_list.append("\n")

        if self.basis_dir != '':
            with open(self.basis_dir) as f:
                basis_list = f.readlines()

        str_list += basis_list
        str_list.append("\n")
        str_list.append("\n")

        with open(self.gjf_dir, 'w') as f:
            f.writelines(str_list)

        return

    def _write_history(self, struct) -> None: # Tested: works
        """
        write struct to .xyz file (self.hist_file).
        """
        str_list = []
        for i in range(self.N):
            str_list.append("{}\t{}\t{}\t{}\n".format(self.atom_types_name[i], struct[i, 0], struct[i, 1], struct[i, 2]))
        
        str_list.insert(0, "\n")
        str_list.insert(0, "{}\n".format(self.N))
        
        with open(self.hist_file, 'a') as f:
            f.writelines(str_list)

        return

    def run(self) -> np.array: # TESTED: looks to be fine
        """
        Run the EF-TSS algorithm with the initialized structure and parameters.
        """
        # Start by running gaussian force calcs on init structure
        curr_x = self.init_coords
        iter = 0
        # EF-TSS main loop
        while True:
            self._write_gaussian(curr_x)
            fail_flag = self._sub_gaussian()
            if fail_flag:
                raise RuntimeError("Force calculations failed. Check {} file".format(self.log_dir))

            #convert .chk file to .fchk
            # os.system("formchk {} {}".format(self.chk_dir, self.fchk_dir))

            # read gradient and Hessian from .fchk file
            self._get_grad()
            self._get_hessian()

            # Get eigenvectors and values

            evals, U = np.linalg.eig(self.H) ## U^T H U = evals

            # TODO Tricky: find a way to select the correct eigenvalue for the transition path....
            mode = 0

            # Convert gradient to eigenvector basis
            gamma = U.T @ self.G.T

            # Calculate ksi
            ksi = self._get_ksi(evals, gamma, mode)

            # Get step size
            dx = U @ ksi.T

            # Resize if bigger than trust radius
            if np.linalg.norm(dx) > self.R_trust:
                dx = self.R_trust * dx/np.linalg.norm(dx)

            # write current geometry to history file
            self._write_history(curr_x)

            # Check conversion criteria
            if np.linalg.norm(dx) <= self.R_conv:
                print("R_conv satisfied. Exiting...")
                return
            
            if np.linalg.norm(self.G) <= self.G_conv:
                print("G_conv satisfied. Exiting...")
                return
            
            if iter >= self.max_iter:
                print("Max_iter reached. Exiting...")
                return
            iter += 1

            # update geometry
            curr_x += dx.reshape(11, 3)

            # save values from last iter
            self.G_old = self.G
            self.H_old = self.H
            return












