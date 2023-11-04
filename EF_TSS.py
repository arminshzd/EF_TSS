import json
import os
import subprocess
from collections import defaultdict
from matplotlib.pyplot import delaxes
import numpy as np
from pathlib import Path
from scipy.optimize import fsolve

class EF_TSS:
    """
    """
    def __init__(self, settings_json, initial_structure) -> None:
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
        self.R_conv = settings_dict['conv-radius'] if not (settings_dict['conv-radius'] == '') else 0.1
        self.G_conv = settings_dict['conv-grad'] if not (settings_dict['conv-grad'] == '') else 1e-6
        self.max_iter = settings_dict['max-iter'] if not (settings_dict['max-iter'] == '') else 10
        self.reset_H_every = settings_dict['reset-H-every'] if not (settings_dict['reset-H-every'] == '') else 20
        
        self.basis_dir = Path(settings_dict['working-dir']) / (settings_dict['basis-f-name']) if not (settings_dict['basis-f-name'] == '') else ''
        self.hist_file = Path(settings_dict['working-dir']) / ((settings_dict['history-f-name'] + '.xyz') if not (settings_dict['history-f-name'] == '') else 'history.xyz')
        self.final_file = Path(settings_dict['working-dir']) / ((settings_dict['final-f-name'] + '.xyz') if not (settings_dict['final-f-name'] == '') else 'final.xyz')
        self.gjf_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.gjf') if not (settings_dict['gaussian-f-name'] == '') else 'in.gjf')
        self.log_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.log') if not (settings_dict['gaussian-f-name'] == '') else 'in.log')
        self.chk_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.chk') if not (settings_dict['gaussian-f-name'] == '') else 'in.chk')
        self.fchk_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.fchk') if not (settings_dict['gaussian-f-name'] == '') else 'in.fchk')
        self.submit_dir = settings_dict['submit-f-dir']

        self.energy_calc_header = settings_dict['force-header-calc'] if not (settings_dict['force-header-calc'] == '') else "#P wB97XD/6-31G** nosymm force" 
        self.hess_calc_header = settings_dict['hess-header-calc'] if not (settings_dict['hess-header-calc'] == '') else "#P wB97XD/6-31G** nosymm freq"

        self.init_coords, self.atom_types, self.periphery = self._read_coords(initial_structure)

        self.num_moving_atoms = 3*(self.N - (-sum(self.periphery)))

        self.atom_dict_r = {1 : "H", 2  : "He", 3  : "Li", 4  : "Be", 5  : "B", 6  : "C", 7  : "N", 8  : "O", 9  : "F", 10 : "Ne", 11 : "Na",\
                       12 : "Mg", 13 : "Al", 14 : "Si", 15 : "P", 16 : "S", 17 : "Cl", 18 : "Ar", 19 : "K", 20 : "Ca", 21 : "Sc", 22 : "Ti",\
                       23 : "V", 24 : "Cr", 25 : "Mn", 26 : "Fe", 27 : "Co", 28 : "Ni", 29 : "Cu", 30 : "Zn", 31 : "Ga", 32 : "Ge", 33 : "As",\
                       34 : "Se", 35 : "Br", 36 : "Kr", 37 : "Rb", 38 : "Sr", 39 : "Y", 40 : "Zr", 41 : "Nb", 42 : "Mo", 43 : "Tc", 44 : "Ru",\
                       45 : "Rh", 46 : "Pd", 47 : "Ag", 48 : "Cd", 49 : "In", 50 : "Sn", 51 : "Sb", 52 : "Te", 53 : "I", 54 : "Xe",  55 : "Cs",\
                       56 : "Ba", 57 : "La", 58 : "Ce", 59 : "Pr", 60 : "Nd", 61 : "Pm", 62 : "Sm", 63 : "Eu", 64 : "Gd", 65 : "Tb", 66 : "Dy",\
                       67 : "Ho", 68 : "Er", 69 : "Tm", 70 : "Yb", 71 : "Lu", 72 : "Hf", 73 : "Ta", 74 : "W ", 75 : "Re", 76 : "Os", 77 : "Ir",\
                       78 : "Pt", 79 : "Au", 80 : "Hg", 81 : "Tl", 82 : "Pb", 83 : "Bi", 84 : "Po", 85 : "At", 86 : "Rn"}
        self.atom_dict = {}
        for i, val in enumerate(list(self.atom_dict_r.values())):
            self.atom_dict[val] = list(self.atom_dict_r.keys())[i]
        self.atom_types_name = [self.atom_dict_r[i] for i in self.atom_types.squeeze()]
        

        self.E = 0
        self.E_old = 1e5
        self.H = np.zeros((3*self.N, 3*self.N))
        self.H_old = np.zeros((3*self.N, 3*self.N))
        self.G = np.zeros((1, 3*self.N))
        self.G_old = np.zeros((1, 3*self.N))
        self.dx = np.zeros((1, 3*self.N))
        self.evec_mem = []

        #Bofill's method default parameters
        self.R_trust = settings_dict['trust-radius'] if not (settings_dict['trust-radius'] == '') else 0.15
        self.bofill_params = {"Sf": 2, "Lb": 0, "Ub": 2, "r_l": 0.25, "r_u": 1.75, "R_1": self.R_trust}
        
    def _read_coords(self, coord_f):
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

    def _get_priphery_H(self):
        """
        Returns the hessian of the moving atoms only so it's invertible and non-zero
        """
        peri_H = np.copy(self.H)
        peri_H = peri_H[~np.all(peri_H == 0, axis=1)]
        peri_H = peri_H.T[~np.all(peri_H == 0, axis=0)]
        return peri_H

    def _get_padded_dx(self, dx):
        """Returns the padded dx vector

        Args:
            dx (np.array): dx for moving atoms: (3*self.num_moving_atoms, )

        Returns:
            padded_dx (np.array): dx padded with zeros for frozen atoms: (3*self.N, )
        """
        cnt = 0
        padded_dx = np.zeros((self.N*3, ))
        for i in range(len(self.periphery)):
            if self.periphery[i] != -1:
                padded_dx[3*i] = dx[cnt]
                padded_dx[3*i+1] = dx[cnt+1]
                padded_dx[3*i+2] = dx[cnt+2]
                cnt += 3
        
        return padded_dx

    def _get_grad(self, inplace=True) -> np.array:
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
        G_out = np.zeros_like(self.G)
        for line in G_raw:
            line_list = line.split()
            for num in line_list:
                G_out[0, G_ind] = float(num)
                G_ind += 1
        
        if inplace:
            self.G = G_out
        else:
            return G_out

    def _get_hessian(self) -> None:
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
            
    def _get_energy(self, inplace=True) -> float:
        """
        Get SCF energy from fchk file.
        """
        with open(self.fchk_dir) as f:
            f_cnt = f.readlines()
            for ind, line in enumerate(f_cnt):
                line_s = line.split()
                if all(i in line_s for i in ['SCF', 'Energy']):
                    line_ind = ind
                    break
            
            E_list = f_cnt[line_ind].split()

        if inplace:
            self.E = float(E_list[-1])
            return
        else:
            return float(E_list[-1])

    def _get_lamda_bath(self, gamma, evals, mode) -> float:
        """
        Find lambda_BATH. Returns smallest root of equation 8.3.1 from Reaction Rate Theory and Rare Events.
        Gamma: Grad in eigenvector basis. 1xN vector.
        evals: vector of eigenvalues. 1XN vector.
        mode: selected mode to follow.
        """
        gamma_no_k = [i for i in gamma]
        gamma_no_k.pop(mode)
        gamma_no_k = [i for i in gamma_no_k if i!=0]
        evals_no_k = [i for i in evals]
        evals_no_k.pop(mode)
        evals_no_k = [i for i in evals_no_k if i!=0]

        # Instead of solving for all roots, just start with the minimum guess.
        init_guess = [1.1*i for i in evals_no_k]
        init_guess.insert(0, 0)
        init_guess = list(set(init_guess))
        res = []

        def _func(x):
            return x - sum([i**2/(x-j) for i,j in zip(gamma_no_k, evals_no_k)])
        
        for guess in init_guess:
            res.append(fsolve(_func, guess))

        return min(res)

    def _get_lambda_RC(self, gamma_k, evals_k) -> float:
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

    def _get_ksi(self, evals, gamma, mode) -> np.array:
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

    def _get_H_estim(self) -> None:
        """Calculate and set self.H from Bofill's method to estimate Hessian
        Returns:
            None
        """
        gamma_k = self.G - self.G_old
        delta_k = self.dx.reshape((1, 3*self.N))
        ksi_k = gamma_k - delta_k@self.H_old
        phi_k = 1 - (delta_k @ gamma_k.T)**2/((delta_k@delta_k.T)*(gamma_k@gamma_k.T))
        B_ms = self.H_old + 1/(delta_k@ksi_k.T)*(ksi_k.T@ksi_k)
        B_p = self.H_old - (delta_k@ksi_k.T)/(delta_k@delta_k.T)**2 * delta_k.T@delta_k + \
            1/(delta_k@delta_k.T)*(ksi_k.T@delta_k + delta_k.T@ksi_k)
        self.H = (1-phi_k)*B_ms + phi_k*B_p
        return

    def _get_g_estim(self) -> None: # Wrong!
        """Numerically estimate gradient from dE/dx"""
        loc_dx = self.dx[:self.num_moving_atoms]
        loc_G = (self.E - self.E_old)*np.ones_like(loc_dx)/loc_dx
        loc_G = np.pad(loc_G, (0, 3*self.N - self.num_moving_atoms), 'constant', constant_values=(0, 0))
        self.G = loc_G.reshape(1, 3*self.N)
        return

    def _sub_gaussian(self) -> bool:
        """
        Submit a gaussian job for input file self.gjf_dir. Returns 0 for successful and 1 for failed jobs.
        """
        subprocess.run('{} {} {} > {}'.format(self.submit_dir ,self.gjf_dir, self.log_dir, self.log_dir), shell=True, check=True)
        with open(self.log_dir) as f:
            f_cnt = f.readlines()
            f_cnt = f_cnt[::-1]
            f_cnt = f_cnt[:100]
            for line in f_cnt:
                if all(i in line.split() for i in ['Normal', 'termination']):
                    return 0
            return 1
    
    def _write_gaussian(self, struct, c_type='H') -> None:
        """
        write a freq calculation g16 input file with struct coords and self.basis for basis.
        """
        str_list = ["%NProcShared={}\n".format(self.N_procs),
        "%chk={}\n".format(self.chk_dir),
        "{}\n".format(self.hess_calc_header if c_type == 'H' else self.energy_calc_header),
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

    def _write_history(self, struct, fname) -> None:
        """
        write struct to .xyz file.
        """
        str_list = []
        for i in range(self.N):
            str_list.append("{}\t{}\t{}\t{}\n".format(self.atom_types_name[i], struct[i, 0], struct[i, 1], struct[i, 2]))
        
        str_list.insert(0, "\n")
        str_list.insert(0, "{}\n".format(self.N))
        
        with open(fname, 'a') as f:
            f.writelines(str_list)

        return

    def run(self) -> np.array:
        """
        Run the EF-TSS algorithm with the initialized structure and parameters.
        """
        # Start by running gaussian force calcs on init structure
        curr_x = self.init_coords
        self._write_gaussian(curr_x)
        fail_flag = self._sub_gaussian()
        if fail_flag:
                raise RuntimeError("Initial force calculations failed. Check {} file".format(self.log_dir))
        
        os.system("formchk {} {} > /dev/null 2>&1".format(self.chk_dir, self.fchk_dir))
        self._get_energy()
        self._get_grad()
        self._get_hessian()
        
        iter = 1
        bofill_flag = True
        if self.reset_H_every == 1:
            calc_H_flag = True
        else:
            calc_H_flag = False
        
        # EF-TSS main loop
        while True:
            if iter%self.reset_H_every == 0:
                self._write_gaussian(curr_x)
                bofill_flag = False
            else:
                if calc_H_flag:
                    self._write_gaussian(curr_x, c_type="E")
                bofill_flag = True
            
            if (calc_H_flag) or (iter%self.reset_H_every == 0):
                fail_flag = self._sub_gaussian()
            
            if fail_flag:
                raise RuntimeError("Force calculations failed. Check {} file".format(self.log_dir))

            #convert .chk file to .fchk
            os.system("formchk {} {} > /dev/null 2>&1".format(self.chk_dir, self.fchk_dir))

            # Update the energy and the energy history
            self.E_old = self.E
            self._get_energy()

            # read gradient and Hessian from .fchk file or estimate it with Bofill's method
            # save values from last iter
            
            self.G_old = self.G
            self.H_old = self.H

            if not bofill_flag:
                print("Replacing Hessian from freq calculations.")
                self._get_grad()
                self._get_hessian()
            else:
                print("Bofill estimate.")
                

            # Get eigenvectors and values

            evals, U = np.linalg.eig(self.H) ## U^T H U = evals

            if iter == 1:
                print(evals.squeeze())
                print(sorted(evals.squeeze(), reverse=True))
                mode = int(input("Which mode to follow: "))
                self.evec_mem.append(U[mode, :])
                
            else:
                overlap = 1
                if calc_H_flag:
                    overlap = 0
                    mode = 0
                    old_mix_evec = np.sum(np.array(self.evec_mem), axis=0)
                    old_mix_evec /= np.linalg.norm(old_mix_evec)
                    for i in range(3*self.N):
                        c_overlap = abs(np.dot(old_mix_evec, U[i, :]))
                        if c_overlap >= overlap:
                            mode = i
                            overlap = c_overlap
                    self.evec_mem.append(U[mode, :] if np.dot(old_mix_evec, U[mode, :]) > 0 else -1*U[mode, :])
                
                print("Following mode {} with eigenvalue {} with overlap {} with last followed eigenvector.".format(mode, evals.squeeze()[mode], overlap))

            #eigenvalues without the followed mode and zeros
            evals_no_mode = np.concatenate((evals[:mode], evals[mode+1:]), axis=0)
            evals_no_mode = [i for i in evals_no_mode if i!=0]
            

            if bofill_flag:
                # if we are estimating H from Bofill's method
                # the complete H has zeros all over the place and is singular (b/c we have frozen atoms).
                # We'll calculate dx for the atoms that move and then pad dx with zeros.
                calc_H_flag = True
                peri_H = self._get_priphery_H()
                peri_G = self.G[self.G != 0]
                dx = -np.linalg.inv(peri_H)@(peri_G.T)
                dx = self._get_padded_dx(dx)
                temp_G = self.G
                temp_E = self.E
                iter_dx_size = np.linalg.norm(dx)
                if (np.linalg.norm(dx) > self.R_trust) or (evals[mode] > 0) or (len([i for i in evals if i<0]) != 1):
                    # if ||H^-1 G|| > R_trust or we have more than 1 negative eigen values or the eigen value of the mode we are following is positive
                    alphas = np.linalg.solve(U.T, self.G.T)
                    v_k_old = np.linalg.norm(self.G)/self.R_trust + max(-min(evals_no_mode), evals[mode])
                    while True:
                        # While loop to solve for v_k
                        delta_vk = -(alphas[mode])/(evals[mode] - v_k_old)*U[mode, :] - np.sum([(alphas[i])/(evals[i] + v_k_old)*U[i, :] for i in range(3*self.N) if ((i != mode) & (evals[i]!=0))], axis=0)
                        delta_vk_p = ((alphas[mode]**2)/(evals[mode] - v_k_old)**3 - np.sum([(alphas[i]**2)/(evals[i] + v_k_old)**3 for i in range(3*self.N) if ((i != mode) & (evals[i]!=0))], axis=0))/(np.linalg.norm(delta_vk))
                        v_k = v_k_old + (1-(np.linalg.norm(delta_vk))/(self.R_trust))*(np.linalg.norm(delta_vk))/(delta_vk_p)

                        if abs(v_k-v_k_old) < 1e-6:
                            if v_k <= max(evals[mode], -min(evals_no_mode)):
                                self.R_trust /= self.bofill_params['Sf']
                                continue
                            else:
                                dx = delta_vk
                                temp_x = curr_x + dx.reshape(self.N, 3)
                                self._write_gaussian(temp_x, c_type='E')
                                fail_flag = self._sub_gaussian()

                                if fail_flag:
                                    raise RuntimeError("Gaussian calculations failed. Check {} file".format(self.log_dir))

                                os.system("formchk {} {} > /dev/null 2>&1".format(self.chk_dir, self.fchk_dir))
                                temp_E = self._get_energy(inplace = False)
                                temp_G = self._get_grad(inplace=False)
                                q = self.E + temp_G@dx[:, np.newaxis] + 0.5*dx[:, np.newaxis].T@self.H@dx[:, np.newaxis]
                                r_k = (temp_E - self.E)/(q - self.E)
                                if (r_k > self.bofill_params['r_u']) or (r_k < self.bofill_params['r_l']):
                                    self.R_trust /= self.bofill_params['Sf']
                                elif np.abs(np.linalg.norm(dx)-self.R_trust) < 1e-3:
                                    self.R_trust *= self.bofill_params['Sf']**(1/2)
                                if (r_k > self.bofill_params['Ub']) or (r_k < self.bofill_params['Lb']):
                                    dx = np.zeros_like(dx)
                                    temp_G = self.G
                                    temp_E = self.E
                                    calc_H_flag = False
                                    break
                                else:
                                    break

                        v_k_old = v_k
                    
                self.dx = dx
                self.G = temp_G
                self.E = temp_E
                if calc_H_flag:
                    self._get_H_estim()
                

            else:    
                # calculating based on actual H
                # Convert gradient to eigenvector basis
                # TODO: take a step in the mixture of previous eigenvectors instead of just the one from this iteration.
                gamma = U.T @ self.G.T

                # Calculate ksi
                ksi = self._get_ksi(evals, gamma, mode)

                # Get step size
                self.dx = U @ ksi.T
                iter_dx_size = np.linalg.norm(self.dx)
            
            
                # Resize if bigger than trust radius
                if np.linalg.norm(self.dx) > self.R_trust:
                    self.dx = self.R_trust * self.dx/np.linalg.norm(self.dx)


            print("Iteration {}:\tdx: {}\tgrad: {}\n". format(iter, iter_dx_size, np.linalg.norm(self.G)))
            # write current geometry to history file
            self._write_history(curr_x, self.hist_file)

            # Check conversion criteria
            if np.linalg.norm(iter_dx_size) <= self.R_conv:
                print("R_conv satisfied. Writing the final structure to final.xyz")
                break
            
            if np.linalg.norm(self.G) <= self.G_conv:
                print("G_conv satisfied. Writing the final structure to final.xyz")
                break
            
            if iter >= self.max_iter:
                print("Max_iter reached. Writing the final structure to final.xyz")
                break
            iter += 1

            # update geometry
            curr_x += self.dx.reshape(self.N, 3)
        
        self._write_history(curr_x, self.final_file)












