import json
import os
import subprocess
from collections import defaultdict
import numpy as np
from pathlib import Path
from scipy.optimize import fsolve

class EF_TSS:
    def __init__(self, settings_json, initial_structure) -> None:
        """
        Initialize an EF_TSS (Eigenvector Following - Transition State Search) object with the specified settings
        and molecular structure.

        Parameters
        ----------
        settings_json : str or Path
            Path to a JSON file containing input parameters for the EF-TSS run. The dictionary may contain:

            Required:
                - "N": (int) Number of atoms in the structure.
                - "working-dir": (str) Path to the working directory where all intermediate and output files are stored.
                - "submit-f-dir": (str) Path to the script or command used to submit GAUSSIAN jobs (e.g., g16, sbatch wrapper).

            Optional (with default behavior if not set or set to ""):
                - "charge": (int or str) Total molecular charge (default: 0).
                - "spin": (int or str) Spin multiplicity (default: 1).
                - "N-procs": (int or str) Number of processors for GAUSSIAN jobs (default: 8).
                - "conv-radius": (float) Convergence threshold for the coordinate update norm (default: 0.1).
                - "conv-grad": (float) Convergence threshold for the gradient norm (default: 1e-6).
                - "max-iter": (int) Maximum number of optimization iterations (default: 10).
                - "reset-H-every": (int) Number of iterations after which to recompute the full Hessian (default: 20).
                - "trust-radius": (float) Trust radius for controlling step sizes in optimization (default: 0.15).
                - "basis-f-name": (str) Name of the file containing the GAUSSIAN basis set block (default: "").
                - "history-f-name": (str) Name of the XYZ file to store optimization trajectory (default: "history.xyz").
                - "final-f-name": (str) Name of the XYZ file to store final structure (default: "final.xyz").
                - "gaussian-f-name": (str) Base name for GAUSSIAN input/output files (default: "in").
                - "energy-header-calc": (str) GAUSSIAN header for single-point energy + gradient calculation (default: "#P wB97XD/6-31G** nosymm force").
                - "hess-header-calc": (str) GAUSSIAN header for Hessian (frequency) calculation (default: "#P b3lyp/6-31G** nosymm freq").

        initial_structure : str or Path
            Path to a plain text file containing the initial geometry. The file should have exactly N lines.
            Each line must follow the format:
                atomic_number   freeze_flag(0 or -1)   x   y   z

            Here, `freeze_flag = -1` marks the atom as frozen (not optimized), and 0 means it is free to move.

        Notes
        -----
        - Initializes geometry, gradient, Hessian matrices, and optimization parameters.
        - File paths for GAUSSIAN `.gjf`, `.log`, `.chk`, `.fchk` files are constructed from `working-dir` and `gaussian-f-name`.
        - If `basis-f-name` is provided, its contents will be appended to GAUSSIAN input files.
        - A dictionary of atomic numbers and element names is created for use in XYZ output.
        - Convergence and Bofill update controls are initialized.
        """

        # Read the setting from the setting.json file
        with open(settings_json) as f:
            settings_dict_in = json.load(f)
        
        # Check that the required keys are available
        required_keys = ['N', 'working-dir', 'submit-f-dir']
        for key in required_keys:
            if not settings_dict_in.get(key):
                raise ValueError(f"Missing required setting: {key}")
        
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

        self.energy_calc_header = settings_dict['energy-header-calc'] if not (settings_dict['energy-header-calc'] == '') else "#P wB97XD/6-31G** nosymm force" 
        self.hess_calc_header = settings_dict['hess-header-calc'] if not (settings_dict['hess-header-calc'] == '') else "#P wB97XD/6-31G** nosymm freq"

        self.init_coords, self.atom_types, self.periphery = self._read_coords(initial_structure)

        self.num_moving_atoms = np.sum(self.periphery != -1) * 3

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
        self.evec_mem_size = 10

        #Bofill's method default parameters
        self.R_trust = settings_dict['trust-radius'] if not (settings_dict['trust-radius'] == '') else 0.15
        self.bofill_params = {"Sf": 2, "Lb": 0, "Ub": 2, "r_l": 0.25, "r_u": 1.75, "R_1": self.R_trust}
        
    def _read_coords(self, coord_f):
        """
        Read atomic coordinates, atom types, and freeze flags from an input file.

        Parameters
        ----------
        coord_f : str or Path
            Path to the coordinate input file. The file must contain exactly N lines,
            each formatted as:
                atomic_number   freeze_flag(0 or -1)   x   y   z

        Returns
        -------
        init_coords : np.ndarray, shape (N, 3)
            Array of Cartesian coordinates for all atoms.
        
        atom_types : np.ndarray, shape (N,)
            Array of atomic numbers for each atom.

        periphery : np.ndarray, shape (N,)
            Array of freeze flags: `0` for movable atoms, `-1` for frozen atoms.
        
        Notes
        -----
        This method assumes that the number of lines in the file matches the value of self.N,
        and that the format of each line is strictly followed.
        """

        init_coords = np.zeros((self.N, 3))
        periphery = np.zeros((self.N, ), dtype='int')
        atom_types = np.zeros((self.N, ), dtype='int')
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
        Extract the sub-block of the full Hessian corresponding to only the moving atoms.

        Returns
        -------
        peri_H : np.ndarray
            Square Hessian matrix of shape (3 * num_moving_atoms, 3 * num_moving_atoms),
            containing only the non-zero rows and columns corresponding to atoms not marked
            as frozen (i.e., periphery != -1).

        Notes
        -----
        This reduced Hessian is useful for numerical stability in inverse and eigendecomposition
        operations, since the full Hessian includes zero blocks for frozen atoms.
        """

        mask = np.repeat(self.periphery != -1, 3)
        peri_H = self.H[np.ix_(mask, mask)]
        return peri_H

    def _get_padded_dx(self, dx):
        """
        Pad a displacement vector for moving atoms with zeros for frozen atoms.

        Parameters
        ----------
        dx : np.ndarray, shape (3 * num_moving_atoms,)
            Displacement vector computed only for the movable atoms.

        Returns
        -------
        padded_dx : np.ndarray, shape (3 * N,)
            Full displacement vector where entries corresponding to frozen atoms are set to zero,
            and entries for movable atoms are inserted in the correct positions.

        Notes
        -----
        This is used to maintain consistent vector dimensions for update steps and file writing,
        while respecting frozen atom constraints.
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

    def _get_grad(self, inplace=True):
        """
        Extract the Cartesian gradient vector from a Gaussian .fchk file.

        Parameters
        ----------
        inplace : bool, optional (default=True)
            If True, updates self.G with the parsed gradient.
            If False, returns the parsed gradient without modifying object state.

        Returns
        -------
        G_out : np.ndarray, shape (1, 3 * N), optional
            The Cartesian gradient vector, returned only if `inplace=False`.

        Notes
        -----
        The method locates the "Cartesian Gradient" section in the .fchk file
        and reads the corresponding gradient values for all atoms.
        """

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
        """
        Extract the full Cartesian Hessian matrix from a Gaussian .fchk file and store it in self.H.

        Returns
        -------
        None

        Notes
        -----
        This method reads the "Cartesian Force Constants" section from the .fchk file, which stores
        the lower triangular part of the symmetric Hessian matrix in a packed format. It reconstructs
        the full symmetric Hessian and assigns it to self.H.
        """

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

        return None
            
    def _get_energy(self, inplace=True) -> float:
        """
        Extract the SCF electronic energy from a Gaussian .fchk file.

        Parameters
        ----------
        inplace : bool, optional (default=True)
            If True, sets the extracted energy to self.E.
            If False, returns the extracted energy as a float.

        Returns
        -------
        energy : float, optional
            The SCF energy value, returned only if `inplace=False`.

        Notes
        -----
        This method searches for the "SCF Energy" field in the .fchk file and parses its value.
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
        Compute the lambda_BATH parameter as described in Equation 8.3.1 of
        "Reaction Rate Theory and Rare Events".

        Parameters
        ----------
        gamma : list or np.ndarray
            Gradient vector projected into the eigenvector basis. Shape: (N,)
        
        evals : list or np.ndarray
            Eigenvalues of the Hessian matrix. Shape: (N,)
        
        mode : int
            Index of the eigenmode being followed during the transition state search.

        Returns
        -------
        lambda_bath : float
            The smallest real root of the lambda_BATH equation, used for computing the
            search direction in the eigenvector basis.

        Notes
        -----
        The lambda_BATH parameter governs the projection of the gradient orthogonal to the selected mode.
        This implementation solves the root-finding problem using SciPy's `fsolve`.
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
        Compute the lambda_RC parameter as described in Equation 8.3.2 of
        "Reaction Rate Theory and Rare Events".

        Parameters
        ----------
        gamma_k : float
            The gradient component along the selected reaction coordinate (eigenmode).
        
        evals_k : float
            The eigenvalue of the Hessian corresponding to the selected eigenmode.

        Returns
        -------
        lambda_RC : float
            The largest real root of the lambda_RC equation, used in computing the
            reaction coordinate step in the eigenvector basis.

        Notes
        -----
        This parameter adjusts the step size along the chosen transition mode.
        """

        init_guess = [0, 1.1*evals_k]
        res = []

        def _func(x):
            return x**2 - evals_k*x - gamma_k**2
        for guess in init_guess:
            res.append(fsolve(_func, guess))
            
        return max(res)[0]

    def _get_ksi(self, evals, gamma, mode) -> np.array:
        """
        Compute the ksi vector used for transition state displacement, as defined
        in Equation 8.3.3 of "Reaction Rate Theory and Rare Events".

        Parameters
        ----------
        evals : np.ndarray
            Eigenvalues of the Hessian matrix. Shape: (N,)
        
        gamma : np.ndarray
            Gradient vector projected into the eigenvector basis. Shape: (1, N)
        
        mode : int
            Index of the eigenmode being followed.

        Returns
        -------
        ksi : np.ndarray, shape (1, N)
            Displacement vector in the eigenvector basis, incorporating both lambda_RC
            and lambda_BATH components.

        Notes
        -----
        This displacement vector forms the search direction in eigenvector-following,
        with special treatment of the selected mode versus all other orthogonal modes.
        """

        l_bath = self._get_lamda_bath(gamma, evals, mode)
        l_RC = self._get_lambda_RC(gamma[mode], evals[mode])

        ksi = np.zeros((1, len(gamma)))
        for i in range(len(gamma)):
            if i != mode:
                ksi[0, i] = -gamma[i]/(evals[i]-l_bath + 1e-8)
            else:
                ksi[0, i] = -gamma[i]/(evals[i]-l_RC + 1e-8)
        
        return ksi

    def _get_H_estim(self) -> None:
        """
        Estimate the Hessian matrix using Bofill's update formula based on gradient
        and geometry changes between iterations.

        Returns
        -------
        None

        Notes
        -----
        This method uses both the symmetric (MS) and Powell (P) update forms to compute
        a blended approximation of the Hessian:
            H_new = (1 - phi_k) * B_MS + phi_k * B_P

        It avoids explicit frequency calculations, enabling efficient updates when
        full Hessian recomputation is unnecessary.
        """

        mask = np.repeat(self.periphery != -1, 3)
        gamma_k = (self.G - self.G_old)[0, mask].reshape(1, -1)
        delta_k = self.dx[0, mask].reshape(1, -1)
        H_old = self.H_old[np.ix_(mask, mask)]

        ksi_k = gamma_k - delta_k @ H_old
        denom = (delta_k @ ksi_k.T)
        phi_k = 1 - (delta_k @ gamma_k.T) ** 2 / ((delta_k @ delta_k.T) * (gamma_k @ gamma_k.T) + 1e-12)

        B_ms = H_old + (ksi_k.T @ ksi_k) / (denom + 1e-12)
        B_p = (
            H_old
            - ((delta_k @ ksi_k.T) / (delta_k @ delta_k.T) ** 2) * (delta_k.T @ delta_k)
            + (1 / (delta_k @ delta_k.T)) * (ksi_k.T @ delta_k + delta_k.T @ ksi_k)
        )

        H_new = (1 - phi_k) * B_ms + phi_k * B_p

        self.H[np.ix_(mask, mask)] = H_new
        return None

    def _get_g_estim(self) -> None:
        """
        Estimate the gradient numerically using finite difference of energy with respect to geometry.
        Frozen atoms are assigned zero gradient.
        """

        # Identify which indices correspond to movable atoms (x, y, z for each)
        mask = np.repeat(self.periphery != -1, 3)  # e.g., [False, False, False, True, True, True]
        
        dx_flat = self.dx.flatten()  # shape: (3N,)
        G_est = np.zeros_like(dx_flat)  # initialize with zeros

        delta_E = self.E - self.E_old

        # Identify where displacement is both nonzero and atom is movable
        safe_mask = (np.abs(dx_flat) > 1e-6) & mask

        # Apply finite difference only where atoms are movable
        G_est[safe_mask] = delta_E / (dx_flat[safe_mask])

        self.G = G_est.reshape(1, 3 * self.N)
        return None

    def _sub_gaussian(self) -> bool:
        """
        Submit a Gaussian job using the current input file and submission command.

        Returns
        -------
        success_flag : int
            Returns 0 if the Gaussian job terminated normally, 1 otherwise.

        Notes
        -----
        The job is launched via a shell command using the path specified in `self.submit_dir`.
        The method scans the last ~100 lines of the Gaussian log file for the phrase
        "Normal termination" to verify success.
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
        Write a Gaussian input (.gjf) file for energy or Hessian calculation.

        Parameters
        ----------
        struct : np.ndarray, shape (N, 3)
            Cartesian coordinates of the current structure.

        c_type : str, optional (default='H')
            Calculation type: 'H' for Hessian (frequency), anything else for energy/gradient.

        Returns
        -------
        None

        Notes
        -----
        - The Gaussian route section is determined by `self.hess_calc_header` or `self.energy_calc_header`.
        - If a basis set file is provided via `self.basis_dir`, its contents are appended to the input.
        - Structure lines include atom type, periphery flag, and coordinates.
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
        Append the given structure to an XYZ file.

        Parameters
        ----------
        struct : np.ndarray, shape (N, 3)
            Cartesian coordinates to write.

        fname : str or Path
            Path to the XYZ file where the structure will be appended.

        Returns
        -------
        None

        Notes
        -----
        Each structure is written in XYZ format with a two-line header (atom count and blank line),
        followed by element symbols and coordinates. The atomic symbols are determined using
        `self.atom_types_name`.
        """

        str_list = []
        for i in range(self.N):
            str_list.append("{}\t{}\t{}\t{}\n".format(self.atom_types_name[i], struct[i, 0], struct[i, 1], struct[i, 2]))
        
        str_list.insert(0, "\n")
        str_list.insert(0, "{}\n".format(self.N))
        
        with open(fname, 'a') as f:
            f.writelines(str_list)

        return None
    
    def _eig_decomp_reduced_H(self):
        """
        Perform eigendecomposition on the reduced Hessian and map eigenvectors to full space.

        Returns
        -------
        evals : np.ndarray, shape (3 * num_moving_atoms,)
            Eigenvalues of the reduced Hessian.
        U_full : np.ndarray, shape (3 * N, 3 * num_moving_atoms)
            Full-space eigenvectors padded with zeros at frozen atom indices.
        """
        peri_H = self._get_priphery_H()
        evals, U = np.linalg.eigh(peri_H)

        mask = np.repeat(self.periphery != -1, 3)
        U_full = np.zeros((3 * self.N, len(evals)))
        U_full[mask, :] = U

        return evals, U_full

    def run(self):
        """
        Run the Eigenvector Following Transition State Search (EF-TSS) algorithm.

        Returns
        -------
        final_coords : np.ndarray, shape (N, 3)
            Cartesian coordinates of the final optimized structure.
        """

        # Initial force + Hessian calculation
        curr_x = self.init_coords
        self._write_gaussian(curr_x, c_type="H")
        if self._sub_gaussian():
            raise RuntimeError(f"Initial force calculations failed. Check {self.log_dir}")

        os.system(f"formchk {self.chk_dir} {self.fchk_dir} > /dev/null 2>&1")
        self._get_energy()
        self._get_grad()
        self._get_hessian()

        iter = 1
        mode = None

        while True:
            # Decide whether to recompute Hessian from scratch or use Bofill estimate
            recalc_H = (iter % self.reset_H_every == 0)

            # Prepare Gaussian input
            self._write_gaussian(curr_x, c_type="H" if recalc_H else "E")
            if self._sub_gaussian():
                raise RuntimeError(f"Force calculations failed. Check {self.log_dir}")
            os.system(f"formchk {self.chk_dir} {self.fchk_dir} > /dev/null 2>&1")

            # Store previous iteration values
            self.E_old = self.E
            self.G_old = self.G
            self.H_old = self.H

            # Update energy
            self._get_energy()

            if recalc_H:
                print("[EF-TSS] | Using full Hessian from freq calculation.", flush=True)
                self._get_grad()
                self._get_hessian()
            else:
                print("[EF-TSS] | Using Bofill Hessian estimation.", flush=True)
                self._get_g_estim()
                self._get_H_estim()

            # Eigen decomposition
            evals, U = self._eig_decomp_reduced_H()

            if iter == 1:
                print("[EF-TSS] | Initial Hessian eigenvalues:", evals.squeeze(), flush=True)
                while True: # input sanitation
                    try:
                        mode = int(input("[EF-TSS] | Select mode to follow: "))
                        if 0 <= mode < len(evals):
                            break
                        else:
                            print("[EF-TSS] | Mode index out of range.", flush=True)
                    except ValueError:
                        print("[EF-TSS] | Invalid input. Enter an integer.", flush=True)
                self.evec_mem.append(U[:, mode])
                if len(self.evec_mem) > self.evec_mem_size:
                    self.evec_mem.pop(0)
            else:
                # Follow the eigenvector most aligned with previous steps
                old_vec = np.sum(np.array(self.evec_mem), axis=0)
                old_vec /= np.linalg.norm(old_vec)

                overlaps = np.abs(U.T @ old_vec)
                mode = np.argmax(overlaps)
                aligned_vec = U[:, mode]
                if np.dot(aligned_vec, old_vec) < 0:
                    aligned_vec *= -1
                self.evec_mem.append(aligned_vec)
                if len(self.evec_mem) > self.evec_mem_size:
                    self.evec_mem.pop(0)
                print(f"[EF-TSS] | Following mode {mode} with eigenvalue {evals[mode]:.4e} and overlap {overlaps[mode]:.4f}", flush=True)

            evals_no_mode = np.delete(evals, mode)
            evals_no_mode = evals_no_mode[evals_no_mode != 0]

            if not recalc_H:
                # Bofill logic
                peri_H = self._get_priphery_H()
                mask = np.repeat(self.periphery != -1, 3)  # For x, y, z of each atom
                peri_G = self.G[0, mask]  # This extracts exactly the 3M entries for moving atoms
                try:
                    dx = -np.linalg.solve(peri_H, peri_G.T)
                except np.linalg.LinAlgError:
                    print("[EF-TSS] | Warning: peri_H is near-singular; using pseudoinverse.", flush=True)
                    dx = -np.linalg.pinv(peri_H) @ peri_G.T
                dx = self._get_padded_dx(dx)
                temp_G = self.G
                temp_E = self.E
                
                if (np.linalg.norm(dx) > self.R_trust or 
                    evals[mode] > 0 or 
                    np.sum(evals < 0) != 1):

                    try:
                        alphas = np.linalg.solve(U.T, self.G.T)
                    except np.linalg.LinAlgError:
                        print("[EF-TSS] | Warning: U.T is near-singular; falling back to pseudoinverse.", flush=True)
                        alphas = np.linalg.pinv(U.T) @ self.G.T
                    v_k_old = np.linalg.norm(self.G) / self.R_trust + max(-min(evals_no_mode, default=0), evals[mode])

                    while True:
                        delta_vk = (
                            -(alphas[mode]) / (evals[mode] - v_k_old + 1e-8) * U[:, mode]
                            - np.sum([(alphas[i]) / (evals[i] + v_k_old + 1e-8) * U[:, i] 
                                for i in range(3*self.N) if i != mode and evals[i] != 0],
                                axis=0)
                            )

                        delta_vk_p = (
                            (alphas[mode] ** 2) / (evals[mode] - v_k_old + 1e-8) ** 3
                            - np.sum([(alphas[i] ** 2) / (evals[i] + v_k_old + 1e-8) ** 3 
                                for i in range(3*self.N) if i != mode and evals[i] != 0])
                            ) / (np.linalg.norm(delta_vk)+1e-8)

                        v_k = v_k_old + (1 - np.linalg.norm(delta_vk) / (self.R_trust+1e-8)) * (np.linalg.norm(delta_vk) / (delta_vk_p + 1e-8))
                        if abs(v_k - v_k_old) < 1e-6:
                            if v_k <= max(evals[mode], -min(evals_no_mode, default=0)):
                                self.R_trust /= self.bofill_params["Sf"]
                                continue
                            else:
                                dx = delta_vk
                                temp_x = curr_x + dx.reshape(self.N, 3)
                                self._write_gaussian(temp_x, c_type="E")
                                if self._sub_gaussian():
                                    raise RuntimeError(f"Gaussian calculations failed. Check {self.log_dir}")

                                os.system(f"formchk {self.chk_dir} {self.fchk_dir} > /dev/null 2>&1")
                                temp_E = self._get_energy(inplace=False)
                                temp_G = self._get_grad(inplace=False)

                                q = self.E + temp_G @ dx[:, np.newaxis] + 0.5 * dx[:, np.newaxis].T @ self.H @ dx[:, np.newaxis]
                                denom = q - self.E
                                r_k = (temp_E - self.E) / denom if abs(denom) > 1e-6 else 1.0

                                if r_k < self.bofill_params['r_l'] or r_k > self.bofill_params['r_u']:
                                    self.R_trust /= self.bofill_params['Sf']
                                elif abs(np.linalg.norm(dx) - self.R_trust) < 1e-3:
                                    self.R_trust *= self.bofill_params['Sf'] ** 0.5

                                if r_k < self.bofill_params['Lb'] or r_k > self.bofill_params['Ub']:
                                    dx = np.zeros_like(self.dx)
                                    temp_G = self.G
                                    temp_E = self.E
                                    break
                                else:
                                    break
                        v_k_old = v_k

                self.dx = dx
                self.G = temp_G
                self.E = temp_E
            else:
                # Full Hessian logic
                gamma = U.T @ self.G.T
                ksi = self._get_ksi(evals, gamma, mode)
                self.dx = U @ ksi.T

                # Trust radius check
                dx_norm = np.linalg.norm(self.dx)
                if dx_norm > self.R_trust:
                    self.dx = self.R_trust * self.dx / dx_norm

            # Logging and convergence
            iter_dx_size = np.linalg.norm(self.dx)
            iter_G_size = np.linalg.norm(self.G)
            print(f"[EF-TSS] | Iteration {iter}:\tdx = {iter_dx_size:.4e}\tgrad = {iter_G_size:.4e}", flush=True)
            self._write_history(curr_x, self.hist_file)

            if iter_dx_size <= self.R_conv:
                print("[EF-TSS] | R_conv satisfied. Writing the final structure to final.xyz", flush=True)
                break
            if iter_G_size <= self.G_conv:
                print("[EF-TSS] | G_conv satisfied. Writing the final structure to final.xyz", flush=True)
                break
            if iter >= self.max_iter:
                print("[EF-TSS] | Max_iter reached. Writing the final structure to final.xyz", flush=True)
                break

            curr_x += self.dx.reshape(self.N, 3)
            iter += 1

        self._write_history(curr_x, self.final_file)
        return curr_x
