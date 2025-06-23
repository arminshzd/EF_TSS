import pytest
import numpy as np
import json
from unittest.mock import patch, MagicMock

@pytest.fixture
def minimal_json(tmp_path):
    data = {
        "N": 2,
        "working-dir": str(tmp_path),
        "submit-f-dir": "dummy_submit"
    }
    json_path = tmp_path / "settings.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path

@pytest.fixture
def full_json(tmp_path):
    data = {
        "N": 2,
        "working-dir": str(tmp_path),
        "submit-f-dir": "dummy_submit",
        "charge": 0,
        "spin": 1,
        "N-procs": 4,
        "conv-radius": 0.01,
        "conv-grad": 1e-5,
        "max-iter": 5,
        "reset-H-every": 2,
        "trust-radius": 0.1,
        "basis-f-name": "",
        "history-f-name": "hist",
        "final-f-name": "fin",
        "gaussian-f-name": "calc",
        "energy-header-calc": "#P test force",
        "hess-header-calc": "#P test freq"
    }
    json_path = tmp_path / "full_settings.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path

@pytest.fixture
def initial_structure(tmp_path):
    path = tmp_path / "structure.txt"
    content = "8 0 0.0 0.0 0.0\n1 0 0.0 0.0 1.0\n"
    with open(path, "w") as f:
        f.write(content)
    return path

def test_minimal_initialization(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    assert ef.N == 2
    assert ef.charge == 0
    assert ef.spin == 1
    assert ef.N_procs == 8
    assert ef.max_iter == 10

def test_full_initialization(full_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(full_json, initial_structure)
    assert ef.N_procs == 4
    assert ef.R_trust == 0.1
    assert ef.energy_calc_header == "#P test force"

def test_atom_parsing(initial_structure, minimal_json):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    assert np.array_equal(ef.periphery, np.array([0, 0]))
    assert np.array_equal(ef.atom_types, np.array([8, 1]))
    assert ef.init_coords.shape == (2, 3)

def test_get_padded_dx(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    dx = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    padded = ef._get_padded_dx(dx)
    assert padded.shape == (6,)
    assert np.allclose(padded[:6], dx)

def test_write_history(tmp_path, minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    hist_file = tmp_path / "test.xyz"
    ef._write_history(coords, hist_file)
    with open(hist_file) as f:
        lines = f.readlines()
    assert len(lines) == 4  # N=2 atoms = 2 header lines + 2 atom lines

def test_get_lambda_RC():
    from ef_tss.EF_TSS import EF_TSS
    dummy = EF_TSS.__new__(EF_TSS)  # bypass __init__
    result = dummy._get_lambda_RC(1.0, -0.5)
    assert isinstance(result, float)

def test_get_ksi():
    from ef_tss.EF_TSS import EF_TSS
    dummy = EF_TSS.__new__(EF_TSS)
    dummy._get_lamda_bath = lambda gamma, evals, mode: 0.1
    dummy._get_lambda_RC = lambda gamma_k, eval_k: 0.5
    gamma = np.array([1.0, 0.0, -1.0])
    evals = np.array([-0.5, 0.1, 0.2])
    ksi = dummy._get_ksi(evals, gamma, mode=0)
    assert ksi.shape == (1, 3)

def test_grad_fallback_estimate(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    ef.E = 10.0
    ef.E_old = 9.5
    ef.dx = np.ones((1, 6)) * 0.1
    ef._get_g_estim()
    mask = np.repeat(ef.periphery != -1, 3)  # (N,) → (3N,) mask for x, y, z
    assert np.all(ef.G[0][mask] != 0)

def test_dx_trust_radius_scaling(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    ef.R_trust = 0.5
    dx = np.array([10.0] * 6)
    ef.dx = dx
    norm = np.linalg.norm(dx)
    if norm > ef.R_trust:
        scaled = ef.R_trust * dx / norm
        assert np.allclose(np.linalg.norm(scaled), ef.R_trust)

def test_eigenvector_mode_alignment_logic():
    from ef_tss.EF_TSS import EF_TSS
    dummy = EF_TSS.__new__(EF_TSS)
    U = np.eye(6)
    dummy.N = 2
    dummy.evec_mem = [np.array([1.0, 0, 0, 0, 0, 0])]
    old_vec = np.sum(dummy.evec_mem, axis=0)
    old_vec /= np.linalg.norm(old_vec)
    overlaps = np.abs(U.T @ old_vec)
    mode = np.argmax(overlaps)
    assert mode == 0

def test_frozen_atom_gradients_zero(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    ef.periphery = np.array([-1, 0])
    ef.num_moving_atoms = 3*(ef.N - (-sum(ef.periphery)))
    ef.dx = np.ones((1, 6)) * 0.1
    ef.E = 5.0
    ef.E_old = 4.5
    ef._get_g_estim()
    # x,y,z of atom 0 (frozen) should be 0
    assert np.allclose(ef.G[0][:3], 0.0)

@patch("builtins.input", return_value="0")
def test_mode_selection_input_sanitization(mock_input, minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    ef._write_gaussian = lambda *args, **kwargs: None
    ef._sub_gaussian = lambda: False
    ef._get_energy = lambda: None
    ef._get_grad = lambda: None
    ef._get_hessian = lambda: None
    ef._get_g_estim = lambda: None
    ef._get_H_estim = lambda: None
    ef._write_history = lambda *args, **kwargs: None
    ef._get_ksi = lambda *args, **kwargs: np.zeros((1, 6))
    ef.chk_dir = ef.fchk_dir = ef.log_dir = "/dev/null"
    ef.init_coords = np.zeros((2, 3))
    ef.periphery = np.array([0, 0])
    ef.N = 2
    ef.R_conv = ef.G_conv = 1e-5
    ef.max_iter = 1
    ef.E = ef.E_old = 0
    ef.G = np.zeros((1, 6))
    ef.H = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    result = ef.run()
    assert result.shape == (2, 3)

@patch("ef_tss.EF_TSS.subprocess.run")
def test_sub_gaussian_success(mock_subproc_run, minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)

    with patch("ef_tss.EF_TSS.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.readlines.return_value = ["Some line", "Normal termination"]
        mock_open.return_value.__enter__.return_value = mock_file
        result = ef._sub_gaussian()
        assert result == 0

def test_get_priphery_H(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    ef.periphery = np.array([1, -1])  # Only first atom is moving
    ef.H = np.eye(6)
    reduced = ef._get_priphery_H()
    assert reduced.shape == (3, 3)
    assert np.allclose(reduced, np.eye(3))

def test_eig_decomp_reduced_H(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS

    ef = EF_TSS(minimal_json, initial_structure)

    # Set up a Hessian with known eigenvalues for the movable atoms
    full_H = np.zeros((6, 6))
    reduced_block = np.diag([-1.0, 2.0, 3.0])
    full_H[:3, :3] = reduced_block  # Assume atom 0 is movable

    ef.H = full_H
    ef.periphery = np.array([0, -1])  # Only atom 0 is movable

    evals, U_full = ef._eig_decomp_reduced_H()

    assert evals.shape == (3,)
    assert U_full.shape == (6, 3)

    # Check that the padded eigenvectors are non-zero for movable atoms and zero elsewhere
    assert not np.allclose(U_full[:3, :], 0)
    assert np.allclose(U_full[3:, :], 0)

def test_get_energy_not_inplace(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)

    ef.fchk_dir = "/dummy/path"

    mock_file = MagicMock()
    mock_file.readlines.return_value = ["Some line", "SCF Energy  =     -123.456", "Another line"]

    with patch("ef_tss.EF_TSS.open", new_callable=MagicMock) as mock_open_func:
        mock_open_func.return_value.__enter__.return_value = mock_file

        ef.E = 0.0
        energy = ef._get_energy(inplace=False)

    assert energy == -123.456
    assert ef.E == 0.0  # Confirm energy was not updated in-place

def test_get_ksi_zero_gradient():
    from ef_tss.EF_TSS import EF_TSS
    dummy = EF_TSS.__new__(EF_TSS)
    dummy._get_lamda_bath = lambda g, e, m: 1.0
    dummy._get_lambda_RC = lambda gk, ek: 1.0
    gamma = np.zeros(3)
    evals = np.array([-0.5, 0.1, 0.2])
    ksi = dummy._get_ksi(evals, gamma, mode=0)
    assert np.allclose(ksi, 0.0)

def test_invalid_json_raises(tmp_path, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    bad_path = tmp_path / "bad.json"
    with open(bad_path, "w") as f:
        f.write("not json")
    with pytest.raises(json.JSONDecodeError):
        EF_TSS(bad_path, initial_structure)

def test_zero_displacement_in_get_g_estim(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS
    ef = EF_TSS(minimal_json, initial_structure)
    ef.E = 10.0
    ef.E_old = 9.5
    ef.dx = np.zeros((1, 6))
    ef._get_g_estim()
    print(ef.G)
    assert np.allclose(ef.G, 0.0)

def test_gradient_direction_preservation(minimal_json, initial_structure):
    from ef_tss.EF_TSS import EF_TSS

    ef = EF_TSS(minimal_json, initial_structure)
    
    # Set up known displacement direction
    known_grad_dir = np.array([1.054, 2.043, -1.033, 0.523, -0.565, 1.544])
    known_grad_dir /= np.linalg.norm(known_grad_dir)  # normalize

    # Use small displacement in known direction
    epsilon = 1e-4
    ef.dx = epsilon/known_grad_dir.reshape(1, -1)
    ef.E_old = 0.0
    ef.E = epsilon

    ef._get_g_estim()
    est_grad = ef.G.flatten()

    # Only compare over movable atoms
    mask = np.repeat(ef.periphery != -1, 3)
    est_grad = est_grad[mask]

    # Normalize estimated gradient
    est_grad /= np.linalg.norm(est_grad)

    # Check cosine similarity close to 1 or -1
    cosine_sim = np.dot(est_grad, known_grad_dir[mask])
    assert np.isclose(np.abs(cosine_sim), 1.0, atol=1e-2)
