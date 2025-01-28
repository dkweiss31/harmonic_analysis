import numpy as np
import scqubits as scq
import harmonic_analysis as ha


def test_correctness():
    EJ = 20.0
    ECJ = 0.8
    ECs = 0.8
    evals_count = 10
    harmonic_dimon = ha.Dimon(EJ1=EJ, EJ2=EJ, ECJ1=ECJ, ECJ2=ECJ, ECs=ECs, num_modes=2, mode_dim=12)
    evals = harmonic_dimon.eigenvals(evals_count=evals_count)
    evals = evals - evals[0]
    dimon_yaml = f"""# dimon
            branches:
            - ["JJ", 0,1, EJ={EJ}, ECJ={ECJ}]
            - ["JJ", 0,2, {EJ}, {ECJ}]
            - ["C", 1,2, ECs={ECs}]
            """
    dimon = scq.Circuit(dimon_yaml, from_file=False)
    dimon.cutoff_n_1 = 21
    dimon.cutoff_n_2 = 21
    true_evals = dimon.eigenvals(evals_count=evals_count)
    true_evals = true_evals - true_evals[0]
    assert np.allclose(evals, true_evals, atol=1e-2, rtol=1e-2)
