import numpy as np
import scqubits as scq
import harmonic_analysis as ha


def test_correctness():
    EJ = 20.0
    EC = 0.2
    harmonic_tmon = ha.Transmon(EJ=EJ, EC=EC, num_modes=1, mode_dim=12)
    evals = harmonic_tmon.eigenvals(evals_count=5)
    evals = evals - evals[0]
    scq_tmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.0, ncut=21)
    true_evals = scq_tmon.eigenvals(evals_count=5)
    true_evals = true_evals - true_evals[0]
    assert np.allclose(evals, true_evals, atol=1e-2, rtol=1e-2)
