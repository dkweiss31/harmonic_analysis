from abc import ABC, abstractmethod
import dynamiqs as dq
import scipy as sp
import numpy as np
from numpy import ndarray


class HarmonicAnalysis(ABC):
    def __init__(self, num_modes, mode_dim):
        self.num_modes = num_modes
        self.mode_dim = mode_dim
        self.dim_list = self.num_modes * [self.mode_dim]
        # units where e = hbar = 1
        self.Phi0 = 0.5  # Phi0 = hbar / (2 * e)
        self.Z0 = 0.25  # Z0 = hbar / (2 * e)**2

    @abstractmethod
    def gamma_matrix(self) -> ndarray:
        """Returns linearized potential matrix

        Note that we must divide by Phi_0^2 since Ej/Phi_0^2 = 1/Lj,
        or one over the effective impedance of the junction.
        """
        pass

    @abstractmethod
    def capacitance_matrix(self) -> ndarray:
        pass

    @abstractmethod
    def potential_matrix(self) -> ndarray:
        pass

    def hamiltonian(self):
        return self.kinetic_matrix() + self.potential_matrix()

    def EC_matrix(self) -> ndarray:
        """Returns the charging energy matrix"""
        return 0.5 * sp.linalg.inv(self.capacitance_matrix())

    def eigensystem_normal_modes(self) -> (ndarray, ndarray):
        """Returns squared normal mode frequencies, matrix of eigenvectors"""
        omega_squared, normal_mode_eigenvectors = sp.linalg.eigh(
            self.gamma_matrix(), b=self.capacitance_matrix()
        )
        return omega_squared, normal_mode_eigenvectors

    def Xi_matrix(self) -> ndarray:
        """Returns Xi matrix of the normal-mode eigenvectors normalized
        according to \Xi^T C \Xi = \Omega^{-1}/Z0, or equivalently \Xi^T
        \Gamma \Xi = \Omega/Z0. The \Xi matrix
        simultaneously diagonalizes the capacitance and effective
        inductance matrices by a congruence transformation.
        """
        omega_squared_array, eigenvectors = self.eigensystem_normal_modes()
        normalization_factors = omega_squared_array ** (-1 / 4) / np.sqrt(self.Z0)
        return eigenvectors * normalization_factors

    def a_operator(self, mode_index: int) -> ndarray:
        """Returns the lowering operator associated
        with the mu^th d.o.f. in the full Hilbert space

        Parameters
        ----------
        mode_index: int
            which degree of freedom, 0<=dof_index<=self.number_degrees_freedom
        """
        if self.num_modes == 1:
            return dq.destroy(self.mode_dim)
        return np.asarray(dq.destroy(*self.dim_list)[mode_index])

    def n_j(self, node_index: int) -> ndarray:
        """Charge number operator of a node, expressed in the mode basis."""
        Xi = self.Xi_matrix()
        Xi_inv_T = sp.linalg.inv(Xi).T
        a_ops = [self.a_operator(mode_index) for mode_index in range(self.num_modes)]
        a_minus_a_dag = np.stack([a - a.T for a in a_ops])
        return np.einsum("i,ijk->jk", Xi_inv_T[node_index], -1j * a_minus_a_dag) / np.sqrt(2)

    def phi_j(self, node_index: int) -> ndarray:
        """Position/phase operator of a node, expressed in the mode basis."""
        Xi = self.Xi_matrix()
        a_ops = [self.a_operator(mode_index) for mode_index in range(self.num_modes)]
        a_plus_a_dag = np.stack([a + a.T for a in a_ops])
        return np.einsum("i,ijk->jk", Xi[node_index], a_plus_a_dag) / np.sqrt(2)

    def exp_i_phi_j(self, node_index) -> ndarray:
        r"""$\exp(i\phi_{j})$ operator expressed in the mode basis."""
        return sp.linalg.expm(1j * self.phi_j(node_index))

    def kinetic_matrix(self) -> ndarray:
        """Kinetic energy matrix."""
        EC_mat = self.EC_matrix()
        n_ops = np.stack([self.n_j(node_index) for node_index in range(self.num_modes)])
        return np.einsum("imn,ij,jnl->ml", n_ops, 4.0 * EC_mat, n_ops)

    def eigenvals(self, evals_count: int = 6) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = sp.linalg.eigh(
            hamiltonian_mat, eigvals_only=True, subset_by_index=(0, evals_count - 1)
        )
        return np.sort(evals)

    def eigensys(self, evals_count: int = 6) -> tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(
            hamiltonian_mat, eigvals_only=False, subset_by_index=(0, evals_count - 1)
        )
        ordered_evals_indices = evals.argsort()
        evals = evals[ordered_evals_indices]
        evecs = evecs[:, ordered_evals_indices]
        return evals, evecs

    def bare_labels(self):
        return list(np.ndindex(*self.dim_list))

    def _overlaps(self, evecs, bare_indices):
        bare_states = np.asarray(dq.basis(self.dim_list, bare_indices))
        return np.einsum("ji,bjd->ib", evecs, bare_states)

    def get_bare_indices(self, evecs: ndarray):
        """Get bare labels of the eigenvectors evecs.

        Assumption is that `evecs` are column vectors as returned by `eigensys` and
        `eigh`
        """
        overlaps = self._overlaps(evecs, self.bare_labels())
        max_idxs = np.argmax(np.abs(overlaps), axis=1).astype(int)
        return np.array(self.bare_labels())[max_idxs]

    def get_dressed_indices(self, evecs: ndarray, bare_indices: ndarray):
        overlaps = self._overlaps(evecs, bare_indices)
        return np.argmax(np.abs(overlaps), axis=0).astype(int)


class Transmon(HarmonicAnalysis):
    def __init__(self, EJ, EC, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.EJ = EJ
        self.EC = EC

    def capacitance_matrix(self) -> ndarray:
        return np.array([[1. / (2 * self.EC)]])

    def gamma_matrix(self) -> ndarray:
        return np.array([[self.EJ]]) / self.Phi0 ** 2

    def potential_matrix(self) -> ndarray:
        exp_phi = self.exp_i_phi_j(0)
        return -0.5 * self.EJ * (exp_phi + np.conj(exp_phi).T)


class Dimon(HarmonicAnalysis):
    def __init__(self, EJ1, EJ2, ECJ1, ECJ2, ECs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.ECJ1 = ECJ1
        self.ECJ2 = ECJ2
        self.ECs = ECs

    def capacitance_matrix(self) -> ndarray:
        C1 = 1. / (2 * self.ECJ1)
        C2 = 1. / (2 * self.ECJ2)
        Cs = 1. / (2 * self.ECs)
        return np.array([[C1 + Cs, -Cs],
                         [-Cs, C2 + Cs]])

    def gamma_matrix(self) -> ndarray:
        return np.array([[self.EJ1, 0.0],
                         [0.0, self.EJ2]]) / self.Phi0 ** 2

    def potential_matrix(self) -> ndarray:
        exp_phi_1 = self.exp_i_phi_j(0)
        exp_phi_2 = self.exp_i_phi_j(1)
        return -0.5 * (
            self.EJ1 * (exp_phi_1 + np.conj(exp_phi_1).T)
            + self.EJ2 * (exp_phi_2 + np.conj(exp_phi_2).T)
        )
