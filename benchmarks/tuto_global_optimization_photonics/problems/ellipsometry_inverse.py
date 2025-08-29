import numpy as np
import PyMoosh as pm
from .photonic_problem import photonic_problem


class ellipsometry(photonic_problem):
    def __init__(
        self,
        mat_env,
        mat_substrate,
        nb_layers,
        min_thick,
        max_thick,
        min_eps,
        max_eps,
        wavelengths,
        angle,
    ):
        self.n = nb_layers * 2
        self.mat_env = mat_env
        self.mat_substrate = mat_substrate
        self.nb_layers = nb_layers
        self.min_thick = min_thick
        self.max_thick = max_thick
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.wavelengths = wavelengths
        self.angle = angle
        self.lb = np.array(
            [self.min_eps] * self.nb_layers + [self.min_thick] * self.nb_layers
        )
        self.ub = np.array(
            [self.max_eps] * self.nb_layers + [self.max_thick] * self.nb_layers
        )
        x_ref = np.random.uniform(self.lb, self.ub, self.n)
        struct_ref = self.setup_structure(x_ref)
        self.ref_ellipso = np.zeros(len(self.wavelengths), dtype=complex)
        for i, wav in enumerate(self.wavelengths):
            r_s, _, _, _ = pm.coefficient(struct_ref, wav, self.angle, 0)
            r_p, _, _, _ = pm.coefficient(struct_ref, wav, self.angle, 1)
            self.ref_ellipso[i] = r_p / r_s

    def setup_structure(self, x):
        """helper to create pymoosh structure object with user-defined
        thicknesses and materials

        Args:
            X (list): long list of material permittivities and thicknesses
                (first half / second half)
            mat_env (float, str): material of environment (above stack)
            mat_substrate (float, str): material of substrate (below stack)

        Returns:
            PyMoosh.structure: multi-layer structure object
        """
        x = list(x)
        # available materials
        materials = (
            [self.mat_env] + [_m for _m in x[: self.nb_layers]] + [self.mat_substrate]
        )
        # material sequence of layer-stack
        stack = [i for i in range(self.nb_layers + 2)]
        # thicknesses of layers
        thicknesses = np.array([0] + [_t for _t in x[self.nb_layers :]] + [0])
        structure = pm.Structure(materials, stack, np.array(thicknesses), verbose=False)
        return structure

    def __call__(self, x):
        """cost function: MAE between simulated and measured (ref) spectrum

        Args:
            x (list): materials (first half) & thicknesses (second half) of all
                layers
            ref_ellipso (np.ndarray): reference spectrum at `eval_wls`
            wavelengths (np.ndarray): wavelengths to evaluate

        Returns:
            float: Reflectivity at target wavelength
        """
        x = np.clip(x, self.lb, self.ub)
        structure = self.setup_structure(x)
        # the actual PyMoosh reflectivity simulation
        ellips = np.zeros(len(self.wavelengths), dtype=complex)
        for i, wav in enumerate(self.wavelengths):
            r_s, _, _, _ = pm.coefficient(structure, wav, self.angle, 0)
            r_p, _, _, _ = pm.coefficient(structure, wav, self.angle, 1)
            ellips[i] = r_p / r_s
        # diff = ellips - ref_ellipso
        cost = np.sum(np.abs(ellips - self.ref_ellipso))
        return cost
