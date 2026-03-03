import numpy as np
import PyMoosh as pm
from .photonic_problem import photonic_problem


class brag_mirror(photonic_problem):
    def __init__(self, nb_layers, target_wl, mat_env, mat1, mat2):
        """
        Args:
            mat_env (float): environment ref. index
            mat1 (float): material 1 ref. index
            mat2 (float): material 2 ref. index
        """
        super().__init__()
        self.n = nb_layers
        self.nb_layers = nb_layers
        self.target_wl = target_wl
        self.mat_env = mat_env
        self.mat1 = mat1
        self.mat2 = mat2
        self.min_thick = 0
        self.max_thick = target_wl / (2 * mat1)
        self.lb = np.array([self.min_thick] * self.n)
        self.ub = np.array([self.max_thick] * self.n)

    def setup_structure(self, x):
        """helper to create pymoosh structure object, alternating 2 materials

        Args:
            x (list): list of thicknesses, top layer first

        Returns:
            PyMoosh.structure: multi-layer structure object
        """
        x = list(x)
        # n = len(x)
        materials = [
            self.mat_env**2,
            self.mat1**2,
            self.mat2**2,
        ]  # permittivities!
        # periodic stack. first layer: environment, last layer: substrate
        stack = [0] + [2, 1] * (self.n // 2) + [2]
        thicknesses = [0.0] + x + [0.0]
        structure = pm.Structure(materials, stack, np.array(thicknesses), verbose=False)
        return structure

    def __call__(self, x):
        """cost function: maximize reflectance of a layer-stack

        Args:
            x (list): thicknesses of all the layers, starting with the upper one.

        Returns:
            float: 1 - Reflectivity at target wavelength
        """
        x = np.clip(x, self.lb, self.ub)
        structure = self.setup_structure(x)
        # the actual PyMoosh reflectivity simulation
        _, R = pm.coefficient_I(structure, self.target_wl, 0.0, 0)
        cost = 1 - R

        return cost
