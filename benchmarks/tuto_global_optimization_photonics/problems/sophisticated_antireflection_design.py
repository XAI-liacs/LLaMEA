import numpy as np
import PyMoosh as pm
from .photonic_problem import photonic_problem


class sophisticated_antireflection_design(photonic_problem):
    def __init__(
        self,
        nb_layers,
        min_thick,
        max_thick,
        wl_min,
        wl_max,
        thick_aSi=30000,
        number_pts=300,
        pola=0,
        incidence=0,
    ):
        super().__init__()
        self.n = nb_layers
        self.nb_layers = nb_layers
        self.min_thick = min_thick
        self.max_thick = max_thick
        self.lb = np.array([self.min_thick] * self.n)
        self.ub = np.array([self.max_thick] * self.n)
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.thick_aSi = thick_aSi
        self.number_pts = number_pts
        self.pola = pola
        self.incidence = incidence

    def setup_structure(self, x):
        """helper to create pymoosh structure object, alternating 2 materials

        the substrate is amorphous silicon and the light is incident through air (n=1).
        The structure is made of alternating layers of eps=2 and eps=3.

        Args:
            X (list): long list of thicknesses

        Returns:
            PyMoosh.structure: multi-layer structure object
        """
        x = list(x)
        # available materials (alternating eps=2 and eps=3)
        materials = [1.0, 2.0, 3.0, "SiA"]
        # material sequence of layer-stack
        stack = [0] + [1, 2] * (self.n // 2) + [3]
        # thicknesses of layers
        thicknesses = [0] + x + [self.thick_aSi]
        structure = pm.Structure(materials, stack, np.array(thicknesses), verbose=False)
        return structure

    def __call__(self, x):
        """cost function: (negative) efficiency of solar cell

        Args:
            x (list): materials (first half) & thicknesses (second half) of all
                layers
            wl_min, wl_max (float): spectral limits of efficiency evaluation

        Returns:
            float: 1 - Reflectivity at target wavelength
        """
        x = np.clip(x, self.lb, self.ub)
        structure = self.setup_structure(x)
        # the actual PyMoosh reflectivity simulation
        active_lay = len(x) + 1
        eff, _, _, _, _, _ = pm.photo(
            structure,
            self.incidence,
            self.pola,
            self.wl_min,
            self.wl_max,
            active_lay,
            self.number_pts,
        )
        cost = 1 - eff
        return cost
