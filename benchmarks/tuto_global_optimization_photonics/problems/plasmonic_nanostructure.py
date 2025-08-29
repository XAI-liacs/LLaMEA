import warnings
import numpy as np
from pyGDM2 import structures
from pyGDM2 import linear
from pyGDM2 import tools
from pyGDM2 import visu
from .photonic_problem import photonic_problem


class plasmonic_nanostructure(photonic_problem):
    def __init__(self, element_sim, method, verbose=0):
        """
        Args:
            mat_env (float): environment ref. index
            mat_substrate (str): substrate material
            nb_layers (int): number of layers
            min_thick (float): minimum thickness
            max_thick (float): maximum thickness
            min_eps (float): minimum permittivity
            max_eps (float): maximum permittivity
            wavelengths (np.array): wavelengths
            angle (float): angle
        """
        super().__init__()
        self.n = 20
        self.element_sim = element_sim
        self.method = method
        self.verbose = verbose

    def setup_structure(self, x):
        """helper to create structure, from positions of gold elements
        each positions in units of discretization steps

        Args:
            XY_coords_blocks (list): list gold element positions (x1,x2,x3,...,y1,y2,....)
            element_sim (`pyGDM2.core.simulation`): single element simulation

        Returns:
            pyGDM2.structures.struct: instance of nano-geometry class
        """
        x_new = x * 5.0
        n = len(x_new) // 2
        x_list = x_new[:n]
        y_list = x_new[n:]
        pos = np.transpose([x_list, y_list])

        struct_list = []
        for _p in pos:
            x_new, y = _p
            # displace by steps of elementary block-size
            _s = self.element_sim.struct.copy()
            DX = _s.geometry[:, 0].max() - _s.geometry[:, 0].min() + _s.step
            DY = _s.geometry[:, 1].max() - _s.geometry[:, 1].min() + _s.step
            _s = structures.shift(_s, np.array([DX * int(x_new), DY * int(y), 0.0]))

            # do not add the block if too close to emitter at (0,0)
            if np.abs(x_new) >= 1 or np.abs(y) >= 1:
                struct_list.append(_s)

        if len(struct_list) == 0:
            struct_list.append(_s + [DX, DY, 0])  # add at least one block

        full_struct = structures.combine_geometries(
            struct_list, step=self.element_sim.struct.step
        )
        full_sim = self.element_sim.copy()
        full_sim.struct = full_struct
        return full_sim

    # ------- the optimization target function -------

    def __call__(self, x):
        """cost function: maximize scattering towards small solid angle

        Args:
            x (list): optimization params --> pos of elements
            element_sim (`pyGDM2.core.simulation`): single element simulation
            method (str): pyGDM2 solver method

        Returns:
            float: 1 - Reflectivity at target wavelength
        """
        sim = self.setup_structure(x)
        sim.scatter(method=self.method, verbose=self.verbose)

        # 2D scattering evaluation in upper hemisphere
        warnings.filterwarnings("ignore")
        Nteta, Nphi = 18, 32
        NtetaW, NphiW = 4, 5
        Delta_angle = np.pi * 10 / 180  # +/- 10 degrees target angle
        I_full = linear.farfield(
            sim,
            field_index=0,
            return_value="int_Etot",
            phimin=0,
            phimax=2 * np.pi,
            tetamin=0,
            tetamax=np.pi / 2,
            Nteta=Nteta,
            Nphi=Nphi,
        )
        I_window = linear.farfield(
            sim,
            field_index=0,
            return_value="int_Etot",
            # supposed to start at zero, excluding last point
            phimin=-np.pi / 6,
            phimax=np.pi / 6 + (np.pi / 3) / NphiW,
            tetamin=np.pi / 2 - Delta_angle,
            tetamax=np.pi / 2 + Delta_angle,
            Nteta=NtetaW,
            Nphi=NphiW,
        )

        cost = -1 * (I_window / I_full)
        if self.verbose:
            print("cost: {:.5f}".format(cost))

        return cost
