import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import PyMoosh as pm


def setup_structure(thick_list, mat_env, mat1, mat2):
    """helper to create pymoosh structure object, alternating 2 materials

    Args:
        thick_list (list): list of thicknesses, top layer first
        mat_env (float): environment ref. index
        mat1 (float): material 1 ref. index
        mat2 (float): material 2 ref. index

    Returns:
        PyMoosh.structure: multi-layer structure object
    """
    thick_list = list(
        thick_list
    )  # convert to list for convenience when stacking layers
    n = len(thick_list)

    materials = [mat_env**2, mat1**2, mat2**2]  # permittivities!
    # periodic stack. first layer: environment, last layer: substrate
    stack = [0] + [2, 1] * (n // 2) + [2]
    thicknesses = [0.0] + thick_list + [0.0]

    structure = pm.Structure(materials, stack, np.array(thicknesses), verbose=False)

    return structure


def objective_f(x):
    target_wl = 600.0  # nm
    mat_env = 1.0  # materials: ref. index
    mat1 = 1.4
    mat2 = 1.8
    return cost_minibragg(x, mat_env, mat1, mat2, target_wl)


# ------- the optimization target function -------
def cost_minibragg(x, mat_env, mat1, mat2, eval_wl):
    """cost function: maximize reflectance of a layer-stack

    Args:
        x (list): thicknesses of all the layers, starting with the upper one.

    Returns:
        float: 1 - Reflectivity at target wavelength
    """
    structure = setup_structure(x, mat_env, mat1, mat2)

    # the actual PyMoosh reflectivity simulation
    _, R = pm.coefficient_I(structure, eval_wl, 0.0, 0)
    cost = 1 - R

    return cost


def upper_lower_bound():
    nb_layers = 10  # number of layers of full stack
    target_wl = 600.0  # nm
    mat_env = 1.0  # materials: ref. index
    mat1 = 1.4
    mat2 = 1.8
    min_thick = 0  # no negative thicknesses
    max_thick = target_wl / (2 * mat1)  # lambda/2n
    return min_thick, max_thick


def old():
    # ------- define "mini-bragg" optimization problem
    nb_layers = 10  # number of layers of full stack
    target_wl = 600.0  # nm
    mat_env = 1.0  # materials: ref. index
    mat1 = 1.4
    mat2 = 1.8
    min_thick = 0  # no negative thicknesses
    max_thick = target_wl / (2 * mat1)  # lambda/2n

    # ------- setup the parametrization
    budget = 50000  # stop criterion: allowed number of evaluations
    N_population = 30  # population size (DE algo-specific!)

    # parametrization: define the free parameters
    init = [
        min_thick + (max_thick - min_thick) * np.random.rand() for _ in range(nb_layers)
    ]
    args_geo_ng = ng.p.Array(
        init=init,
        lower=min_thick,  # lower and upper bounds for the parameters
        upper=max_thick,
    )

    # wrap free and fixed arguments
    instrumentation = ng.p.Instrumentation(
        # --- optimization args (multilayer-geometry)
        x=args_geo_ng,
        # --- additional, fixed args
        mat_env=mat_env,
        mat1=mat1,
        mat2=mat2,
        eval_wl=target_wl,
    )

    return instrumentation, budget


# ------- setup the optimizer
# # configure specific variant from DE optimizer family
# optim_algos_DE = ng.families.DifferentialEvolution(
#     crossover="twopoints", popsize=N_population)

# initialize the optimizer


# optimizer = optim_algos_DE(instrumentation, budget=budget)
