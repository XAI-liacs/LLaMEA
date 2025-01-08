# !pip install pymoosh
# !pip install nevergrad==1.0.0

# Let us get rid of some deprecation warning in SkLearn.
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


# Helper function.
def doint(s):  # Converting a string into an int.
   return 7 + sum([ord(c)*i for i, c in enumerate(s)])

def get_color(o):  # Converting a string into a color.
  colors = ['b', 'g', 'r', 'c', 'y', 'k']
  algs = ["CMA", "QODE", "BFGS", "QNDE", "DE"]
  if o in algs:
      return colors[algs.index(o)]
  return colors[doint(o) % len(colors)]


# Please modify the parameters below at your best convenience.
maximum_time_per_run_in_seconds = 6000  # Maximum number of seconds per run (then: interruption).
num_experiments = 1  # Number of times we reproduce each experiments.
maximum_optimization_overhead = 50  # Maximum ratio between the computational cost and the computational cost of the objective function only.
optims_id = int(sys.argv[1])  # The optimization id is used to generate the output file.
run_id = int(sys.argv[2])  # The run id is used to generate the output file.
list_optims_choice = ["CMA", "QODE", "BFGS", "QNDE", "DE"]
list_optims = [list_optims_choice[optims_id]]  # The list of optimization methods we want to compare.

Init = -7.0
Grad = True
Full = False
Para = False
NoverD = 10
Factor = 0


# List of optimization methods in the extended setup.
# list_optims = ["ChainMetaModelSQP", "MetaModel", "SQP", "CMA", "DE", "RotatedTwoPointsDE", "OnePlusOne", "QODE", "QNDE", "TwoPointsDE", "PSO", "GeneticDE", "NelderMead", "Cobyla" ,"Powell"]
# list_optims = ["CMA", "QODE", "BFGS", "QNDE", "DE"] # , "QODE", "BFGS", "QNDE", "DE"
#list_optims = ["NGOpt", "NGOptRW", "DE", "QODE", "QNDE", "BFGS", "GradBFGS", "CMA"]
# if you want more, you might add:  list_optims += ["BayesOptimBO", "PCABO", "RCobyla", "Shiwa", "CMandAS2", "NGOpt", "NGOptRW"]


# Choice of objective function. List of possibilities readable just below.
obj_name = "photovoltaics"
assert obj_name in ["bragg", "photovoltaics", "bigbragg", "bigphotovoltaics", "ellipsometry", "minibragg", "hugephotovoltaics"]
run_performance = True  # Whether we want to run the comparison between various optimization methods.

if obj_name == "bragg":
  nb_layers = 20
  opti_wave = 600
  mat1 = 1.4
  mat2 = 1.8
  min_th = 0 # We don't want negative thicknesses.
  max_th = opti_wave/(2*mat1) # A thickness of lambda/2n + t has the same behaviour as a thickness t

elif obj_name == "bigbragg":
  nb_layers = 40
  opti_wave = 600
  mat1 = 1.4
  mat2 = 1.8
  min_th = 0 # We don't want negative thicknesses.
  max_th = opti_wave/(2*mat1) # A thickness of lambda/2n + t has the same behaviour as a thickness t

elif obj_name == "ellipsometry":
  nb_layers = 1
  min_th = 50
  max_th = 150

elif obj_name == "photovoltaics":
  nb_layers = 10
  min_th = 30
  max_th = 250

elif obj_name == "bigphotovoltaics":
  nb_layers = 20
  min_th = 30
  max_th = 250

elif obj_name == "minibragg":
  nb_layers = 10
  opti_wave = 600
  mat1 = 1.4
  mat2 = 1.8
  min_th = 0 # We don't want negative thicknesses.
  max_th = opti_wave/(2*mat1) # A thickness of lambda/2n + t has the same behaviour as a thickness t

elif obj_name == "hugephotovoltaics":
  nb_layers = 32
  min_th = 30
  max_th = 250

else:
  assert False, f"Unknown objective function {obj_name}"

dim = 2 * nb_layers if "ellipsometry" in obj_name else nb_layers

budget = dim * 500
if obj_name == "ellipsometry":
  budget = 1000

# if dim < 15 and budget < 1200:  # We remove Bayesian optimization from high-dimensional contexts.
#   list_optims += ["BO"]  #, "PCABO"]

# All problems.
min_ind = 1.1
max_ind = 3

context_string = f"We work on {obj_name}, dim={dim}, budget={budget}, bounds=[{min_th},{max_th}]"
print(context_string)

import nevergrad as ng
import matplotlib.pyplot as plt
import PyMoosh as pm
import numpy as np

def bragg(x):
  # This cost function corresponds to the problem
  # of maximizing the reflectance, at a given wavelength,
  # of a multilayered structure with alternating refractive
  # indexes. This problem is inspired by the first cases studied in
  # https://www.nature.com/articles/s41598-020-68719-3
  # :-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
  # The input, x, corresponds to the thicknesses of all the layers, :
  # starting with the upper one.                                    :
  # :-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
  x = list(x)
  n = len(x)
  # Working wavelength
  wl = 600.
  materials = [1,1.4**2,1.8**2]
  stack = [0] + [2,1] * (n//2) + [2]
  thicknesses = [0.] + list(x) + [0.]
  structure = pm.Structure(materials,stack,np.array(thicknesses),verbose = False)
  _, R = pm.coefficient_I(structure,wl,0.,0)
  cost = 1-R
  return cost

def photovoltaics(x):
  n = len(x)
  materials = [1., 2., 3., "SiA"]
  stack = [0] + [1,2] * (n//2) + [3]
  thicknesses = [0] + list(x) + [30000]
  structure = pm.Structure(materials, stack, np.array(thicknesses), verbose = False)
  incidence = 0
  pola = 0
  wl_min = 375
  wl_max = 750
  active_lay = len(thicknesses) - 1
  number_pts = 300
  eff, curr, curr_max, wl, spectrum, absorb4 = pm.photo(structure, incidence, pola, wl_min, wl_max, active_lay, number_pts)
  cost = 1 - eff
  return cost

mat = [1.] + [np.random.random()*(max_ind-min_ind) + min_ind for _ in range(nb_layers)] + ["Gold"]
layers = list(range(nb_layers+2))
structure = [0] + [np.random.random()*(max_th-min_th) + min_th for _ in range(nb_layers)] + [0]


angle = 40*np.pi/180
wav_list = np.linspace(400, 800, 100)



def ref_structure(mat, layers, structure, wav_list, angle):
    struct = pm.Structure(mat, layers, structure,verbose = False)
    ellips = np.zeros(len(wav_list), dtype=complex)
    for i, wav in enumerate(wav_list):
        r_s, _, _, _ = pm.coefficient(struct, wav, angle, 0)
        r_p, _, _, _ = pm.coefficient(struct, wav, angle, 1)

        ellips[i] = r_p/r_s
    return ellips, struct

ref_ellips, ellips_structure = ref_structure(mat, layers, structure, wav_list, angle)

def ellipsometry(X, ref_ellips=ref_ellips, wav_list=wav_list, angle=angle, nb_layers=nb_layers):
    mat = [1.] + [x for x in X[:nb_layers]] + ["Gold"]
    layers = [i for i in range(nb_layers+2)]
    structure = np.array([0] + [x for x in X[nb_layers:]] + [0])

    ellips = np.zeros(len(wav_list), dtype=complex)
    interface = pm.Structure(mat, layers, structure, verbose=False)
    for i, wav in enumerate(wav_list):
        r_s, _, _, _ = pm.coefficient_A(interface, wav, angle, 0)
        r_p, _, _, _ = pm.coefficient_A(interface, wav, angle, 1)

        ellips[i] = r_p/r_s

    return np.sum(np.abs(ellips - ref_ellips))


objective_function = {"bragg": bragg, "minibragg": bragg, "bigbragg": bragg, "photovoltaics": photovoltaics, "bigphotovoltaics": photovoltaics, "ellipsometry": ellipsometry}[obj_name]

if run_performance:

    import time
    # Now we run all algorithms and we store results in dictionaries.
    scores = {}
    computational_cost = {}
    yval = {}
    for optim_name in list_optims:
        print("Running ", optim_name)
        scores[optim_name] = []
        computational_cost[optim_name] = []
        yval[optim_name] = []
        for xp in range(num_experiments):
            print(f"Experiment {xp+1} out of {num_experiments}.")
            start_time = time.time()
            # We slightly randomize the upper bound, for checking the robustness.
            r = 0. if xp == 0 else (
                np.random.rand() - 0.5) * (max_th - min_th) * .1
            if obj_name == "ellipsometry":
                array1 = ng.p.Array(shape=(nb_layers,),
                                    lower=min_ind, upper=max_ind)
                array2 = ng.p.Array(shape=(nb_layers,),
                                    lower=min_th, upper=max_th - r)
                instrumentation = ng.p.Instrumentation(
                    array1,
                    array2,
                )
            else:
                v = min_th + (max_th - min_th - r) * Init / 6.
                if Init < 0:
                    init_array = [min_th + (max_th - min_th - r) *
                            np.random.rand() for _ in range(dim)]
                else:
                    init_array = [v] * dim
                instrumentation = ng.p.Array(init=np.array(
                    init_array), lower=min_th, upper=max_th - r)
            optim = ng.optimizers.registry[optim_name](instrumentation, budget)
            if Init < -7 and not "BFGS" in optim_name:
                for _ in range(30):
                    optim.suggest([min_th + (-min_th + max_th - r)
                                  * np.random.rand() for _ in range(dim)])
            best_y = float("inf")
            t0 = time.time()
            xval = []
            obj_time = float("inf")
            for k in range(budget):
                if k % 200 == 0:
                    print(f"Step {k+1} out of {budget}.")
                if time.time() - t0 < min(maximum_time_per_run_in_seconds, maximum_optimization_overhead * (k+1) * obj_time):
                    x = optim.ask()
                    t1 = time.time()
                    val = x.value
                    if obj_name == "ellipsometry":
                        val = list(val[0][0]) + list(val[0][1])
                    elif obj_name == "ellipsometry2":
                        val = [val[0][i][0] for i in range(len(val[0]))]
                    y = objective_function(val)
                    obj_time = float(time.time() - t1)
                    optim.tell(x, y)
                    if y < best_y:
                        best_y = y
                if int(np.log2(k + 1) + .999999) == int(np.log2((k + 1))) or k == budget - 1:
                    xval += [k+1]
                    if len(yval[optim_name]) < len(xval):
                        yval[optim_name] += [[]]
                    yval[optim_name][len(xval)-1] += [best_y]
            computational_cost[optim_name] += [time.time() - t0]
            scores[optim_name] += [best_y]
            print(f"Run {xp+1} done in {time.time() - start_time} seconds.")
xval = []
for k in range(budget):
    if int(np.log2(k + 1) + .999999) == int(np.log2((k + 1))) or k == budget - 1:
        xval += [k+1]
print(len(xval))
print(xval)
keys = ["alg", "xval", "yval", "run"]
values = []
for run in range(num_experiments):
    for k in yval.keys():
        for i in range(len(yval[k])):
            values += [[k, xval[i], yval[k][i][run], run]]
import pandas as pd
df = pd.DataFrame(values, columns=keys)
df.to_csv(f"exp_data/CAI/benchmark/{obj_name}_{list_optims[0]}_{run_id}.csv", index=False)
print(df)
# print(np.array(values))