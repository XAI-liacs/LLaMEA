################################################################################
# Benjamin Doerr, Huu Phuoc Le, Régis Makhmara, and Ta Duy Nguyen. 2017.
# Fast genetic algorithms.
# In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17).
# Association for Computing Machinery, New York, NY, USA, 777–784.
# https://doi.org/10.1145/3071178.3071301
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def discrete_power_law(n, alpha, beta):
    half_n = int(n/2)
    C_beta_half_n = 0
    for i in range(1, half_n+1):
        C_beta_half_n += i**(-beta)
    probability_alpha = C_beta_half_n**(-1) * alpha**(-beta)
    return probability_alpha


cud = ["#e69f00", "#56b4e9", "#009e73", "#f0e442",
       "#0072b2", "#d55e00", "#cc79a7", "#000000"]
plt.figure(figsize=(8, 4))
n = 100
for beta in [1.5, 2.0, 3.0]:
    x = []
    y = []
    X = np.linspace(1, int(n/2)+1, 5000)
    for alpha in X:
        x += [alpha / n]
        y += [discrete_power_law(n, alpha, beta)]
    plt.plot(x, y, label=r"$\beta$="+str(beta), color=cud.pop(0))
plt.xscale("log")
plt.legend()
plt.xlabel("mutation rate")
plt.ylabel("probability")
plt.tight_layout()
# plt.title("Discrete Power Law")
plt.savefig("discrete_power_law.png")
