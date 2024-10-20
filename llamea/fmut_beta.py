################################################################################
# Benjamin Doerr, Huu Phuoc Le, Régis Makhmara, and Ta Duy Nguyen. 2017.
# Fast genetic algorithms.
# In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17).
# Association for Computing Machinery, New York, NY, USA, 777–784.
# https://doi.org/10.1145/3071178.3071301
################################################################################

import numpy as np


def discrete_power_law_distribution(n, beta):
    def discrete_power_law(n, alpha, beta):
        half_n = int(n/2)
        C_beta_half_n = 0
        for i in range(1, half_n+1):
            C_beta_half_n += i**(-beta)
        probability_alpha = C_beta_half_n**(-1) * alpha**(-beta)
        return probability_alpha
    half_n = int(n/2)
    elements = [alpha for alpha in range(1, half_n+1)]
    probabilities = [discrete_power_law(n, alpha, beta) for alpha in elements]
    sample = np.random.choice(elements, p=probabilities)
    return sample
