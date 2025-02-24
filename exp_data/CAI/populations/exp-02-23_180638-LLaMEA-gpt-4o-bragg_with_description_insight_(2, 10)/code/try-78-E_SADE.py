import numpy as np
from scipy.optimize import minimize

class E_SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.scaling_factor = 0.5
        self.crossover_rate = 0.9
        self.spiral_factor = 0.22  # Increased spiral factor slightly for improved exploration
        self.func_evals = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        self.func_evals += self.population_size
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        while self.func_evals < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = self.periodicity_guided_mutation(pop[i], a, b, c, i, fitness)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, pop[i])
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                trial_fitness = func(trial)
                self.func_evals += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial

                if self.func_evals >= self.budget:
                    break

            # Local search enhancement
            if self.func_evals < self.budget:
                res = minimize(func, best, bounds=bounds, method='L-BFGS-B')
                if res.fun < fitness[best_idx]:
                    best = res.x
                    fitness[best_idx] = res.fun
                    self.func_evals += res.nfev

        return best

    def periodicity_guided_mutation(self, target, a, b, c, i, fitness):
        diff = b - c
        # Modify the spiral factor based on standard deviation of fitness for improved adaptability
        spiral_step = self.spiral_factor * (a - target) * (fitness[i] / (np.std(fitness) + 1e-10))  # Changed line
        periodic_factor = np.cos(2 * np.pi * i / self.population_size)  # Changed line
        adaptive_scaling = self.scaling_factor * (0.5 + 0.5 * periodic_factor)
        mutant = target + adaptive_scaling * diff + spiral_step
        return mutant