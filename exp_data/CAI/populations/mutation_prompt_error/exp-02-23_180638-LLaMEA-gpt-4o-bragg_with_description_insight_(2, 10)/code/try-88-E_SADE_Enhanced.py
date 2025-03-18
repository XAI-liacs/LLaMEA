import numpy as np
from scipy.optimize import minimize

class E_SADE_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.scaling_factor = 0.6  # Changed line
        self.crossover_rate = 0.9
        self.spiral_factor = 0.3  # Changed line
        self.func_evals = 0
        self.fitness_history = []
        self.chaos_map = self.init_chaos_map()  # Added line

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        self.func_evals += self.population_size
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        self.fitness_history.append(np.min(fitness))
        
        while self.func_evals < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = self.dynamic_spiral_mutation(pop[i], a, b, c, i, fitness)
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
                self.fitness_history.append(np.min(fitness))

                if self.func_evals >= self.budget:
                    break

            if self.func_evals < self.budget:
                res = minimize(func, best, bounds=bounds, method='L-BFGS-B')
                if res.fun < fitness[best_idx]:
                    best = res.x
                    fitness[best_idx] = res.fun
                    self.func_evals += res.nfev

        return best

    def dynamic_spiral_mutation(self, target, a, b, c, i, fitness):
        diff = b - c
        fitness_improvement = max(self.fitness_history[-5:]) - min(self.fitness_history[-5:]) if len(self.fitness_history) > 5 else 0
        spiral_step = self.spiral_factor * (a - target) * fitness_improvement
        periodic_factor = np.sin(2 * np.pi * i / self.population_size)
        adaptive_scaling = self.scaling_factor * (0.5 + 0.5 * periodic_factor)
        chaos_factor = self.chaos_map[i]  # Added line
        mutant = target + adaptive_scaling * diff + spiral_step * chaos_factor  # Modified line
        return mutant

    def init_chaos_map(self):  # Added lines
        chaos_map = np.random.rand(self.population_size)
        for i in range(1, len(chaos_map)):
            chaos_map[i] = 4 * chaos_map[i-1] * (1 - chaos_map[i-1])  # Logistic map
        return chaos_map