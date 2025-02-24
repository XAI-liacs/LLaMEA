import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def differential_evolution(self, func, bounds, pop_size=15, F=0.8, Cr=0.9):
        population = np.random.rand(pop_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += pop_size
        best_idx = np.argmin(fitness)
        best_ind = population[best_idx]

        while self.eval_count < self.budget:
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Self-adaptive mutation factor
                F_adaptive = F + 0.1 * np.random.rand() - 0.05
                F_adaptive = np.clip(F_adaptive, 0.5, 1.0)
                
                # Self-adaptive crossover rate
                Cr_adaptive = Cr + 0.1 * np.random.rand() - 0.05
                Cr_adaptive = np.clip(Cr_adaptive, 0.5, 1.0)

                mutant = np.clip(a + F_adaptive * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < Cr_adaptive, mutant, population[i])
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_ind = trial

        return best_ind, fitness[best_idx]

    def local_optimization(self, func, x0):
        result = minimize(func, x0, method='BFGS', options={'maxiter': self.budget - self.eval_count})
        self.eval_count += result.nfev
        return result.x, result.fun

    def enforce_periodicity(self, x, period):
        # Constrain periodicity by averaging
        x = np.array(x)
        sections = np.split(x, self.dim // period)
        averaged_sections = np.mean(sections, axis=0)
        return np.tile(averaged_sections, self.dim // period)

    def __call__(self, func):
        bounds = func.bounds
        period = self.dim // 2  # Aim for a periodic solution with half the dimensions
        x_global, f_global = self.differential_evolution(func, bounds)

        # Introduce elitism by using the best solution found globally
        x_periodic = self.enforce_periodicity(x_global, period)
        x_local, f_local = self.local_optimization(func, x_periodic)

        return x_local if f_local < f_global else x_global