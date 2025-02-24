import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, pop_size=20, cross_prob=0.75, diff_weight=0.8):
        population = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += pop_size

        while self.evaluations < self.budget:
            for i in range(pop_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation
                indices = [index for index in range(pop_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + diff_weight * (b - c), bounds.lb, bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < cross_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

            # Enhanced periodicity enforcement using Fourier analysis
            periodic_population = self.enforce_periodicity(population)
            for i in range(pop_size):
                if self.evaluations >= self.budget:
                    break
                periodic_fitness = func(periodic_population[i])
                self.evaluations += 1
                if periodic_fitness < fitness[i]:
                    fitness[i] = periodic_fitness
                    population[i] = periodic_population[i]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def enforce_periodicity(self, population):
        # Use Fourier transform to identify dominant frequencies
        periodic_population = np.copy(population)
        for ind in periodic_population:
            freq_domain = np.fft.fft(ind)
            freq_domain[np.abs(freq_domain) < np.max(np.abs(freq_domain)) * 0.5] = 0
            ind[:] = np.real(np.fft.ifft(freq_domain))
        return periodic_population

    def local_search(self, func, x0, bounds):
        std_bounds = [(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)]
        result = minimize(func, x0, bounds=std_bounds, method='L-BFGS-B')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        # Initial global optimization using Differential Evolution
        best_solution, best_fitness = self.differential_evolution(func, bounds)

        # Refine the best solution using local search
        best_solution, best_fitness = self.local_search(func, best_solution, bounds)

        return best_solution