import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(15 * dim, budget // 10)
        self.f = 0.5 + np.random.rand() * 0.3
        self.cr = 0.7 + np.random.rand() * 0.2
        self.current_evals = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        center = (lb + ub) / 2
        radius = (ub - lb) / 4
        pop = np.random.uniform(center - radius, center + radius, (self.population_size, self.dim))
        return pop

    def ensure_periodicity(self, solution, period):
        solution = np.roll(solution, np.random.randint(1, period)) 
        return np.tile(solution[:period], self.dim // period + 1)[:self.dim]

    def opposition_based_learning(self, pop, bounds):
        lb, ub = bounds.lb, bounds.ub
        opposite_pop = lb + ub - pop
        return np.clip(opposite_pop, lb, ub)

    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        opposite_population = self.opposition_based_learning(population, bounds)
        population = np.vstack((population, opposite_population))
        best_solution = None
        best_score = float('inf')
        adaptive_period = 2

        while self.current_evals < self.budget:
            self.population_size = max(5, self.population_size - (self.current_evals // (self.budget // 5)))
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                donor = a + self.f * (b - c) + np.random.normal(0, 0.2, self.dim)  # Coordinated Mutation
                donor = np.clip(donor, bounds.lb, bounds.ub)

                trial = np.copy(population[i])
                self.cr = 0.7 + (1 - (self.current_evals / self.budget)) * 0.3  # Adaptive crossover rate
                for j in range(self.dim):
                    if np.random.rand() < self.cr:
                        trial[j] = donor[j]
                
                diversity_measure = np.std(population, axis=0).mean()  # Line modified for diversity-based periodicity
                adaptive_period = 1 if diversity_measure < 0.1 else 2  # Adaptive Periodicity
                trial = self.ensure_periodicity(trial, period=adaptive_period)

                score = func(trial)
                if score < best_score:
                    best_score = score
                    best_solution = trial
                
                if score < func(population[i]):
                    population[i] = trial

                self.current_evals += 1

        return best_solution

    def local_refinement(self, func, best_solution, bounds):
        result = minimize(func, best_solution, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        bounds = func.bounds
        initial_solution = self.differential_evolution(func, bounds)
        refined_solution = self.local_refinement(func, initial_solution, bounds)
        return refined_solution