import numpy as np
from scipy.optimize import minimize

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.7
        self.CR = 0.9
        self.num_populations = 4
        self.local_search_method = 'L-BFGS-B'
        self.mutation_adapt_rate = 0.1

    def quasi_oppositional_init(self, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        opposite_population = bounds.lb + bounds.ub - population
        return np.vstack((population, opposite_population))

    def differential_evolution_step(self, population, bounds):
        new_population = np.empty_like(population)
        for i in range(population.shape[0]):
            idxs = [idx for idx in range(population.shape[0]) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_vector = np.where(cross_points, mutant_vector, population[i])
            new_population[i] = trial_vector
            new_population[i] += np.random.normal(0, self.mutation_adapt_rate, self.dim)
        return new_population

    def local_search(self, candidate, func, bounds):
        bounds_list = [(low, high) for low, high in zip(bounds.lb, bounds.ub)]
        result = minimize(func, candidate, method=self.local_search_method, bounds=bounds_list)
        return result.x if result.success else candidate

    def enforce_periodicity(self, candidate, periodicity=5):
        partition_size = self.dim // periodicity
        for i in range(periodicity):
            candidate[i * partition_size:(i + 1) * partition_size] = candidate[:partition_size]
        return candidate

    def __call__(self, func):
        bounds = func.bounds
        populations = [self.quasi_oppositional_init(bounds) for _ in range(self.num_populations)]
        evaluations = 0
        
        best_candidate = None
        best_score = float('inf')

        while evaluations < self.budget:
            for pop_idx, population in enumerate(populations):
                new_population = self.differential_evolution_step(population, bounds)
                
                for candidate in new_population:
                    candidate = self.enforce_periodicity(candidate, periodicity=np.random.choice([5, 10]))  # Changed
                    score = func(candidate)
                    evaluations += 1
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate
                    
                    if evaluations >= self.budget:
                        break

                populations[pop_idx] = new_population

                local_best = self.local_search(best_candidate, func, bounds)
                if evaluations % 50 == 0:
                    self.F = np.random.uniform(0.5, 1.0)
                if func(local_best) < best_score:
                    best_score = func(local_best)
                    best_candidate = local_best

                if evaluations >= self.budget:
                    break

        elite_preserved = self.local_search(best_candidate, func, bounds)  # Added
        return elite_preserved  # Changed