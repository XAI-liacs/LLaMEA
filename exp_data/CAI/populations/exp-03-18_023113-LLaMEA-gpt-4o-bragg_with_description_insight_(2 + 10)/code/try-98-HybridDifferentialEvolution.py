import numpy as np
from scipy.optimize import minimize

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 10 * dim
        self.F = 0.8
        self.CR = 0.92
        self.num_populations = 5
        self.local_search_method = 'L-BFGS-B'
        self.adaptive_factor = 0.9  # Added adaptive factor

    def quasi_oppositional_init(self, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.base_population_size, self.dim))
        opposite_population = bounds.lb + bounds.ub - population
        return np.vstack((population, opposite_population))

    def differential_evolution_step(self, population, bounds):
        new_population = np.empty_like(population)
        for i in range(population.shape[0]):
            adaptive_F = self.F * self.adaptive_factor  # Adaptive mutation factor
            adaptive_CR = self.CR * self.adaptive_factor  # Adaptive crossover probability
            idxs = [idx for idx in range(population.shape[0]) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + adaptive_F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < adaptive_CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_vector = np.where(cross_points, mutant_vector, population[i])
            new_population[i] = trial_vector
        return new_population

    def local_search(self, candidate, func, bounds):
        bounds_list = [(low, high) for low, high in zip(bounds.lb, bounds.ub)]
        result = minimize(func, candidate, method=self.local_search_method, bounds=bounds_list)
        return result.x if result.success else candidate

    def enforce_periodicity(self, candidate, periodicity=4):  # Adjusted periodicity
        partition_size = self.dim // periodicity
        base_pattern = candidate[:partition_size]
        for i in range(periodicity):
            candidate[i * partition_size:(i + 1) * partition_size] = base_pattern
        return candidate

    def __call__(self, func):
        bounds = func.bounds
        populations = [self.quasi_oppositional_init(bounds) for _ in range(self.num_populations)]
        evaluations = 0
        
        best_candidate = None
        best_score = float('inf')

        while evaluations < self.budget:
            for pop_idx, population in enumerate(populations):
                # Dynamic population size adjustment
                if evaluations > self.budget / 2:
                    population = population[:self.base_population_size // 2]

                new_population = self.differential_evolution_step(population, bounds)
                
                for candidate in new_population:
                    candidate = self.enforce_periodicity(candidate)
                    score = func(candidate)
                    evaluations += 1
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate
                    
                    if evaluations >= self.budget:
                        break

                populations[pop_idx] = new_population

                local_best = self.local_search(best_candidate, func, bounds)
                if func(local_best) < best_score:
                    best_score = func(local_best)
                    best_candidate = local_best

                if evaluations >= self.budget:
                    break

        return best_candidate