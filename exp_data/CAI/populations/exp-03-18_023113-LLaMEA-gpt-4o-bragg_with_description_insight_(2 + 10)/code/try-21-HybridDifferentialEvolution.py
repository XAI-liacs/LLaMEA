import numpy as np
from scipy.optimize import minimize

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.7  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.num_populations = 3
        self.local_search_method = 'L-BFGS-B'

    def quasi_oppositional_init(self, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        opposite_population = bounds.lb + bounds.ub - population
        return np.vstack((population, opposite_population))

    def differential_evolution_step(self, population, bounds):
        new_population = np.empty_like(population)
        for i in range(population.shape[0]):
            idxs = [idx for idx in range(population.shape[0]) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            F_dynamic = np.random.rand() * self.F  # Dynamic adjustment
            mutant_vector = np.clip(a + F_dynamic * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_vector = np.where(cross_points, mutant_vector, population[i])
            new_population[i] = trial_vector
        return new_population

    def local_search(self, candidate, func, bounds):
        result = minimize(func, candidate, method=self.local_search_method, bounds=list(zip(bounds.lb, bounds.ub)))
        return result.x if result.success else candidate

    def enforce_periodicity(self, candidate, periodicity=2):
        partition_size = self.dim // periodicity
        for i in range(periodicity):
            candidate[i * partition_size:(i + 1) * partition_size] = candidate[:partition_size]
        return candidate

    def simulated_annealing(self, candidate, func):
        temperature = 1.0
        cooling_rate = 0.9
        current_score = func(candidate)
        while temperature > 0.1:
            new_candidate = candidate + np.random.normal(0, 0.1, self.dim)
            new_score = func(new_candidate)
            if new_score < current_score or np.exp((current_score - new_score) / temperature) > np.random.rand():
                candidate = new_candidate
                current_score = new_score
            temperature *= cooling_rate
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
                    candidate = self.enforce_periodicity(candidate)
                    score = func(candidate)
                    evaluations += 1
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate
                    
                    if evaluations >= self.budget:
                        break

                population = new_population

                # Apply local search to the best candidate of each population
                local_best = self.simulated_annealing(best_candidate, func)  # Integrated Simulated Annealing
                if func(local_best) < best_score:
                    best_score = func(local_best)
                    best_candidate = local_best

                if evaluations >= self.budget:
                    break

        return best_candidate