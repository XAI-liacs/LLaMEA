import numpy as np

class DifferentialEvolutionLocalSearch:
    def __init__(self, budget=10000, dim=10, population_size=20, crossover_prob=0.9, differential_weight=0.8):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.differential_weight = differential_weight

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = self.population_size
        
        # Update best solution found
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                indices = np.arange(self.population_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = population[a] + self.differential_weight * (population[b] - population[c])
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(cross_points, mutant_vector, population[i])
                
                # Selection
                f_trial = func(trial_vector)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = f_trial

                    # Local exploitation
                    if np.random.rand() < 0.1:  # Small chance to perform local search
                        local_vector = trial_vector + np.random.normal(0, 0.1, self.dim)
                        local_vector = np.clip(local_vector, func.bounds.lb, func.bounds.ub)
                        f_local = func(local_vector)
                        evaluations += 1
                        if f_local < f_trial:
                            population[i] = local_vector
                            fitness[i] = f_local

                # Update global best
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]

        return self.f_opt, self.x_opt