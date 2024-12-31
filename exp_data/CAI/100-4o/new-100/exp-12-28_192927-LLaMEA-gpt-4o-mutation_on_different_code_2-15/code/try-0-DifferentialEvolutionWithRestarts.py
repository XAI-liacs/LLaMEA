import numpy as np

class DifferentialEvolutionWithRestarts:
    def __init__(self, budget=10000, dim=10, population_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        
        def initialize_population():
            return np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))

        def mutate(x, a, b, c):
            mutant = x + self.F * (a - b + c - x)
            return np.clip(mutant, bounds[0], bounds[1])

        def crossover(target, mutant):
            cross_points = np.random.rand(self.dim) < self.CR
            trial = np.where(cross_points, mutant, target)
            return trial

        # Initialize main population
        population = initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Choose random indices for mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Mutation and Crossover
                mutant = mutate(population[i], a, b, c)
                trial = crossover(population[i], mutant)
                
                # Evaluate trial individual
                f_trial = func(trial)
                evaluations += 1

                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Check budget
                if evaluations >= self.budget:
                    break

            # Restart Mechanism
            if evaluations < self.budget:
                best_indices = np.argsort(fitness)[:self.population_size // 5]
                new_population = initialize_population()
                for idx in best_indices:
                    local_search_area = np.random.normal(population[idx], 0.1, (3, self.dim))
                    local_fitness = np.array([func(ind) for ind in local_search_area])
                    evaluations += 3
                    local_best_idx = np.argmin(local_fitness)
                    if local_fitness[local_best_idx] < fitness[idx]:
                        population[idx] = local_search_area[local_best_idx]
                        fitness[idx] = local_fitness[local_best_idx]
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = population[idx]

                if evaluations < self.budget:
                    population = np.vstack((population, new_population))
                    population = population[:self.population_size]
                    fitness = np.array([func(ind) for ind in population])
                    evaluations += len(population)

        return self.f_opt, self.x_opt