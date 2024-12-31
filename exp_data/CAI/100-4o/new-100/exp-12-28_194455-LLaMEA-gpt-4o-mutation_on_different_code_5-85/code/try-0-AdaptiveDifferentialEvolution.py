import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], 
                                       (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.update_optimal(population, fitness)

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = x1 + self.mutation_factor * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
                
                # Crossover
                trial_vector = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]

                # Selection
                trial_fitness = func(trial_vector)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial_vector
                    self.update_optimal([trial_vector], [trial_fitness])

                if eval_count >= self.budget:
                    break

            # Adaptive adjustment based on population diversity
            diversity = np.mean(np.std(population, axis=0))
            self.mutation_factor = 0.5 + 0.3 * (1.0 - diversity / (self.bounds[1] - self.bounds[0]))

        return self.f_opt, self.x_opt

    def update_optimal(self, candidates, fitness):
        idx = np.argmin(fitness)
        if fitness[idx] < self.f_opt:
            self.f_opt = fitness[idx]
            self.x_opt = candidates[idx]