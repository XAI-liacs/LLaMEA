import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(4 + int(3 * np.log(dim)), budget)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)

    def __call__(self, func):
        evaluations = 0
        
        # Evaluate initial population
        for i in range(self.population_size):
            if evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                evaluations += 1

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                # Selection
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

        # Return the best solution found
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]