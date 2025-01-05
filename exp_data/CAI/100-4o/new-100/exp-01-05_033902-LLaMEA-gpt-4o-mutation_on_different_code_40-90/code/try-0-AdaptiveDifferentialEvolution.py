import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Population size is a common choice of 10 times the dimension
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.bounds = (-5.0, 5.0)
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, population, best_idx):
        indices = list(range(len(population)))
        indices.remove(best_idx)
        np.random.shuffle(indices)
        r1, r2, r3 = indices[:3]
        mutant_vector = (
            population[r1] 
            + self.f * (population[r2] - population[r3])
        )
        mutant_vector = np.clip(mutant_vector, *self.bounds)
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_vector = np.array([
            mutant_vector[i] if np.random.rand() <= self.cr else target_vector[i]
            for i in range(self.dim)
        ])
        return crossover_vector

    def select(self, target_vector, trial_vector, func):
        target_fit = func(target_vector)
        trial_fit = func(trial_vector)
        return (trial_vector, trial_fit) if trial_fit < target_fit else (target_vector, target_fit)

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.bounds[0], self.bounds[1], size=(self.population_size, self.dim)
        )
        fitness = np.array([func(indiv) for indiv in population])
        best_idx = np.argmin(fitness)
        
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Mutation
                mutant_vector = self.mutate(population, best_idx)

                # Crossover
                trial_vector = self.crossover(population[i], mutant_vector)

                # Selection
                population[i], fitness[i] = self.select(population[i], trial_vector, func)
                eval_count += 1

                # Update best solution
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]
                    best_idx = i

        return self.f_opt, self.x_opt