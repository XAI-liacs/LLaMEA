import numpy as np

class ImprovedMultiDynamicDE(MultiDynamicDE):
    def __call__(self, func):
        populations = [np.random.uniform(-5.0, 5.0, (self.budget // self.num_populations, self.dim)) for _ in range(self.num_populations)]
        best_solutions = np.zeros((self.num_populations, self.dim))
        best_fitness = np.ones(self.num_populations) * np.inf
        
        for i in range(self.num_populations):
            fitness = np.array([func(x) for x in populations[i]])
            best_idx = np.argmin(fitness)
            best_solutions[i] = populations[i][best_idx]
            best_fitness[i] = fitness[best_idx]
        
        for _ in range(self.budget - self.num_populations):
            for i in range(self.num_populations):
                best_idx = np.argmin(best_fitness)
                mutant = populations[i][np.random.choice(len(populations[i]), 3, replace=False)]
                mutated_vector = best_solutions[i] + self.mutation_factors[i] * (mutant[0] - mutant[1])
                mutated_vector = np.clip(mutated_vector, -5.0, 5.0)
                trial_vector = np.where(np.random.rand(self.dim) < self.mutation_factors[i], mutated_vector, best_solutions[i])
                trial_fitness = func(trial_vector)
                if trial_fitness < best_fitness[i]:
                    populations[i][best_idx] = trial_vector
                    best_solutions[i] = trial_vector
                    best_fitness[i] = trial_fitness

                if np.random.rand() < 0.1:  # Update mutation factor based on individual performance
                    diff_fitness = best_fitness[i] - trial_fitness
                    if diff_fitness > 0:
                        self.mutation_factors[i] = np.clip(self.mutation_factors[i] * (1 - 0.1 * diff_fitness), self.mutation_factor_range[0], self.mutation_factor_range[1])
                    else:
                        self.mutation_factors[i] = np.clip(self.mutation_factors[i] * (1 + 0.1 * abs(diff_fitness)), self.mutation_factor_range[0], self.mutation_factor_range[1])
        
        overall_best_idx = np.argmin(best_fitness)
        return best_solutions[overall_best_idx]