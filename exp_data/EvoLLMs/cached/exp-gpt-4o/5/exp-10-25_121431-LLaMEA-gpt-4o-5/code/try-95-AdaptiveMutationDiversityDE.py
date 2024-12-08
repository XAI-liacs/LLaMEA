import numpy as np

class AdaptiveMutationDiversityDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 12 + int(self.dim * np.log(self.dim)))
        self.mutation_factor = np.random.uniform(0.4, 0.9, self.population_size)
        self.crossover_rate = np.random.uniform(0.7, 0.9, self.population_size)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0
        self.memory = self.population.copy()
        self.memory_fitness = np.full(self.population_size, np.inf)
        self.global_best = self.population[0].copy()
        self.global_best_fitness = np.inf

    def __call__(self, func):
        self.evaluate_population(func)

        while self.eval_count < self.budget:
            current_population_size = min(self.population_size, self.budget - self.eval_count)
            
            for i in range(current_population_size):
                if self.eval_count >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                
                if np.random.rand() < 0.05:
                    diversity_boost = np.random.uniform(-1, 1, self.dim)
                    gradient = 0.3 * (self.memory[np.random.choice(self.population_size)] - x0) + diversity_boost
                else:
                    gradient = 0.02 * (self.global_best - x0)
                
                feedback = 0.15 * (self.global_best - x0) + 0.1 * (x0 - x1)
                mutant_vector = x0 + self.mutation_factor[i] * (x1 - x2) + gradient + feedback
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate[i], mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.global_best_fitness:
                        self.global_best_fitness = trial_fitness
                        self.global_best = trial_vector.copy()

                if trial_fitness < self.memory_fitness[i]:
                    self.memory[i] = trial_vector.copy()
                    self.memory_fitness[i] = trial_fitness

        return self.global_best

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.eval_count += 1
            if self.fitness[i] < self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best = self.population[i].copy()