import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * dim  # Population size, can be tuned
        self.mutation_factor = 0.5  # Initial mutation factor
        self.crossover_rate = 0.7  # Initial crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Mutation with enhanced strategy
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.mutation_factor * (population[b] - population[c]) + 0.1 * (self.x_opt - population[a])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Crossover
                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate or j == np.random.randint(self.dim):
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

                # Adaptive parameters update with new condition
                if eval_count % (self.pop_size * 5) == 0:
                    self.mutation_factor = 0.5 + 0.3 * np.random.rand()
                    self.crossover_rate = 0.5 + 0.5 * np.random.rand()

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt