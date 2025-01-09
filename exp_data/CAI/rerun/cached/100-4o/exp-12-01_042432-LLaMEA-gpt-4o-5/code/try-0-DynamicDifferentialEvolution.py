import numpy as np

class DynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * self.dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.scale_factor = 0.5
        self.crossover_prob = 0.7
    
    def __call__(self, func):
        evaluations = 0
        best_idx = None
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                # Evaluate individual if necessary
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    evaluations += 1
                    if best_idx is None or self.fitness[i] < self.fitness[best_idx]:
                        best_idx = i

                # Mutation and Crossover
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.scale_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, self.population[i])

                # Evaluate trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.fitness[best_idx]:
                        best_idx = i

            # Dynamically update parameters
            if evaluations % (self.budget // 10) == 0:
                self.scale_factor = 0.5 + 0.1 * np.random.rand()
                self.crossover_prob = 0.7 + 0.1 * np.random.rand()

        return self.population[best_idx], self.fitness[best_idx]