import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim
        self.CR = 0.9
        self.F = 0.8
        self.history_CR = [self.CR]
        self.history_F = [self.F]
        self.memory_size = 5

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        eval_count = self.population_size
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation and Crossover
                idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant_vector = population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]])
                mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial_vector = np.where(cross_points, mutant_vector, population[i])

                # Selection
                trial_fitness = func(trial_vector)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial_vector
                        best_fitness = trial_fitness

                    # Adapt CR and F
                    self.history_CR.append(self.CR)
                    self.history_F.append(self.F)

                    if len(self.history_CR) > self.memory_size:
                        del self.history_CR[0]
                    if len(self.history_F) > self.memory_size:
                        del self.history_F[0]

                    if np.random.rand() < 0.5:
                        self.CR = np.mean(self.history_CR) + np.std(self.history_CR) * np.random.randn()
                        self.CR = np.clip(self.CR, 0, 1)
                    else:
                        self.F = np.mean(self.history_F) + np.std(self.history_F) * np.random.randn()
                        self.F = np.clip(self.F, 0, 2)

                if eval_count >= self.budget:
                    break

        return best_solution, best_fitness