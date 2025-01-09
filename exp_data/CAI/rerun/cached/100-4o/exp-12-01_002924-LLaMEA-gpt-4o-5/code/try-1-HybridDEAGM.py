import numpy as np

class HybridDEAGM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.population = np.random.uniform(-5, 5, (self.pop_size, dim))
        self.best_solution = None
        self.best_value = float('inf')

    def __call__(self, func):
        evaluations = 0
        fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[idxs]
                mutant = np.clip(a + self.mutation_factor * (b - c), -5, 5)

                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, self.population[i])

                # Fitness Evaluation
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness

                # Adaptive Gaussian Mutation
                if np.random.rand() < 0.1:
                    std_dev = 0.1 * (evaluations / self.budget)
                    gaussian_mutant = np.clip(trial + np.random.normal(0, std_dev, self.dim), -5, 5)
                    gm_fitness = func(gaussian_mutant)
                    evaluations += 1
                    if gm_fitness < fitness[i]:
                        self.population[i] = gaussian_mutant
                        fitness[i] = gm_fitness

            # Update best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_value:
                self.best_value = fitness[best_idx]
                self.best_solution = self.population[best_idx]

        return self.best_solution