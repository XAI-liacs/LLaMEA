import numpy as np

class HybridDELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.pop_size, self.dim)
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.pop_size

        while evaluations < self.budget:
            # Differential Evolution Mutation and Crossover
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Local Search: Hill Climbing Step
                for _ in range(5):  # Attempt a few local steps
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    candidate = np.clip(trial + perturbation, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < func(trial):
                        trial = candidate
                    if evaluations >= self.budget:
                        break

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        # Return the best found solution
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]