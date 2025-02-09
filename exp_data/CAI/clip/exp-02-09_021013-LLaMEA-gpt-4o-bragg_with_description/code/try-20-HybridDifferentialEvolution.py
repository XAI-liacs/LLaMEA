import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        num_individuals = 50
        evaluations = 0
        F = 0.5  # Mutation factor
        CR = 0.9  # Crossover rate

        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_individuals, self.dim))
        fitness = np.full(num_individuals, float('-inf'))

        # Evaluate initial population
        for i in range(num_individuals):
            fitness[i] = func(population[i])
            evaluations += 1

        while evaluations < self.budget:
            for i in range(num_individuals):
                # Select indices for mutation
                indices = np.random.choice(num_individuals, 3, replace=False)
                x1, x2, x3 = population[indices[0]], population[indices[1]], population[indices[2]]

                # Mutate and Crossover
                mutant = x1 + F * (x2 - x3)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1

                # Select between trial and current individual
                if trial_fitness > fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Dynamic mutation factor adjustment based on convergence
            if evaluations % (self.budget // 10) == 0:
                F = 0.5 + 0.3 * (1 - evaluations / self.budget)

            # Ensure not exceeding budget
            if evaluations >= self.budget:
                break

        best_idx = np.argmax(fitness)
        return population[best_idx], fitness[best_idx]