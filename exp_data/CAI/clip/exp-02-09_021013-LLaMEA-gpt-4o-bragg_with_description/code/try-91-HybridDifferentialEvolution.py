import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 50
        evaluations = 0

        # DE Hyperparameters
        F = 0.8  # Differential weight
        CR_init = 0.9  # Initial crossover probability
        CR_final = 0.5  # Final crossover probability

        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        # Initialize elite memory
        elite_memory_size = 10
        elite_memory = population[np.argsort(fitness)[:elite_memory_size]]

        while evaluations < self.budget:
            for i in range(population_size):
                # Select three random distinct indices for mutation
                indices = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                
                # Mutation
                mutant_vector = x1 + F * (x2 - x3)

                # Ensure bounds
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                # Adapt crossover probability
                CR = CR_init + (CR_final - CR_init) * (evaluations / self.budget)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                # Ensure bounds for trial vector
                trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)

                # Selection
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Update elite memory
                    if trial_fitness < np.max([func(elite) for elite in elite_memory]):
                        worst_index = np.argmax([func(elite) for elite in elite_memory])
                        elite_memory[worst_index] = trial_vector

            # Ensure not exceeding budget
            if evaluations >= self.budget:
                break

        # Return the best individual and its fitness
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]