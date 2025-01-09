import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Mutation factor
        self.CR = 0.7  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select three random indices different from i
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[indices]

                # Mutation
                mutant_vector = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                trial_vector = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial_vector[crossover_points] = mutant_vector[crossover_points]

                # Evaluate the trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

            # Adaptive random search phase
            if evaluations < self.budget:
                best_idx = np.argmin(fitness)
                best_individual = population[best_idx]
                for _ in range(self.population_size):
                    perturbation = np.random.randn(self.dim) * (self.upper_bound - self.lower_bound) / np.sqrt(evaluations)
                    candidate = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evaluations += 1

                    if candidate_fitness < fitness[best_idx]:
                        population[best_idx] = candidate
                        fitness[best_idx] = candidate_fitness
                        best_individual = candidate
                        if evaluations >= self.budget:
                            break

        return population[np.argmin(fitness)]