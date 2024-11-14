import numpy as np

class ImprovedDynamicMutationMultiStepMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 1.0
        self.mutation_factors = np.ones(dim)
        self.steps = 5

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            for _ in range(self.steps):
                adaptive_step_sizes = np.random.uniform(0.5, 1.5, self.dim)
                candidate_solution = best_solution + self.mutation_rate * np.random.uniform(-1, 1, self.dim) * self.mutation_factors * adaptive_step_sizes
                candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
                candidate_fitness = func(candidate_solution)

                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness

                # Update mutation factors based on individual performance
                self.mutation_factors *= 0.9 if candidate_fitness < best_fitness else 1.1
                self.mutation_factors = np.clip(self.mutation_factors, 0.1, 2.0)

            # Update mutation rate based on population diversity
            self.mutation_rate *= 0.99

        return best_solution