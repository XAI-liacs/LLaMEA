import numpy as np

class EnhancedAdaptiveDifferentialSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = min(5 * self.dim, self.budget // 2)  # Dynamic population size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.best_solution = population[np.argmin(fitness)]
        self.best_value = np.min(fitness)

        evaluations = population_size
        step_size = (ub - lb) / 10

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Select three random indices different from i
                indices = np.random.choice(population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(population_size, 3, replace=False)

                x0, x1, x2 = population[indices]
                mutation_vector = x0 + np.random.uniform(0.5, 1.0) * (x1 - x2)
                candidate_solution = np.clip(mutation_vector, lb, ub)
                candidate_value = func(candidate_solution)
                evaluations += 1

                if candidate_value < fitness[i]:
                    population[i] = candidate_solution
                    fitness[i] = candidate_value

                if candidate_value < self.best_value:
                    self.best_solution = candidate_solution
                    self.best_value = candidate_value

            # Adaptive mutation scaling and step size adjustment
            step_size *= 0.95 if evaluations < 0.7 * self.budget else 1.05

        return self.best_solution, self.best_value

# Example usage:
# optimizer = EnhancedAdaptiveDifferentialSearch(budget=100, dim=5)
# best_solution, best_value = optimizer(your_func)