import numpy as np

class EnhancedAdaptiveDifferentialSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf
        self.success_rate = 0.5
        self.population_size = 5  # Dynamic population size
        self.local_search_steps = 3  # Local search steps

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        values = np.array([func(ind) for ind in population])

        for _ in range(self.budget - self.population_size):
            # Rank population
            sorted_indices = np.argsort(values)
            population = population[sorted_indices]
            values = values[sorted_indices]

            # Select parents and generate offspring
            parent1, parent2 = population[:2]
            diff_vector = (parent1 - parent2) * np.random.uniform(0.5, 1.0, self.dim)
            rand_solution = np.random.uniform(lb, ub, self.dim)
            candidate_solution = parent1 + diff_vector + self.success_rate * (rand_solution - parent1)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            # Local search refinement
            for _ in range(self.local_search_steps):
                local_step = np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                refined_candidate = np.clip(candidate_solution + local_step, lb, ub)
                refined_value = func(refined_candidate)
                if refined_value < candidate_value:
                    candidate_solution, candidate_value = refined_candidate, refined_value

            # Update population
            if candidate_value < values[-1]:
                population[-1] = candidate_solution
                values[-1] = candidate_value

            # Update best solution
            if candidate_value < self.best_value:
                self.best_solution = candidate_solution
                self.best_value = candidate_value

            # Update strategy parameters
            if candidate_value < values[0]:
                self.success_rate = min(1.0, self.success_rate + 0.05)
            else:
                self.success_rate = max(0.0, self.success_rate - 0.05)

        return self.best_solution, self.best_value