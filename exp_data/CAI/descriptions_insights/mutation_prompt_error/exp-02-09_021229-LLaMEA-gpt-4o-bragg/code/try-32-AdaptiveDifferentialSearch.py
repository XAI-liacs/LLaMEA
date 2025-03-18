import numpy as np

class AdaptiveDifferentialSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf
        self.success_rate = 0.5
        self.population = 10  # New: Initialize population size
        self.archive = []  # New: Introduce archive to store diverse solutions

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        solutions = np.random.uniform(lb, ub, (self.population, self.dim))  # New: Multiple initial solutions
        values = np.array([func(sol) for sol in solutions])  # Evaluate all initial solutions
        step_size = (ub - lb) / 10

        for _ in range(self.budget - self.population):
            # New: Select best solution from population for differential vector calculation
            best_idx = np.argmin(values)
            best_solution = solutions[best_idx]
            diff_vector = np.random.uniform(-step_size, step_size, self.dim) * (0.5 + 0.5 * np.random.rand())
            rand_idx = np.random.choice(self.population)
            candidate_solution = best_solution + diff_vector + self.success_rate * (solutions[rand_idx] - best_solution)
            candidate_solution = np.clip(candidate_solution, lb, ub)
            candidate_value = func(candidate_solution)

            # New: Dynamic crowding distance to maintain diversity
            if candidate_value < max(values):
                max_idx = np.argmax(values)
                values[max_idx] = candidate_value
                solutions[max_idx] = candidate_solution
            else:
                self.archive.append(candidate_solution)  # Store in archive if not adopted
                
            if candidate_value < self.best_value:
                self.best_solution = candidate_solution
                self.best_value = candidate_value

        return self.best_solution, self.best_value