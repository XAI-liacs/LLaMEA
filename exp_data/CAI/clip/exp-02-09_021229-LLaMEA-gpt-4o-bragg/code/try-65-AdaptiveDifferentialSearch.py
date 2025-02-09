import numpy as np

class AdaptiveDifferentialSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf
        self.success_rate = 0.5
        self.population_size = 5  # New: Introduced population size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = [np.random.uniform(lb, ub, self.dim) for _ in range(self.population_size)]  # New: Use population

        for i in range(self.budget - 1):
            step_size = (ub - lb) / 10
            new_population = []
            for current_solution in population:
                mutation_factor = np.random.uniform(0.4, 0.9)
                diff_vector = np.random.uniform(-step_size, step_size, self.dim) * mutation_factor
                rand_solution = np.random.uniform(lb, ub, self.dim)
                candidate_solution = current_solution + diff_vector + self.success_rate * (rand_solution - current_solution)
                candidate_solution = np.clip(candidate_solution, lb, ub)
                candidate_value = func(candidate_solution)

                if candidate_value < func(current_solution):
                    new_population.append(candidate_solution)
                else:
                    new_population.append(current_solution)

            new_population.sort(key=func)  # New: Sort based on function value
            population = new_population[:self.population_size]  # New: Keep best solutions

            if func(population[0]) < self.best_value:
                self.best_solution = population[0]
                self.best_value = func(population[0])

        return self.best_solution, self.best_value