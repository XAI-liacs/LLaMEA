import numpy as np

class AIHMO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 30
        self.population = []

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            fitness = float('inf')
            population.append({'position': position, 'fitness': fitness})
        return population

    def hypermutation(self, position, lb, ub, evaluation_ratio):
        mutation_rate = np.exp(-evaluation_ratio * 5)  # Adaptive mutation rate
        mutation_vector = (ub - lb) * (np.random.rand(self.dim) - 0.5) * mutation_rate
        new_position = position + mutation_vector
        return np.clip(new_position, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            for individual in self.population:
                individual['fitness'] = func(individual['position'])
                evaluations += 1

                if individual['fitness'] < self.best_value:
                    self.best_value = individual['fitness']
                    self.best_solution = individual['position'].copy()

                if evaluations >= self.budget:
                    break

            # Evaluate hypermutation for new candidate solutions
            for individual in self.population:
                if evaluations >= self.budget:
                    break
                new_position = self.hypermutation(individual['position'], lb, ub, evaluations / self.budget)
                new_fitness = func(new_position)
                evaluations += 1

                if new_fitness < self.best_value:
                    self.best_value = new_fitness
                    self.best_solution = new_position.copy()

        return self.best_solution, self.best_value