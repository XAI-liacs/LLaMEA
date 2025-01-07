import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 10
        self.quantum_population = np.pi * np.random.rand(self.population_size, self.dim)

    def update_quantum_population(self, lb, ub):
        angles = np.random.uniform(-np.pi / 4, np.pi / 4, (self.population_size, self.dim))
        self.quantum_population += angles
        self.solutions = lb + (ub - lb) * 0.5 * (1 + np.sin(self.quantum_population))
        
    def quantum_crossover(self):
        new_population = []
        for i in range(self.population_size):
            partner_index = np.random.randint(self.population_size)
            crossover_point = np.random.randint(1, self.dim)
            new_individual = np.concatenate((self.quantum_population[i, :crossover_point], 
                                             self.quantum_population[partner_index, crossover_point:]))
            new_population.append(new_individual)
        self.quantum_population = np.array(new_population)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0

        while evaluations < self.budget:
            self.update_quantum_population(lb, ub)

            for solution in self.solutions:
                value = func(solution)
                evaluations += 1
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = solution

                if evaluations >= self.budget:
                    break

            self.quantum_crossover()

        return self.best_solution, self.best_value