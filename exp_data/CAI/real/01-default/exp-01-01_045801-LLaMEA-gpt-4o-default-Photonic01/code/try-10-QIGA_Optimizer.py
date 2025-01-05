import numpy as np

class QIGA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.05  # Quantum rotation angle step size
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')

    def initialize_population(self, lb, ub):
        # Use quantum superposition to initialize population in [0, 1]
        self.population = np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))
    
    def evaluate(self, solution):
        # Transform [0, 1] to [lb, ub] before evaluation
        scaled_solution = self.lb + (solution * (self.ub - self.lb))
        return self.func(scaled_solution)

    def quantum_rotation(self, individual, best_individual):
        # Rotate each individual's qubits towards the best solution
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if individual[i] < best_individual[i]:
                    individual[i] += self.alpha * (best_individual[i] - individual[i])
                else:
                    individual[i] -= self.alpha * (individual[i] - best_individual[i])
            individual[i] = np.clip(individual[i], 0, 1)

    def __call__(self, func):
        self.func = func
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        evaluations = 0

        self.initialize_population(self.lb, self.ub)

        while evaluations < self.budget:
            for i in range(self.population_size):
                self.scores[i] = self.evaluate(self.population[i])
                if self.scores[i] < self.best_score:
                    self.best_score = self.scores[i]
                    self.best_solution = self.population[i].copy()
            evaluations += self.population_size

            # Apply quantum rotation towards the best individual
            for i in range(self.population_size):
                self.quantum_rotation(self.population[i], self.best_solution)

        # Transform best solution back to original bounds
        final_solution = self.lb + (self.best_solution * (self.ub - self.lb))
        return {'solution': final_solution, 'fitness': self.best_score}