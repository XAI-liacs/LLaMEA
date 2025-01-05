import numpy as np

class AQIGA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.q_population = None  # Quantum population
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.theta = np.pi / 4  # Rotation angle for quantum bits

    def initialize_population(self):
        self.q_population = np.random.uniform(0, np.pi / 2, (self.population_size, self.dim))
        self.scores = np.full(self.population_size, float('inf'))

    def decode(self, q_individual):
        return np.cos(q_individual) ** 2  # Decoding quantum bits to classical bits

    def evaluate(self, solution):
        return self.func(solution)

    def measure(self, q_individual):
        return np.array([1 if np.random.rand() < np.cos(theta) ** 2 else 0 for theta in q_individual])

    def quantum_rotation(self, q_individual, classical_individual, best_classical):
        return q_individual + self.theta * (2 * best_classical - classical_individual - 1)

    def select(self, classical_population):
        indices = np.argsort(self.scores)
        return classical_population[indices[:self.population_size // 2]]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population()

        classical_population = np.zeros((self.population_size, self.dim))
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                classical_population[i] = self.decode(self.q_population[i]) * (ub - lb) + lb
                self.scores[i] = self.evaluate(classical_population[i])
                if self.scores[i] < self.best_score:
                    self.best_score = self.scores[i]
                    self.best_solution = classical_population[i].copy()

            best_classical = self.decode(self.q_population[np.argmin(self.scores)])
            selected = self.select(classical_population)

            for i in range(self.population_size):
                measured = self.measure(self.q_population[i])
                self.q_population[i] = self.quantum_rotation(self.q_population[i], measured, best_classical)
                self.q_population[i] = np.clip(self.q_population[i], 0, np.pi / 2)
            
            self.evaluations += self.population_size
            if self.evaluations >= self.budget:
                break
        
        return {'solution': self.best_solution, 'fitness': self.best_score}