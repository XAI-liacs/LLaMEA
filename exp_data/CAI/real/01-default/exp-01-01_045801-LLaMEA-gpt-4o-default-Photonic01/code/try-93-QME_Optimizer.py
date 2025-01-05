import numpy as np

class QME_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.universes = 5
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.universes, self.population_size, self.dim))
        self.scores = np.array([[self.evaluate(ind) for ind in universe] for universe in self.population])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def quantum_superposition(self, lb, ub):
        for u_idx in range(self.universes):
            quantum_universe = np.random.uniform(-1, 1, (self.population_size, self.dim))
            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    self.population[u_idx][i] = self.best_solution + quantum_universe[i] * np.abs(self.population[u_idx][i] - self.best_solution)
                else:
                    random_idx = np.random.randint(self.universes)
                    if random_idx != u_idx:
                        self.population[u_idx][i] = self.population[random_idx][np.random.randint(self.population_size)]
                self.population[u_idx][i] = np.clip(self.population[u_idx][i], lb, ub)
                self.scores[u_idx][i] = self.evaluate(self.population[u_idx][i])
            self.evaluations += self.population_size
            if self.evaluations >= self.budget:
                return

    def update_best(self):
        for u_idx in range(self.universes):
            best_idx = np.argmin(self.scores[u_idx])
            if self.scores[u_idx][best_idx] < self.best_score:
                self.best_solution = self.population[u_idx][best_idx].copy()
                self.best_score = self.scores[u_idx][best_idx]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.quantum_superposition(lb, ub)
            self.update_best()

        return {'solution': self.best_solution, 'fitness': self.best_score}