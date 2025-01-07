import numpy as np

class QIEA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.qubit_population = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def measure(self):
        return np.sign(self.qubit_population) * np.sqrt(np.abs(self.qubit_population))

    def evaluate(self, solution):
        return self.func(solution)

    def quantum_rotation(self, target_idx, best_qubit):
        theta = np.random.uniform(-np.pi/4, np.pi/4, self.dim)
        self.qubit_population[target_idx] += theta * (best_qubit - self.qubit_population[target_idx])

    def update_best(self, measured_population, scores):
        best_idx = np.argmin(scores)
        if scores[best_idx] < self.best_score:
            self.best_score = scores[best_idx]
            self.best_solution = measured_population[best_idx].copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        while self.evaluations < self.budget:
            measured_population = self.measure()
            measured_population = np.clip(measured_population, lb, ub)
            scores = np.array([self.evaluate(ind) for ind in measured_population])

            self.update_best(measured_population, scores)

            for i in range(self.population_size):
                self.quantum_rotation(i, self.qubit_population[np.argmin(scores)])
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return {'solution': self.best_solution, 'fitness': self.best_score}