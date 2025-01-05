import numpy as np

class QuantumInspiredEAWithEntanglement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_qubits = 50
        self.pop_size = 20
        self.population = None
        self.qbits = None
        self.best_solution = None
        self.best_score = float('inf')
        self.rotation_angle = np.pi / 4
        self.entanglement_prob = 0.2

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.qbits = np.random.uniform(0, np.pi, (self.num_qubits, self.dim))

    def measure_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.pop_size):
            for j in range(self.dim):
                prob = np.sin(self.qbits[i % self.num_qubits, j])**2
                self.population[i, j] = lb[j] + (ub[j] - lb[j]) * (prob if np.random.rand() < prob else 1 - prob)

    def update_qbits(self, candidate, best_candidate, bounds):
        for j in range(self.dim):
            if candidate[j] < best_candidate[j]:
                self.qbits[:, j] += self.rotation_angle
            else:
                self.qbits[:, j] -= self.rotation_angle
            self.qbits[:, j] = np.clip(self.qbits[:, j], 0, np.pi)

    def entanglement(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        entangled_pairs = np.random.choice(self.pop_size, (self.pop_size // 2, 2), replace=False)
        for (i, j) in entangled_pairs:
            if np.random.rand() < self.entanglement_prob:
                avg_position = (self.population[i] + self.population[j]) / 2
                self.population[i] = np.clip(avg_position + np.random.uniform(-0.1, 0.1, self.dim), lb, ub)
                self.population[j] = np.clip(avg_position - np.random.uniform(-0.1, 0.1, self.dim), lb, ub)

    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            self.measure_population(func.bounds)
            scores = np.array([func(individual) for individual in self.population])
            evaluations += self.pop_size

            best_index = np.argmin(scores)
            if scores[best_index] < self.best_score:
                self.best_score = scores[best_index]
                self.best_solution = self.population[best_index].copy()

            for i in range(self.pop_size):
                self.update_qbits(self.population[i], self.best_solution, func.bounds)

            self.entanglement(func.bounds)

            if evaluations >= self.budget:
                break