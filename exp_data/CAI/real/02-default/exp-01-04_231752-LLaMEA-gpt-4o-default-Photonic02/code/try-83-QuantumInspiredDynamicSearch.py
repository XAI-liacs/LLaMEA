import numpy as np

class QuantumInspiredDynamicSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.qubits = None
        self.population = None
        self.best_position = None
        self.best_score = np.inf
        self.alpha = 0.5  # Quantum exploration weight
        self.beta = 1.0  # Exploitation weight
        self.adaptive_control = 0.1  # Adaptation rate

    def _initialize_qubits(self):
        self.qubits = np.random.rand(self.population_size, self.dim, 2)
        self.qubits /= np.linalg.norm(self.qubits, axis=2, keepdims=True)

    def _quantum_measurement(self, lb, ub):
        self.population = np.argmax(np.random.rand(self.population_size, self.dim, 1) < self.qubits[:, :, :1], axis=2)
        self.population = self.population * (ub - lb) + lb

    def _update_qubits(self, scores):
        for i in range(self.population_size):
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.best_position = self.population[i]

            # Quantum rotation gates
            theta = self.alpha * np.random.rand(self.dim) * (self.population[i] - self.best_position) / (ub - lb)
            self.qubits[i, :, 0] = self.qubits[i, :, 0] * np.cos(theta) - self.qubits[i, :, 1] * np.sin(theta)
            self.qubits[i, :, 1] = self.qubits[i, :, 0] * np.sin(theta) + self.qubits[i, :, 1] * np.cos(theta)
            self.qubits[i] /= np.linalg.norm(self.qubits[i], axis=1, keepdims=True)

    def _adaptive_exploitation(self):
        self.alpha = np.clip(self.alpha + self.adaptive_control * np.random.uniform(-0.2, 0.2), 0.3, 0.7)
        self.beta = np.clip(self.beta + self.adaptive_control * np.random.uniform(-0.2, 0.2), 0.8, 1.2)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_qubits()

        eval_count = 0
        while eval_count < self.budget:
            self._quantum_measurement(self.lb, self.ub)
            scores = np.array([func(self.population[i]) for i in range(self.population_size)])
            eval_count += self.population_size

            self._update_qubits(scores)
            self._adaptive_exploitation()

        return self.best_position, self.best_score