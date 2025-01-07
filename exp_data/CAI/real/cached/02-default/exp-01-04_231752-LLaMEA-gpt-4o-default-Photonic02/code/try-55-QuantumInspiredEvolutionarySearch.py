import numpy as np

class QuantumInspiredEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.particles = None
        self.best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.alpha = 0.99  # Convergence control parameter

    def _initialize_population(self, lb, ub):
        # Initialize particles using quantum-inspired superposition
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.quantum_states = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10

    def _measure_positions(self, lb, ub):
        # Simulate quantum measurement to update positions
        self.particles += self.quantum_states * np.random.normal(0, 1, (self.population_size, self.dim))
        self.particles = np.clip(self.particles, lb, ub)

    def _update_quantum_states(self):
        # Entanglement and interaction to update quantum states
        for i in range(self.population_size):
            if np.random.rand() < 0.5:
                partner = np.random.randint(self.population_size)
                self.quantum_states[i] = (self.quantum_states[i] + self.quantum_states[partner]) / 2
            self.quantum_states[i] *= self.alpha

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            self._measure_positions(self.lb, self.ub)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.particles[i])
                eval_count += 1

                if score < self.best_scores[i]:
                    self.best_scores[i] = score

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            self._update_quantum_states()

        return self.global_best_position, self.global_best_score