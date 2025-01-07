import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.alpha = 0.05  # Mutation rate in quantum state
        self.betas = np.random.uniform(0.5, 1.0, (self.population_size, self.dim))  # Probabilistic weights
        self.particles = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.best_positions = np.copy(self.particles)
        self.best_scores = np.full(self.population_size, np.inf)

    def _quantum_update(self, lb, ub):
        for i in range(self.population_size):
            quantum_state = np.random.rand(self.dim) < self.betas[i]
            mutation = np.random.randn(self.dim) * self.alpha
            if np.random.rand() < 0.5:  # Explore
                self.particles[i] = quantum_state * (self.best_positions[i] + mutation) + (1 - quantum_state) * (self.global_best_position + mutation)
            else:  # Exploit
                self.particles[i] = quantum_state * (self.global_best_position + mutation) + (1 - quantum_state) * (self.particles[i] + mutation)
            
            self.particles[i] = np.clip(self.particles[i], lb, ub)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.particles[i])
                eval_count += 1

                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
                    # Adapt probabilistic weights slightly
                    self.betas[i] = np.clip(self.betas[i] + np.random.uniform(-0.05, 0.05, self.dim), 0.5, 1.0)

            self._quantum_update(self.lb, self.ub)

        return self.global_best_position, self.global_best_score