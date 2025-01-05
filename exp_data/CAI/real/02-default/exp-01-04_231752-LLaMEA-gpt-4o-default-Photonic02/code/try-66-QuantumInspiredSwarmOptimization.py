import numpy as np

class QuantumInspiredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.alpha = 0.5  # Exploration weight
        self.beta = 0.5   # Exploitation weight
        self.quantum_operator = 0.01  # Quantum operation factor

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _quantum_update(self, particle, lb, ub):
        # Apply quantum-inspired superposition
        quantum_state = particle + self.quantum_operator * (np.random.rand(self.dim) - 0.5)
        return np.clip(quantum_state, lb, ub)

    def _update_particles(self, lb, ub):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = self.alpha * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_component = self.beta * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = cognitive_component + social_component
            quantum_component = self._quantum_update(self.particles[i], lb, ub)

            # Combine classical and quantum updates
            self.particles[i] += self.velocities[i] + quantum_component - self.particles[i]
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

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score