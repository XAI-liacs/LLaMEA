import numpy as np

class QuantumEnhancedES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.sigma = 0.1  # Step size for mutation
        self.evaluations = 0
        self.gbest = None
        self.gbest_score = float('inf')
        self.local_best_positions = np.zeros((self.population_size, dim))
        self.local_best_scores = np.full(self.population_size, float('inf'))

    def quantum_tunneling(self):
        q_tunnel_prob = 0.1  # Probability for applying quantum tunneling
        if np.random.rand() < q_tunnel_prob:
            return np.random.normal(0, self.sigma, self.dim)
        return np.zeros(self.dim)

    def _update_individual(self, idx, func):
        noise = np.random.normal(0, self.sigma, self.dim)
        new_pos = self.positions[idx] + noise + self.quantum_tunneling()
        new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
        new_score = func(new_pos)

        if new_score < self.local_best_scores[idx]:
            self.local_best_positions[idx] = new_pos
            self.local_best_scores[idx] = new_score

        if new_score < self.gbest_score:
            self.gbest = new_pos
            self.gbest_score = new_score

        self.positions[idx] = new_pos
        self.evaluations += 1

    def adaptive_niching(self):
        threshold = 0.2 * self.dim
        for i in range(self.population_size):
            if np.linalg.norm(self.positions[i] - self.gbest) < threshold:
                self.positions[i] = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.dim)
                self.local_best_scores[i] = float('inf')

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.local_best_positions[i] = self.positions[i]
            self.local_best_scores[i] = score
            if score < self.gbest_score:
                self.gbest = self.positions[i]
                self.gbest_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.gbest

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._update_individual(i, func)
                if self.evaluations >= self.budget:
                    break
            self.adaptive_niching()

        return self.gbest