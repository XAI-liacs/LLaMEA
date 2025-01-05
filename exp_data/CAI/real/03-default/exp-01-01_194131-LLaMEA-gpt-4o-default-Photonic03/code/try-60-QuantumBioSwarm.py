import numpy as np

class QuantumBioSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.phi = np.log(2)
        self.evaluations = 0
        self.alpha = 0.1  # Quantum influence
        self.gamma = 0.5  # Symbiosis factor

    def quantum_move(self):
        q_positions = np.random.uniform(-1, 1, self.positions.shape)
        return self.alpha * q_positions

    def symbiotic_interaction(self):
        pair_indices = np.random.permutation(self.population_size)
        symbiotic_moves = np.zeros_like(self.positions)
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size:
                diff = self.positions[pair_indices[i+1]] - self.positions[pair_indices[i]]
                symbiotic_moves[pair_indices[i]] += self.gamma * diff
                symbiotic_moves[pair_indices[i+1]] -= self.gamma * diff
        return symbiotic_moves

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[idx] = (self.phi * self.velocities[idx] +
                                r1 * (self.pbest[idx] - self.positions[idx]) +
                                r2 * (self.gbest - self.positions[idx]))
        quantum_part = self.quantum_move()
        symbiotic_part = self.symbiotic_interaction()[idx]
        new_pos = self.positions[idx] + self.velocities[idx] + quantum_part + symbiotic_part
        new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
        new_score = func(new_pos)

        if new_score < self.pbest_scores[idx]:
            self.pbest[idx] = new_pos
            self.pbest_scores[idx] = new_score

        if new_score < self.gbest_score:
            self.gbest = new_pos
            self.gbest_score = new_score

        self.positions[idx] = new_pos
        self.evaluations += 1

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.pbest[i] = self.positions[i]
            self.pbest_scores[i] = score
            if score < self.gbest_score:
                self.gbest = self.positions[i]
                self.gbest_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.gbest

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._update_particle(i, func)
                if self.evaluations >= self.budget:
                    break

        return self.gbest