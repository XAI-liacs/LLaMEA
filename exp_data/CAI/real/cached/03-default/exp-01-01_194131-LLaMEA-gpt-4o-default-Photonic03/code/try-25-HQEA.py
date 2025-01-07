import numpy as np

class HQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.qbit_angles = np.pi * (2 * np.random.rand(self.population_size, dim) - 1)
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.mutation_rate = 0.1

    def _quantum_rotation(self, angles):
        return np.sign(np.sin(angles)) * np.sqrt(np.abs(np.sin(angles)))

    def _update_individual(self, idx, func):
        # Apply quantum rotation to generate new position
        self.qbit_angles[idx] += np.pi * (2 * np.random.rand(self.dim) - 1) * 0.05
        new_pos = self._quantum_rotation(self.qbit_angles[idx])

        # Crossover with personal best
        crossover_mask = np.random.rand(self.dim) < 0.5
        new_pos[crossover_mask] = self.pbest[idx, crossover_mask]

        # Mutation
        mutation_mask = np.random.rand(self.dim) < self.mutation_rate
        new_pos[mutation_mask] = func.bounds.lb[mutation_mask] + \
                                 (func.bounds.ub[mutation_mask] - func.bounds.lb[mutation_mask]) * np.random.rand(np.sum(mutation_mask))

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
                self._update_individual(i, func)
                if self.evaluations >= self.budget:
                    break

        return self.gbest