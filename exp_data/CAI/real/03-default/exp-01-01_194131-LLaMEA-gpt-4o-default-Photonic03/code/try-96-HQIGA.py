import numpy as np

class HQIGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9

    def _quantum_superposition(self, position, func):
        delta = np.random.rand(self.dim)
        q_position = position + delta * (self.gbest - position)
        q_position = np.clip(q_position, func.bounds.lb, func.bounds.ub)
        q_score = func(q_position)
        return q_position, q_score

    def _differential_mutation(self, idx, func):
        indices = np.random.choice(self.population_size, 3, replace=False)
        mutation_vector = self.positions[indices[0]] + self.mutation_factor * (self.positions[indices[1]] - self.positions[indices[2]])
        mutant = np.clip(mutation_vector, func.bounds.lb, func.bounds.ub)

        trial = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant, self.positions[idx])
        trial_score = func(trial)

        if trial_score < self.pbest_scores[idx]:
            self.pbest[idx] = trial
            self.pbest_scores[idx] = trial_score

        if trial_score < self.gbest_score:
            self.gbest = trial
            self.gbest_score = trial_score

        self.positions[idx] = trial
        self.evaluations += 1

    def _update_mutation_factor(self):
        if self.evaluations % (self.budget // 5) == 0:
            self.mutation_factor = 0.5 + 0.5 * np.random.rand()

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
                q_position, q_score = self._quantum_superposition(self.positions[i], func)
                if q_score < self.pbest_scores[i]:
                    self.pbest[i] = q_position
                    self.pbest_scores[i] = q_score
                if q_score < self.gbest_score:
                    self.gbest = q_position
                    self.gbest_score = q_score

                self._differential_mutation(i, func)
                if self.evaluations >= self.budget:
                    break

            self._update_mutation_factor()

        return self.gbest