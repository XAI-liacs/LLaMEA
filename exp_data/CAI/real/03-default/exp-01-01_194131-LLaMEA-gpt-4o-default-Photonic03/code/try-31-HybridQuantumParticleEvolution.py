import numpy as np

class HybridQuantumParticleEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 7 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.q_factor = 0.5
        self.diff_weight = 0.8
        self.chaos_beta = 0.3
        self.evaluations = 0

    def differential_mutation(self, target_idx, func):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.positions[a] + self.diff_weight * (self.positions[b] - self.positions[c])
        return np.clip(mutant, func.bounds.lb, func.bounds.ub)

    def chaos_local_search(self, pos, func):
        chaos_seq = np.random.rand(self.dim)
        chaos_step = self.q_factor * (chaos_seq - 0.5)
        new_pos = pos + chaos_step
        new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
        score = func(new_pos)
        return new_pos, score

    def _update_particle(self, idx, func):
        new_pos = self.differential_mutation(idx, func)
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        velocity = (self.q_factor * self.velocities[idx] 
                    + r1 * (self.pbest[idx] - self.positions[idx]) 
                    + r2 * (self.gbest - self.positions[idx]))
        candidate_pos = new_pos + velocity
        candidate_pos = np.clip(candidate_pos, func.bounds.lb, func.bounds.ub)
        candidate_score = func(candidate_pos)

        if candidate_score < self.pbest_scores[idx]:
            self.pbest[idx] = candidate_pos
            self.pbest_scores[idx] = candidate_score

        if candidate_score < self.gbest_score:
            self.gbest = candidate_pos
            self.gbest_score = candidate_score

        if np.random.rand() < self.chaos_beta:
            chaos_pos, chaos_score = self.chaos_local_search(candidate_pos, func)
            if chaos_score < candidate_score:
                candidate_pos, candidate_score = chaos_pos, chaos_score

        self.positions[idx] = candidate_pos
        self.evaluations += 1

    def _adapt_q_factor(self):
        if self.evaluations % (self.budget // 5) == 0:
            self.q_factor = max(0.3, self.q_factor - 0.05)

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
            self._adapt_q_factor()

        return self.gbest