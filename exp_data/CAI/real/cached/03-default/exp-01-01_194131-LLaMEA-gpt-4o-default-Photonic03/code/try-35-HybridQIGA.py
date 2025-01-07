import numpy as np

class HybridQIGA:
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
        self.evaluations = 0
        self.num_swarms = max(2, dim // 5)
        self.local_gbests = [None] * self.num_swarms
        self.local_gbest_scores = [float('inf')] * self.num_swarms
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9

    def differential_mutation(self, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant_vector = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
        mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _update_particle(self, idx, func):
        swarm_idx = idx % self.num_swarms
        local_gbest = self.local_gbests[swarm_idx] if self.local_gbests[swarm_idx] is not None else self.gbest
        
        mutant = self.differential_mutation(idx)
        trial = self.crossover(self.positions[idx], mutant)
        
        new_score = func(trial)
        
        if new_score < self.pbest_scores[idx]:
            self.pbest[idx] = trial
            self.pbest_scores[idx] = new_score
        
        if new_score < self.local_gbest_scores[swarm_idx]:
            self.local_gbests[swarm_idx] = trial
            self.local_gbest_scores[swarm_idx] = new_score
        
        if new_score < self.gbest_score:
            self.gbest = trial
            self.gbest_score = new_score
        
        self.positions[idx] = trial
        self.evaluations += 1

    def _adapt_mutation_factor(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.mutation_factor = min(1.0, self.mutation_factor + 0.05)

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
            self._adapt_mutation_factor()

        return self.gbest