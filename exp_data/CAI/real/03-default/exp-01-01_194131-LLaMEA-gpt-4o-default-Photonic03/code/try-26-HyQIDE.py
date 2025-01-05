import numpy as np

class HyQIDE:
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
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.evaluations = 0

    def levy_flight(self, scale=0.01):
        # Generate Levy flight steps
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def differential_mutation(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.positions[a] + self.F * (self.positions[b] - self.positions[c])
        return np.clip(mutant, 0, 1)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _update_particle(self, idx, func):
        mutant = self.differential_mutation(idx)
        trial = self.crossover(self.positions[idx], mutant)
        
        # Quantum-inspired update with Levy flight
        if np.random.rand() < 0.5:
            trial += self.levy_flight()
        
        trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
        trial_score = func(trial)

        if trial_score < self.pbest_scores[idx]:
            self.pbest[idx] = trial
            self.pbest_scores[idx] = trial_score

        if trial_score < self.gbest_score:
            self.gbest = trial
            self.gbest_score = trial_score

        self.positions[idx] = trial
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