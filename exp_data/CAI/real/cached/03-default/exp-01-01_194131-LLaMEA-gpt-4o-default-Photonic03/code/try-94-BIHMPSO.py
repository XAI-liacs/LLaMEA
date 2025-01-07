import numpy as np

class BIHMPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.random.randn(self.population_size, dim)
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.alpha = 0.1
        self.beta = 0.7
        self.gamma = 1.5
        self.evaluations = 0

    def biologically_mimicked_mutation(self, position, func):
        mutation_strength = np.random.uniform(0.8, 1.2)
        mutated_position = position + mutation_strength * np.random.randn(self.dim)
        mutated_position = np.clip(mutated_position, func.bounds.lb, func.bounds.ub)
        return mutated_position

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        inertia_component = self.alpha * self.velocities[idx]
        cognitive_component = self.beta * r1 * (self.pbest[idx] - self.positions[idx])
        social_component = self.gamma * r2 * (self.gbest - self.positions[idx])

        self.velocities[idx] = inertia_component + cognitive_component + social_component
        new_pos = self.positions[idx] + self.velocities[idx]

        if np.random.rand() < 0.2:
            new_pos = self.biologically_mimicked_mutation(new_pos, func)

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

    def _adapt_parameters(self):
        self.alpha = 0.9 - 0.8 * (self.evaluations / self.budget)
        self.beta = 0.5 + 0.4 * (1 - self.evaluations / self.budget)
        self.gamma = 0.5 + 0.5 * (self.evaluations / self.budget)

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
            self._adapt_parameters()

        return self.gbest