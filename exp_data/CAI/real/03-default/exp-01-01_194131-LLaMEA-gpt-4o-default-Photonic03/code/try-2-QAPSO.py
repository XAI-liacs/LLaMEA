import numpy as np

class QAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)  # Start with smaller size, dynamically adapt
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.phi = np.log(2)
        self.beta = 0.3  # Start with smaller beta, adapt based on convergence
        self.evaluations = 0

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[idx] = (self.phi * self.velocities[idx] +
                                r1 * (self.pbest[idx] - self.positions[idx]) +
                                r2 * (self.gbest - self.positions[idx]))
        new_pos = self.positions[idx] + self.velocities[idx]
        
        # Adaptive Quantum-inspired update
        if np.random.rand() < self.beta:
            new_pos = (self.pbest[idx] + self.gbest) / 2 + np.abs(self.gbest - self.positions[idx]) * np.random.normal(size=self.dim)
        
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

    def _adapt_population(self):
        # Increase population size if budget allows and no improvement seen
        if self.evaluations % (self.budget // 4) == 0:
            new_population_size = min(self.population_size + 5, 20 * self.dim)
            if new_population_size > self.population_size:
                additional_positions = np.random.rand(new_population_size - self.population_size, self.dim)
                self.positions = np.vstack((self.positions, additional_positions))
                self.velocities = np.vstack((self.velocities, np.zeros((new_population_size - self.population_size, self.dim))))
                self.pbest = np.vstack((self.pbest, additional_positions))
                self.pbest_scores = np.hstack((self.pbest_scores, np.full(new_population_size - self.population_size, float('inf'))))
                self.population_size = new_population_size

    def _adapt_beta(self):
        # Increase beta in case of stagnation
        if self.evaluations % (self.budget // 10) == 0:
            self.beta = min(1.0, self.beta + 0.05)

    def __call__(self, func):
        # Initialize positions and evaluate initial scores
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

        # Main loop
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._update_particle(i, func)
                if self.evaluations >= self.budget:
                    break
            self._adapt_population()
            self._adapt_beta()

        return self.gbest