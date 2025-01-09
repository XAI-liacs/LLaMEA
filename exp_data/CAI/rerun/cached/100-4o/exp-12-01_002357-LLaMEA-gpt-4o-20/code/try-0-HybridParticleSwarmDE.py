import numpy as np

class HybridParticleSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            # Evaluate the fitness of each particle
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.population_size

            # Update personal bests and global best
            for i in range(self.population_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.personal_best_positions - self.positions) +
                               self.c2 * r2 * (self.global_best_position - self.positions))
            self.positions = self.positions + self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Apply differential evolution strategy
            if evals < self.budget:
                for i in range(self.population_size):
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = self.positions[np.random.choice(idxs, 3, replace=False)]
                    mutant_vector = a + self.F * (b - c)
                    trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[i])
                    trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                    if func(trial_vector) < scores[i]:
                        self.positions[i] = trial_vector
                        scores[i] = func(trial_vector)
                    evals += 1

        return self.global_best_position, self.global_best_score