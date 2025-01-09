import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.9  # Initial inertia weight
        self.f = 0.8
        self.cr = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Initialize particles for PSO
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim)) * (self.upper_bound - self.lower_bound)
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(self.pop_size, float('inf'))
        self.gbest_position = None
        self.gbest_score = float('inf')

    def __call__(self, func):
        eval_count = 0

        while eval_count < self.budget:
            # Evaluate the current population
            scores = np.array([func(p) for p in self.positions])
            eval_count += self.pop_size
            
            # Update personal and global bests
            for i in range(self.pop_size):
                if scores[i] < self.pbest_scores[i]:
                    self.pbest_scores[i] = scores[i]
                    self.pbest_positions[i] = self.positions[i]
                if scores[i] < self.gbest_score:
                    self.gbest_score = scores[i]
                    self.gbest_position = self.positions[i]

            # PSO velocity and position update
            for i in range(self.pop_size):
                rp = np.random.uniform(0, 1, self.dim)
                rg = np.random.uniform(0, 1, self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * rp * (self.pbest_positions[i] - self.positions[i]) +
                                      self.c2 * rg * (self.gbest_position - self.positions[i]))
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            # Adapt inertia weight
            self.w = 0.4 + 0.5 * (1 - eval_count / self.budget)

            # Apply DE operators on part of the swarm
            if eval_count < self.budget:
                for i in range(self.pop_size):
                    indices = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = np.clip(self.positions[a] + self.f * (self.positions[b] - self.positions[c]), 
                                     self.lower_bound, self.upper_bound)
                    trial = np.array([mutant[j] if np.random.rand() < self.cr else self.positions[i][j] for j in range(self.dim)])
                    trial_score = func(trial)
                    eval_count += 1
                    if trial_score < scores[i]:
                        self.positions[i] = trial
                        scores[i] = trial_score
                        if trial_score < self.pbest_scores[i]:
                            self.pbest_scores[i] = trial_score
                            self.pbest_positions[i] = trial
                            if trial_score < self.gbest_score:
                                self.gbest_score = trial_score
                                self.gbest_position = trial
                            if eval_count >= self.budget:
                                break