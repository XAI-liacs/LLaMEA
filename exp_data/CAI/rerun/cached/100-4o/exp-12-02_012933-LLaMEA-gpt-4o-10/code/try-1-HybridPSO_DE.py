import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 40
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.9  # adaptive inertia weight
        self.f = 0.5  # differential weight
        self.cr = 0.9  # crossover probability
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0

    def logistic_map(self, x):
        return 4 * x * (1 - x)

    def chaotic_sequence(self, length, seed):
        x = seed
        sequence = []
        for _ in range(length):
            x = self.logistic_map(x)
            sequence.append(x)
        return np.array(sequence)

    def __call__(self, func):
        chaotic_sequence = self.chaotic_sequence(self.swarm_size * self.dim, seed=0.7).reshape(self.swarm_size, self.dim)
        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                score = func(self.positions[i])
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            for i in range(self.swarm_size):
                r1 = chaotic_sequence[i]
                r2 = chaotic_sequence[(i + 1) % self.swarm_size]
                self.w = 0.9 - (0.5 * (self.evaluations / self.budget))  # Adaptive inertia
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            if self.evaluations < self.budget:
                for i in range(self.swarm_size):
                    if self.evaluations % (self.budget // 4) == 0:  # Chaotic sequence reinitialization
                        chaotic_sequence = self.chaotic_sequence(self.swarm_size * self.dim, seed=np.random.rand()).reshape(self.swarm_size, self.dim)
                    a, b, c = np.random.choice(self.swarm_size, 3, replace=False)
                    mutant = self.positions[a] + self.f * (self.positions[b] - self.positions[c])
                    trial = np.where(np.random.rand(self.dim) < self.cr, mutant, self.positions[i])
                    trial_score = func(trial)
                    self.evaluations += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.positions[i] = trial
                        self.personal_best_scores[i] = trial_score
                        if trial_score < self.global_best_score:
                            self.global_best_score = trial_score
                            self.global_best_position = trial

        return self.global_best_position, self.global_best_score