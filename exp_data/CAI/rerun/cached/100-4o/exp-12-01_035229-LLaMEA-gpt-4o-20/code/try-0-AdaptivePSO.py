import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = min(100, budget // 10)
        self.inertia_weight = 0.7
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_values = np.full(self.num_particles, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = float('inf')
        self.eval_count = 0

    def mutation(self, donor, target):
        return np.clip(target + np.random.uniform(0.5, 1.0) * (donor - target), self.lower_bound, self.upper_bound)

    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.num_particles):
                if self.eval_count >= self.budget:
                    break
                
                value = func(self.particles[i])
                self.eval_count += 1

                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.particles[i]

                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.particles[i]

            for i in range(self.num_particles):
                donor_idx = np.random.choice(self.num_particles)
                if donor_idx != i:
                    donor = self.particles[donor_idx]
                    target = self.particles[i]
                    trial = self.mutation(donor, target)

                    trial_value = func(trial)
                    self.eval_count += 1

                    if trial_value < self.personal_best_values[i]:
                        self.particles[i] = trial
                        self.personal_best_values[i] = trial_value
                        self.personal_best_positions[i] = trial

                        if trial_value < self.global_best_value:
                            self.global_best_value = trial_value
                            self.global_best_position = trial

            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_const * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_velocity = self.social_const * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], self.lower_bound, self.upper_bound)

        return self.global_best_position