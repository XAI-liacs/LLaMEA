import numpy as np

class ImprovedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = min(100, self.budget // 10)
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.swarm_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.f = 0.5  # DE mutation factor
        self.cr = 0.9  # DE crossover probability
        self.current_evaluations = 0
    
    def adaptive_mutation(self, score, best_score):
        return 0.5 + 0.5 * (score - best_score) / (np.abs(best_score) + 1e-8)

    def __call__(self, func):
        while self.current_evaluations < self.budget:
            # Evaluate current particles
            for i in range(self.swarm_size):
                if self.current_evaluations < self.budget:
                    score = func(self.particles[i])
                    self.current_evaluations += 1
                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best_positions[i] = np.copy(self.particles[i])
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = np.copy(self.particles[i])
            
            # Update velocities and positions in PSO manner
            for i in range(self.swarm_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_component = r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = 0.5 * self.velocities[i] + 1.5 * cognitive_component + 1.5 * social_component
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Apply DE mutation and crossover
            for i in range(self.swarm_size):
                if self.current_evaluations < self.budget:
                    indices = np.random.choice(self.swarm_size, 3, replace=False)
                    x1, x2, x3 = self.particles[indices]
                    adaptive_f = self.adaptive_mutation(self.personal_best_scores[i], self.global_best_score)
                    mutant_vector = x1 + adaptive_f * (x2 - x3)
                    trial_vector = np.copy(self.particles[i])
                    j_rand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < self.cr or j == j_rand:
                            trial_vector[j] = mutant_vector[j]
                    trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                    trial_score = func(trial_vector)
                    self.current_evaluations += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector
                        if trial_score < self.global_best_score:
                            self.global_best_score = trial_score
                            self.global_best_position = trial_vector
        return self.global_best_position