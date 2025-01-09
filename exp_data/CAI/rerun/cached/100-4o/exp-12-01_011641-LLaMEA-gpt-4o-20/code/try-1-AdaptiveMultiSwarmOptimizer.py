import numpy as np

class AdaptiveMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.num_swarms = 5
        self.max_iter = self.budget // self.num_particles
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5

    def __call__(self, func):
        particles = [np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.num_particles, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [p.copy() for p in particles]
        personal_best_scores = [np.inf * np.ones(self.num_particles) for _ in range(self.num_swarms)]
        global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        global_best_score = np.inf

        eval_count = 0

        for iteration in range(self.max_iter):
            for swarm_index in range(self.num_swarms):
                for i in range(self.num_particles):
                    if eval_count >= self.budget:
                        break

                    fitness = func(particles[swarm_index][i])
                    eval_count += 1

                    if fitness < personal_best_scores[swarm_index][i]:
                        personal_best_scores[swarm_index][i] = fitness
                        personal_best_positions[swarm_index][i] = particles[swarm_index][i].copy()

                    if fitness < global_best_score:
                        global_best_score = fitness
                        global_best_position = particles[swarm_index][i].copy()

                r1, r2 = np.random.rand(2)
                velocities[swarm_index] = (self.inertia_weight * velocities[swarm_index] +
                                           self.cognitive_param * r1 * (personal_best_positions[swarm_index] - particles[swarm_index]) +
                                           self.social_param * r2 * (global_best_position - particles[swarm_index]))
                
                particles[swarm_index] += velocities[swarm_index]
                particles[swarm_index] = np.clip(particles[swarm_index], self.lower_bound, self.upper_bound)

                if eval_count >= self.budget:
                    break

            # Adaptive behavior: reinitialize swarms periodically
            if iteration % (self.max_iter // 10) == 0:
                for swarm_index in range(self.num_swarms):
                    if np.random.rand() < 0.3:  # Reinitialize with 30% probability
                        particles[swarm_index] = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))

        return global_best_position, global_best_score