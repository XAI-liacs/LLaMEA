import numpy as np

class DynamicMultiSwarmPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 40
        self.num_swarms = 4
        self.swarm_size = self.population_size // self.num_swarms
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        
        self.gbest_positions = [None] * self.num_swarms
        self.gbest_scores = [np.inf] * self.num_swarms
        
        self.c1_initial, self.c1_final = 2.5, 0.5
        self.c2_initial, self.c2_final = 0.5, 2.5
        self.w_initial, self.w_final = 0.9, 0.4
        self.scale_factor = 0.7

    def chaotic_mapping(self, x):
        return 4 * x * (1 - x)

    def __call__(self, func):
        evaluations = 0
        chaotic_factor = np.random.rand()

        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.c1 = self.c1_initial - progress * (self.c1_initial - self.c1_final)
            self.c2 = self.c2_initial + progress * (self.c2_final - self.c2_initial)
            self.w = self.w_initial - progress * (self.w_initial - self.w_final)
            chaotic_factor = self.chaotic_mapping(chaotic_factor)

            for swarm_id in range(self.num_swarms):
                swarm_start = swarm_id * self.swarm_size
                swarm_end = swarm_start + self.swarm_size

                for i in range(swarm_start, swarm_end):
                    score = func(self.particles[i])
                    evaluations += 1

                    if score < self.pbest_scores[i]:
                        self.pbest_scores[i] = score
                        self.pbest_positions[i] = self.particles[i]

                    if score < self.gbest_scores[swarm_id]:
                        self.gbest_scores[swarm_id] = score
                        self.gbest_positions[swarm_id] = self.particles[i]

                    if evaluations >= self.budget:
                        break

                for i in range(swarm_start, swarm_end):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    inertia = chaotic_factor
                    cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
                    social = self.c2 * r2 * (self.gbest_positions[swarm_id] - self.particles[i])
                    self.velocities[i] = inertia * self.w * self.velocities[i] + cognitive + social
                    self.particles[i] += self.velocities[i]
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

                if evaluations < self.budget:
                    for i in range(swarm_start, swarm_end):
                        indices = [idx for idx in range(swarm_start, swarm_end) if idx != i]
                        a, b, c = np.random.choice(indices, 3, replace=False)
                        mutant = self.particles[a] + self.scale_factor * (self.particles[b] - self.particles[c])
                        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                        
                        crossover_rate = 0.8 + 0.1 * chaotic_factor
                        crossover_mask = np.random.rand(self.dim) < crossover_rate
                        trial = np.where(crossover_mask, mutant, self.particles[i])
                        
                        trial_score = func(trial)
                        evaluations += 1

                        if trial_score < self.pbest_scores[i]:
                            self.pbest_scores[i] = trial_score
                            self.pbest_positions[i] = trial

                        if trial_score < self.gbest_scores[swarm_id]:
                            self.gbest_scores[swarm_id] = trial_score
                            self.gbest_positions[swarm_id] = trial

                        if evaluations >= self.budget:
                            break

        overall_best_score = min(self.gbest_scores)
        overall_best_position = self.gbest_positions[np.argmin(self.gbest_scores)]
        return overall_best_score, overall_best_position