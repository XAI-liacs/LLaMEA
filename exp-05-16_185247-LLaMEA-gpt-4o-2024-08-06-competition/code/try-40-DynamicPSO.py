import numpy as np

class DynamicPSO:
    def __init__(self, budget=10000, dim=10, n_particles=30):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        self.swarms = 3  # New line: Introduce multiple swarms
        self.particles_per_swarm = self.n_particles // self.swarms  # New line

    def __call__(self, func):
        # Initialize particle positions and velocities for each swarm
        positions = [np.random.uniform(self.lower_bound, self.upper_bound, (self.particles_per_swarm, self.dim)) for _ in range(self.swarms)]
        velocities = [np.zeros((self.particles_per_swarm, self.dim)) for _ in range(self.swarms)]
        personal_best_positions = [np.copy(positions[i]) for i in range(self.swarms)]
        personal_best_scores = [np.full(self.particles_per_swarm, np.inf) for _ in range(self.swarms)]
        
        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0
        while evaluations < self.budget:
            for swarm_id in range(self.swarms):  # New line: Iterate over swarms
                for i in range(self.particles_per_swarm):
                    if evaluations >= self.budget:
                        break
                    
                    fitness = func(positions[swarm_id][i])
                    evaluations += 1
                    
                    if fitness < personal_best_scores[swarm_id][i]:
                        personal_best_scores[swarm_id][i] = fitness
                        personal_best_positions[swarm_id][i] = positions[swarm_id][i]
                    
                    if np.random.rand() < 0.5:
                        if fitness < global_best_score:
                            global_best_score = fitness
                            global_best_position = positions[swarm_id][i]
                
                inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)
                
                for i in range(self.particles_per_swarm):
                    r1, r2 = np.random.rand(2)
                    learning_factor_cognitive = 2.0 - 1.5 * (evaluations / self.budget)
                    learning_factor_social = 2.0 + 1.5 * (evaluations / self.budget)
                    velocities[swarm_id][i] = (inertia_weight * velocities[swarm_id][i] +
                                               learning_factor_cognitive * r1 * (personal_best_positions[swarm_id][i] - positions[swarm_id][i]) +
                                               learning_factor_social * r2 * (global_best_position - positions[swarm_id][i]))
                    if np.random.rand() < 0.1:
                        velocities[swarm_id][i] += np.random.normal(0, 1, self.dim)
                    if np.random.rand() < 0.3:
                        partner_idx = np.random.randint(self.particles_per_swarm)
                        alpha = np.random.rand(self.dim)
                        positions[swarm_id][i] = alpha * positions[swarm_id][i] + (1 - alpha) * personal_best_positions[swarm_id][partner_idx]
                    velocities[swarm_id][i] = np.clip(velocities[swarm_id][i], -20 * (1 - evaluations/self.budget), 20 * (1 - evaluations/self.budget))
                    positions[swarm_id][i] += velocities[swarm_id][i]
                    
                    positions[swarm_id][i] = np.clip(positions[swarm_id][i], self.lower_bound, self.upper_bound)
        
        return global_best_score, global_best_position