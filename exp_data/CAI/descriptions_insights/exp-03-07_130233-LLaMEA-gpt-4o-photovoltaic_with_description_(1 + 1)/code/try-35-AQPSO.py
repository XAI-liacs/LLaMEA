import numpy as np

class AQPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.positions = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(self.swarm_size, np.inf)
        self.gbest_position = np.zeros(self.dim)
        self.gbest_score = np.inf
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            for i in range(self.swarm_size):
                if self.func_evals >= self.budget:
                    break
                score = func(self.positions[i])
                self.func_evals += 1
                
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i]
                    
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i]

            inertia_weight = 0.5 + np.random.rand() / 2 - (self.func_evals / self.budget) * 0.5
            improvement_rate = (self.gbest_score - score) / max(1, abs(self.gbest_score))
            social_scale = 1 + 0.5 * improvement_rate
            cognitive_scale = 0.5 + 0.5 * improvement_rate
            fitness_variance = np.std(self.pbest_scores)  # Added line for variance calculation
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = cognitive_scale * r1 * (self.pbest_positions[i] - self.positions[i])
                social_component = r2 * social_scale * (self.gbest_position - self.positions[i])
                velocity = self.velocities[i] + cognitive_component + social_component
                self.velocities[i] = inertia_weight * np.clip(velocity, func.bounds.lb/2, func.bounds.ub/2)
                
                quantum_prob = 0.5 + 0.6 * (self.gbest_score - score) / max(1, fitness_variance)  # Modified line for fitness variance
                quantum_behavior = np.random.uniform(-1, 1, self.dim) * (self.gbest_position - self.positions[i])
                if np.random.rand() < quantum_prob:
                    self.positions[i] = self.gbest_position + quantum_behavior
                else:
                    self.positions[i] = self.positions[i] + self.velocities[i] * (1 + 0.1 * improvement_rate)
            
            self.positions = np.clip(self.positions, func.bounds.lb, func.bounds.ub)

        return self.gbest_position, self.gbest_score