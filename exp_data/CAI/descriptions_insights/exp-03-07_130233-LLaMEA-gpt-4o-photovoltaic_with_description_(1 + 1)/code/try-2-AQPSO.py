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
        inertia_weight = 0.9  # Initial inertia weight
        while self.func_evals < self.budget:
            # Evaluate current position
            for i in range(self.swarm_size):
                if self.func_evals >= self.budget:
                    break
                score = func(self.positions[i])
                self.func_evals += 1
                
                # Update personal best
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i]
                    
                # Update global best
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i]

            # Update velocity and position
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = r1 * (self.pbest_positions[i] - self.positions[i])
                social_component = r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component
                
                # Apply quantum behavior for exploration
                quantum_behavior = np.random.uniform(-1, 1, self.dim) * (self.gbest_position - self.positions[i])
                if np.random.rand() < 0.5:
                    self.positions[i] = self.gbest_position + quantum_behavior
                else:
                    self.positions[i] = self.positions[i] + self.velocities[i]
            
            # Bound check
            self.positions = np.clip(self.positions, func.bounds.lb, func.bounds.ub)
            
            # Dynamic inertia weight update
            inertia_weight = 0.4 + 0.5 * ((self.budget - self.func_evals) / self.budget)

        return self.gbest_position, self.gbest_score