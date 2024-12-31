import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia=0.5, cognitive=2, social=2):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None
        self.velocity = np.random.uniform(-1, 1, (swarm_size, dim))
        self.positions = np.random.uniform(-5, 5, (swarm_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(swarm_size, np.Inf)

    def __call__(self, func):
        func_evals = 0
        while func_evals < self.budget:
            for i in range(self.swarm_size):
                if func_evals >= self.budget:
                    break
                
                score = func(self.positions[i])
                func_evals += 1

                # Update personal best with adaptive inertia
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                    self.inertia = max(0.4, self.inertia * 0.99)  # Adapt inertia based on improvement
                
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = self.positions[i]

            # Update velocities and positions
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.cognitive * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.social * r2 * (self.x_opt - self.positions[i])
                self.velocity[i] = self.inertia * self.velocity[i] + cognitive_component + social_component
                self.positions[i] += self.velocity[i]
                
                # Ensure particles remain within bounds
                self.positions[i] = np.clip(self.positions[i], -5, 5)
        
        return self.f_opt, self.x_opt