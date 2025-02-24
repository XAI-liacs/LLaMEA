import numpy as np

class EnhancedAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_inertia_weight = 0.9
        self.min_inertia_weight = 0.4
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.best_global_position = None
        self.best_global_value = float('-inf')
        self.max_velocity = None
        self.success_threshold = 0.05  # Threshold to adaptively adjust velocity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.max_velocity = 0.1 * (ub - lb)
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, float('-inf'))
        neighborhood_size = self.num_particles // 5

        evaluations = 0
        while evaluations < self.budget:
            inertia_weight = self.max_inertia_weight - (self.max_inertia_weight - self.min_inertia_weight) * (evaluations / self.budget)
            for i in range(self.num_particles):
                value = func(particles[i])
                evaluations += 1

                if value > personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = particles[i]

                if value > self.best_global_value:
                    self.best_global_value = value
                    self.best_global_position = particles[i]

            for i in range(self.num_particles):
                neighborhood = np.random.choice(self.num_particles, size=neighborhood_size, replace=False)
                local_best_position = max(neighborhood, key=lambda x: personal_best_values[x])
                local_best_position = personal_best_positions[local_best_position]
                
                r1, r2 = np.random.random(2)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = self.social_param * r2 * (self.best_global_position - particles[i])
                
                if personal_best_values[i] < self.best_global_value * (1 - self.success_threshold):
                    velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity
                else:
                    velocities[i] = (inertia_weight * velocities[i] + 
                                     self.cognitive_param * r1 * (local_best_position - particles[i]) +
                                     social_velocity)
                    
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

        return self.best_global_position