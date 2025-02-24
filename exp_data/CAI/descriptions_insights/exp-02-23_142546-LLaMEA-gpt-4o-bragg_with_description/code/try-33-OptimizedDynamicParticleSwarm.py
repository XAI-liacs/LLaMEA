import numpy as np

class OptimizedDynamicParticleSwarm:
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

    def chaotic_initialization(self, lb, ub):
        chaotic_map = np.random.uniform(0, 1, (self.num_particles, self.dim))
        chaotic_map = np.sin(np.pi * chaotic_map)
        return lb + chaotic_map * (ub - lb)

    def adaptive_neighborhood(self, particle_index, particles, personal_best_positions):
        neighborhood_size = min(5, self.num_particles - 1)
        indices = np.random.choice(np.delete(np.arange(self.num_particles), particle_index), neighborhood_size, replace=False)
        neighborhood_best = max(indices, key=lambda i: func(personal_best_positions[i]))
        return neighborhood_best

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.max_velocity = 0.1 * (ub - lb)
        particles = self.chaotic_initialization(lb, ub)
        velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, float('-inf'))

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
                r1, r2 = np.random.random(2)
                neighborhood_best = personal_best_positions[self.adaptive_neighborhood(i, particles, personal_best_positions)]
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = self.social_param * r2 * (neighborhood_best - particles[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                
                # Introduce random reinitialization for diversity 
                if np.random.rand() < 0.05:  
                    particles[i] = np.random.uniform(lb, ub, self.dim)

        return self.best_global_position