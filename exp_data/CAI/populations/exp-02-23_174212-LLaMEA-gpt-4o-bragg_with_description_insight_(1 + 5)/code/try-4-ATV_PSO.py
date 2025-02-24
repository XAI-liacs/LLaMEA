import numpy as np

class ATV_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.velocity_clamp = 0.1
        self.n_iterations = budget // self.population_size
        self.velocity = np.zeros((self.population_size, self.dim))

    def initialize_particles(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evaluate_particles(self, func, particles):
        return np.array([func(ind) for ind in particles])
    
    def update_velocity(self, particle, velocity, personal_best, global_best):
        inertia = self.inertia_weight * velocity
        cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best - particle)
        social_component = self.social_coeff * np.random.rand(self.dim) * (global_best - particle)
        new_velocity = inertia + cognitive_component + social_component
        return np.clip(new_velocity, -self.velocity_clamp, self.velocity_clamp)
    
    def update_particle(self, particle, velocity, lb, ub):
        new_position = particle + velocity
        return np.clip(new_position, lb, ub)
    
    def enforce_periodicity(self, particle):
        period = np.random.randint(1, self.dim // 2)
        for j in range(0, self.dim, period):
            particle[j:j+period] = particle[:period]
        return particle

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = self.initialize_particles(lb, ub)
        personal_best = particles.copy()
        personal_best_values = self.evaluate_particles(func, particles)
        global_best = particles[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        for iteration in range(self.n_iterations):
            for i in range(self.population_size):
                self.velocity[i] = self.update_velocity(particles[i], self.velocity[i], personal_best[i], global_best)
                particles[i] = self.update_particle(particles[i], self.velocity[i], lb, ub)
                particles[i] = self.enforce_periodicity(particles[i])

            fitness = self.evaluate_particles(func, particles)
            for i in range(self.population_size):
                if fitness[i] < personal_best_values[i]:
                    personal_best_values[i] = fitness[i]
                    personal_best[i] = particles[i]
            
            current_global_best_value = np.min(personal_best_values)
            if current_global_best_value < global_best_value:
                global_best_value = current_global_best_value
                global_best = personal_best[np.argmin(personal_best_values)]

            # Adapt inertia weight for better exploration and exploitation balance
            self.inertia_weight = 0.9 - (0.5 * (iteration / self.n_iterations))

        return global_best