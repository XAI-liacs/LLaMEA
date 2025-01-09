import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best_value = np.min(personal_best_values)
        global_best = particles[np.argmin(personal_best_values)]
        
        evaluations = self.num_particles

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            velocities = (w * velocities +
                          self.c1 * r1 * (personal_best - particles) +
                          self.c2 * r2 * (global_best - particles))
            particles += velocities
            particles = np.clip(particles, bounds[0], bounds[1])  # Ensure particles are within bounds
            
            fitness_values = np.array([func(p) for p in particles])
            evaluations += self.num_particles

            for i in range(self.num_particles):
                if fitness_values[i] < personal_best_values[i]:
                    personal_best_values[i] = fitness_values[i]
                    personal_best[i] = particles[i]
            
            if np.min(fitness_values) < global_best_value:
                global_best_value = np.min(fitness_values)
                global_best = particles[np.argmin(fitness_values)]

        self.f_opt = global_best_value
        self.x_opt = global_best
        return self.f_opt, self.x_opt