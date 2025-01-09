import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(40, budget // 10)
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w_max = 0.9
        self.w_min = 0.4
        self.global_best_position = None
        self.global_best_value = np.inf
        self.particles = self.initialize_particles()

    def initialize_particles(self):
        particles = []
        for _ in range(self.population_size):
            position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            velocity = np.random.uniform(-1, 1, self.dim)
            best_position = position.copy()
            best_value = np.inf
            particles.append({'position': position, 'velocity': velocity, 'best_position': best_position, 'best_value': best_value})
        return particles

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            for particle in self.particles:
                # Evaluate current position
                current_value = func(particle['position'])
                evaluations += 1
                if evaluations >= self.budget:
                    break

                # Update personal best
                if current_value < particle['best_value']:
                    particle['best_value'] = current_value
                    particle['best_position'] = particle['position'].copy()

                # Update global best
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = particle['position'].copy()

            # Update particle velocities and positions
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            for particle in self.particles:
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (particle['best_position'] - particle['position'])
                social_velocity = self.c2 * r2 * (self.global_best_position - particle['position'])
                quantum_velocity = np.random.normal(0, 1, self.dim) * (self.upper_bound - self.lower_bound) * 0.1
                particle['velocity'] = w * particle['velocity'] + cognitive_velocity + social_velocity + quantum_velocity
                
                # Update position
                particle['position'] += particle['velocity']
                particle['position'] = np.clip(particle['position'], self.lower_bound, self.upper_bound)

        return self.global_best_position