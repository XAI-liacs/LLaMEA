import numpy as np

class AdaptiveMultiSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, num_swarms=5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_swarms = num_swarms
        self.bounds = (-5.0, 5.0)
        self.global_best_position = None
        self.global_best_value = np.inf

    def levy_flight(self, lam=1.5, size=1):
        sigma1 = np.power((np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2)) /
                          (np.math.gamma((1 + lam) / 2) * lam * np.power(2, (lam - 1) / 2)), 1/lam)
        u = np.random.normal(0, sigma1, size)
        v = np.random.normal(0, 1, size)
        step = u / np.power(np.abs(v), 1/lam)
        return step

    def initialize_particles(self):
        self.particles = [
            {
                'position': np.random.uniform(*self.bounds, self.dim),
                'velocity': np.random.uniform(-1, 1, self.dim),
                'best_position': None,
                'best_value': np.inf
            } for _ in range(self.num_particles)
        ]

    def update_particle(self, particle, swarm_best_position):
        inertia = 0.5
        local_attraction = 1.5
        global_attraction = 1.5

        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_component = local_attraction * r1 * (particle['best_position'] - particle['position'])
        social_component = global_attraction * r2 * (swarm_best_position - particle['position'])
        
        particle['velocity'] = inertia * particle['velocity'] + cognitive_component + social_component
        particle['position'] += particle['velocity']

        # Apply Levy flights for better exploration
        if np.random.rand() < 0.2:
            particle['position'] += self.levy_flight(size=self.dim)

        # Ensure the new position is within bounds
        particle['position'] = np.clip(particle['position'], *self.bounds)

    def __call__(self, func):
        evaluations = 0
        self.initialize_particles()

        while evaluations < self.budget:
            for swarm_idx in range(self.num_swarms):
                swarm_particles = self.particles[swarm_idx::self.num_swarms]
                swarm_best_value = np.inf
                swarm_best_position = None

                for particle in swarm_particles:
                    if particle['best_position'] is None:
                        particle['best_position'] = particle['position'].copy()

                    fitness_value = func(particle['position'])
                    evaluations += 1

                    if fitness_value < particle['best_value']:
                        particle['best_value'] = fitness_value
                        particle['best_position'] = particle['position'].copy()

                    if fitness_value < swarm_best_value:
                        swarm_best_value = fitness_value
                        swarm_best_position = particle['position'].copy()

                    if fitness_value < self.global_best_value:
                        self.global_best_value = fitness_value
                        self.global_best_position = particle['position'].copy()

                    if evaluations >= self.budget:
                        break

                for particle in swarm_particles:
                    self.update_particle(particle, swarm_best_position)

        return self.global_best_position, self.global_best_value