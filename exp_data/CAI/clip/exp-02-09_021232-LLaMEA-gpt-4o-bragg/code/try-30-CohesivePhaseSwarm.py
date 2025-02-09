import numpy as np

class CohesivePhaseSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.global_best_position = np.zeros(dim)
        self.global_best_fitness = float('inf')
        self.positions = None
        self.velocities = None
        self.particle_best_positions = None
        self.particle_best_fitnesses = np.full(self.num_particles, float('inf'))
        self.phase_offsets = np.random.uniform(0, 2 * np.pi, self.num_particles)

    def initialize_swarm(self, bounds):
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))

    def evaluate(self, func):
        fitnesses = np.array([func(pos) for pos in self.positions])
        for i, fitness in enumerate(fitnesses):
            if fitness < self.particle_best_fitnesses[i]:
                self.particle_best_fitnesses[i] = fitness
                self.particle_best_positions[i] = self.positions[i].copy()
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.positions[i].copy()
        return fitnesses

    def update_positions(self, bounds):
        self.positions += self.velocities
        self.positions = np.clip(self.positions, bounds.lb, bounds.ub)

    def update_velocities(self):
        w = 0.7  # inertia weight
        c1 = 1.5  # cognitive coefficient
        c2 = 1.5  # social coefficient
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = c1 * r1 * (self.particle_best_positions[i] - self.positions[i])
            social_component = c2 * r2 * (self.global_best_position - self.positions[i])
            phase_influence = np.sin(self.phase_offsets[i]) * np.linalg.norm(social_component)
            self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component + phase_influence

    def adjust_phases(self):
        phase_sync_rate = 0.05
        for i in range(self.num_particles):
            self.phase_offsets[i] += np.random.normal(0, phase_sync_rate)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_swarm(bounds)

        evaluations = 0
        while evaluations < self.budget:
            self.evaluate(func)
            self.update_velocities()
            self.update_positions(bounds)
            self.adjust_phases()
            evaluations += self.num_particles  # Each particle's position update counts as an evaluation

        return self.global_best_position