import numpy as np

class HybridBeeParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_bees = 30
        self.num_particles = 30
        self.best_solution = None
        self.best_value = float('-inf')
        self.max_velocity = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.max_velocity = 0.1 * (ub - lb)
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, float('-inf'))

        bees = np.random.uniform(lb, ub, (self.num_bees, self.dim))
        bee_values = np.full(self.num_bees, float('-inf'))

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                value = func(particles[i])
                evaluations += 1

                if value > personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = particles[i]

                if value > self.best_value:
                    self.best_value = value
                    self.best_solution = particles[i]

            for i in range(self.num_bees):
                value = func(bees[i])
                evaluations += 1
                
                if value > bee_values[i]:
                    bee_values[i] = value

                if value > self.best_value:
                    self.best_value = value
                    self.best_solution = bees[i]
            
            # Particle Swarm Update
            inertia_weight = 0.5 + 0.4 * np.random.random()
            cognitive_param = 1.5
            social_param = 1.5
            for i in range(self.num_particles):
                r1, r2 = np.random.random(2)
                cognitive_velocity = cognitive_param * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = social_param * r2 * (self.best_solution - particles[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

            # Bee Phase: Random exploration and exploitation
            best_bee_idx = np.argmax(bee_values)
            for i in range(self.num_bees):
                if i == best_bee_idx:
                    continue
                phi = np.random.uniform(-1, 1, self.dim)
                bees[i] = np.clip(bees[i] + phi * (bees[i] - bees[best_bee_idx]), lb, ub)
                
            # Scout bee phase: Random reset
            if evaluations / self.budget > 0.7:
                for i in range(self.num_bees):
                    if np.random.rand() < 0.1:
                        bees[i] = np.random.uniform(lb, ub, self.dim)

        return self.best_solution