import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = 0.7  # Initial inertia
        self.inertia_min = 0.4  # Minimum inertia value
        self.inertia_max = 0.9  # Maximum inertia value
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.v_max = 0.2 * (5.0 - (-5.0))  # Velocity clamping
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.swarm_size, np.Inf)

        global_best_position = None
        global_best_value = np.Inf

        eval_count = 0

        while eval_count < self.budget:
            for i in range(self.swarm_size):
                if eval_count >= self.budget:
                    break
                # Evaluate fitness
                fitness = func(particles[i])
                eval_count += 1

                # Update personal best
                if fitness < personal_best_values[i]:
                    personal_best_values[i] = fitness
                    personal_best_positions[i] = particles[i].copy()

                # Update global best
                if fitness < global_best_value:
                    global_best_value = fitness
                    global_best_position = particles[i].copy()

            # Update particles' velocities and positions
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            velocities = (
                self.inertia * velocities
                + self.c1 * r1 * (personal_best_positions - particles)
                + self.c2 * r2 * (global_best_position - particles)
            )

            # Velocity clamping
            velocities = np.clip(velocities, -self.v_max, self.v_max)

            # Update positions
            particles += velocities

            # Ensure particles are within bounds
            particles = np.clip(particles, -5.0, 5.0)

            # Adapt inertia
            self.inertia = self.inertia_max - (self.inertia_max - self.inertia_min) * (
                eval_count / self.budget
            )

        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt