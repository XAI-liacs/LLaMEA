import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive (self) weight
        self.c2 = 1.5  # social (group) weight

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(-100, 100, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))

        # Initialize personal bests and global best
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(pos) for pos in positions])
        global_best_value = np.min(personal_best_values)
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]

        # Optimization loop
        evaluations = self.num_particles
        while evaluations < self.budget:
            for i in range(self.num_particles):
                current_value = func(positions[i])
                evaluations += 1
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = positions[i]

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            for i in range(self.num_particles):
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]

            # Adaptive parameter adjustments
            self.w = 0.5 + 0.5 * np.random.rand()

        return global_best_value, global_best_position