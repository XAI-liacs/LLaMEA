import numpy as np

class DynamicNeighborhoodPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = 50
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive (particle) coefficient
        self.c2 = 1.5  # social (swarm) coefficient

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.Inf)

        evaluations = 0
        while evaluations < self.budget:
            # Evaluate current positions
            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break
                f = func(positions[i])
                evaluations += 1
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = positions[i]
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = positions[i]

            # Determine the neighborhood size dynamically
            neighborhood_size = max(1, int(self.num_particles * (1 - evaluations / self.budget)))

            # Update velocities and positions
            for i in range(self.num_particles):
                # Determine neighborhood best
                neighbors = np.random.choice(self.num_particles, neighborhood_size, replace=False)
                neighborhood_best = min(neighbors, key=lambda j: personal_best_scores[j])
                # Update velocity
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * np.random.rand() * (personal_best_positions[neighborhood_best] - positions[i]))
                # Update position
                positions[i] += velocities[i]
                # Ensure the particles stay within bounds
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)

        return self.f_opt, self.x_opt