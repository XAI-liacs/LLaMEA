import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize particle positions and velocities
        particles = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in personal_best_positions])

        # Initialize global best
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        evaluations = self.swarm_size

        # Function to dynamically adjust neighborhood topology
        def get_neighbors(index, size):
            neighbors = [(index + i) % size for i in range(-1, 2)]
            return neighbors

        # PSO loop
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity and position
                neighbors = get_neighbors(i, self.swarm_size)
                local_best = min(neighbors, key=lambda n: personal_best_scores[n])
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (0.5 * velocities[i] +
                                 r1 * (personal_best_positions[i] - particles[i]) +
                                 r2 * (personal_best_positions[local_best] - particles[i]))
                particles[i] += velocities[i]

                # Clamp to bounds
                particles[i] = np.clip(particles[i], self.lb, self.ub)

                # Evaluate and update personal best
                score = func(particles[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

                # Check budget
                if evaluations >= self.budget:
                    break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt