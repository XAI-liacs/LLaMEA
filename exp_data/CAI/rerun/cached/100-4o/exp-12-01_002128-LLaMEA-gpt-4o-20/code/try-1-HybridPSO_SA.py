import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30  # Number of particles in the swarm
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.w = 0.7   # Inertia weight
        self.temperature = 100.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize particles' positions and velocities
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf

        # Evaluation count
        eval_count = 0

        while eval_count < self.budget:
            for i in range(self.num_particles):
                # Evaluate fitness
                score = func(positions[i])
                eval_count += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            for i in range(self.num_particles):
                # Update velocity and position
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Simulated Annealing perturbation
                if eval_count < self.budget:
                    new_position = positions[i] + np.random.normal(0, self.temperature, self.dim)
                    new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                    new_score = func(new_position)
                    eval_count += 1
                    if new_score < personal_best_scores[i] or np.exp((personal_best_scores[i] - new_score) / self.temperature) > np.random.rand():
                        positions[i] = new_position
                        personal_best_scores[i] = new_score
                        personal_best_positions[i] = new_position
                        if new_score < global_best_score:
                            global_best_score = new_score
                            global_best_position = new_position

            # Cool down temperature
            self.temperature *= self.cooling_rate

        return global_best_position, global_best_score