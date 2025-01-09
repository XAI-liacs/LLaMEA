import numpy as np

class HPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 50
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive (particle) weight
        self.c2 = 1.5  # Social (swarm) weight
        self.T_init = 1.0  # Initial temperature for SA
        self.T_min = 0.001  # Minimum temperature for SA
        self.alpha = 0.9  # Cooling rate for SA

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1.0, 1.0, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, np.inf)

        # Evaluate the initial population
        evaluations = 0
        for i in range(self.num_particles):
            score = func(positions[i])
            evaluations += 1
            personal_best_scores[i] = score

        # Find global best
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_score = personal_best_scores[global_best_index]

        # Main optimization loop
        T = self.T_init
        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Update velocities and positions
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Evaluate the new position
                score = func(positions[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                # Simulated Annealing step
                if score < global_best_score or np.exp((global_best_score - score) / T) > np.random.rand():
                    global_best_score = score
                    global_best_position = positions[i].copy()

            # Update temperature
            T = max(self.T_min, self.alpha * T)

        return global_best_position