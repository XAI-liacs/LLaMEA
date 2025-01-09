import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
        self.w = 0.7   # inertia weight
        self.F = 0.5   # differential weight for DE
        self.CR = 0.9  # crossover probability for DE
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        np.random.seed(42)  # for reproducibility

        # Initialize particle positions and velocities
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        iterations = self.budget // self.swarm_size
        for _ in range(iterations):
            for i in range(self.swarm_size):
                # Update velocity and position using PSO rules
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Differential Evolution Mutation and Crossover
                indices = list(range(self.swarm_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = positions[a] + self.F * (positions[b] - positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, positions[i])

                # Evaluate trial vector
                trial_score = func(trial_vector)

                # Select the better solution
                current_score = func(positions[i])
                if trial_score < current_score:
                    positions[i] = trial_vector
                    current_score = trial_score

                # Update personal best
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = positions[i]

        return global_best_position, global_best_score