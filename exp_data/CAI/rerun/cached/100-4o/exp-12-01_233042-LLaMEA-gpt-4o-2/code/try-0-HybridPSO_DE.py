import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(max(int(budget / 100), 10), 100)  # Adaptive population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Differential evolution scaling factor
        self.CR = 0.9  # Crossover probability
        self.c1 = 1.5  # PSO cognitive coefficient
        self.c2 = 2.0  # PSO social coefficient
        self.w = 0.7   # PSO inertia weight

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.pop_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            # Evaluate current positions
            for i in range(self.pop_size):
                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

            # Update velocities and positions (PSO component)
            if evaluations < self.budget:
                r1 = np.random.rand(self.pop_size, self.dim)
                r2 = np.random.rand(self.pop_size, self.dim)
                velocities = (self.w * velocities + 
                             self.c1 * r1 * (personal_best_positions - positions) +
                             self.c2 * r2 * (global_best_position - positions))
                positions += velocities
                positions = np.clip(positions, self.lower_bound, self.upper_bound)

            # Differential Evolution Mutation and Crossover
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                idxs = np.arange(self.pop_size)
                idxs = np.delete(idxs, i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant_vector = positions[a] + self.F * (positions[b] - positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                trial_vector = np.copy(positions[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial_vector[crossover_points] = mutant_vector[crossover_points]

                # Selection
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial_vector

        return global_best_position, global_best_score