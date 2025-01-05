import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 50
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
        self.initial_alpha = 0.9  # initial inertia weight
        self.final_alpha = 0.4  # final inertia weight
        self.alpha = self.initial_alpha
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.5
        self.mutation_factor = self.initial_mutation_factor
        self.crossover_rate = 0.9
        self.velocity_clip = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-self.velocity_clip, self.velocity_clip, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.pop_size

        while evaluations < self.budget:
            self.alpha = self.initial_alpha - (self.initial_alpha - self.final_alpha) * (evaluations / self.budget)
            self.mutation_factor = self.initial_mutation_factor - (
                    self.initial_mutation_factor - self.final_mutation_factor) * (evaluations / self.budget)
            for i in range(self.pop_size):
                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = np.clip(
                    (self.alpha * velocities[i] +
                     self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                     self.c2 * r2 * (global_best_position - positions[i])),
                    -self.velocity_clip, self.velocity_clip)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # DE local search
                if np.random.rand() < self.crossover_rate and evaluations + 3 <= self.budget:
                    candidates = np.random.choice(self.pop_size, 3, replace=False)
                    mutant = (positions[candidates[0]] +
                              self.mutation_factor * (positions[candidates[1]] - positions[candidates[2]]))
                    mutant = np.clip(mutant, lb, ub)
                    trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, positions[i])
                    trial_score = func(trial)
                    evaluations += 1
                    if trial_score < personal_best_scores[i]:
                        positions[i] = trial
                        personal_best_scores[i] = trial_score

                # Evaluate the new position
                score = func(positions[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                # Update optimal solutions if budget allows
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = positions[i]

        return self.f_opt, self.x_opt