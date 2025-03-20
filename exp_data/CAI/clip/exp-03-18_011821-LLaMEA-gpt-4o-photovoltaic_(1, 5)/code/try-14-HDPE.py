import numpy as np

class HDPE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.inertia_weight = 0.9  # Initial inertia weight, changed
        self.mutation_factor = 0.5  # Differential evolution mutation factor
        self.crossover_prob = 0.7  # Differential evolution crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_position = swarm[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            # Adaptive inertia weight, changed
            self.inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)

            # Update velocities and positions
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.c1 * r1 * (personal_best_positions - swarm) +
                          self.c2 * r2 * (global_best_position - swarm))
            swarm = np.clip(swarm + velocities, lb, ub)

            # Differential Evolution mutation and crossover
            self.mutation_factor = 0.5 + 0.3 * (evaluations / self.budget)  # Adaptive mutation factor based on budget utilization
            self.crossover_prob = 0.7 + 0.3 * (1 - evaluations / self.budget) 
            for i in range(self.swarm_size):
                a, b, c = np.random.choice(self.swarm_size, 3, replace=False)
                # Biodiversity preservation: ensure distinct selections, changed
                while len({a, b, c, i}) < 4:
                    a, b, c = np.random.choice(self.swarm_size, 3, replace=False)
                mutant_vector = np.clip(swarm[a] +
                                        self.mutation_factor * (swarm[b] - swarm[c]), lb, ub)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob,
                                        mutant_vector, swarm[i])

                # Evaluate new trial vector
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best_position = trial_vector
                        global_best_score = trial_score
                        if evaluations >= self.budget:
                            break

        return global_best_position, global_best_score