import numpy as np

class HDPE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 50
        self.min_swarm_size = 25
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.9
        self.initial_mutation_factor = 0.5
        self.crossover_prob = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_size = self.initial_swarm_size
        swarm = np.random.uniform(lb, ub, (swarm_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_position = swarm[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = swarm_size

        while evaluations < self.budget:
            self.inertia_weight = 0.9 - 0.5 / (1 + np.exp(-(10 * evaluations / self.budget - 5)))

            # Adaptive mutation factor
            self.mutation_factor = self.initial_mutation_factor + 0.2 * (evaluations / self.budget)

            # Update velocities and positions
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.c1 * r1 * (personal_best_positions - swarm) +
                          self.c2 * r2 * (global_best_position - swarm))
            swarm = np.clip(swarm + velocities, lb, ub)

            self.crossover_prob = 0.7 + 0.3 * (1 - evaluations / self.budget)

            for i in range(swarm_size):
                a, b, c = np.random.choice(swarm_size, 3, replace=False)
                while len({a, b, c, i}) < 4:
                    a, b, c = np.random.choice(swarm_size, 3, replace=False)
                mutant_vector = np.clip(swarm[a] +
                                        self.mutation_factor * (swarm[b] - swarm[c]), lb, ub)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob,
                                        mutant_vector, swarm[i])

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

            # Dynamic swarm size
            if evaluations % (self.budget // 10) == 0:
                swarm_size = max(self.min_swarm_size, swarm_size - 5)
                swarm = swarm[:swarm_size]
                velocities = velocities[:swarm_size]
                personal_best_positions = personal_best_positions[:swarm_size]
                personal_best_scores = personal_best_scores[:swarm_size]

        return global_best_position, global_best_score