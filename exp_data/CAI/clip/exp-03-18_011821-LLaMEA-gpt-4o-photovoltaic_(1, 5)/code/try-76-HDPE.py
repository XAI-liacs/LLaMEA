import numpy as np
import scipy.special as sp

class HDPE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 50
        self.min_swarm_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.9
        self.initial_mutation_factor = 0.4
        self.crossover_prob = 0.7
        self.mutation_prob = 0.1

    def levy_flight(self, dim):
        beta = 1.5
        sigma = (sp.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (sp.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / abs(v)**(1 / beta)
        return step

    def chaotic_local_search(self, position, lb, ub, scale_factor=0.015):
        a = 0.5
        b = 3.0
        chaotic_sequence = a + (b - a) * np.random.rand(self.dim)
        new_position = np.clip(position + chaotic_sequence * (ub - lb) * scale_factor, lb, ub)
        return new_position

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
            self.inertia_weight = 0.5 + 0.1 * np.sin(5 * np.pi * (evaluations / self.budget))  # Modified line
            self.mutation_factor = self.initial_mutation_factor + 0.15 * (evaluations / self.budget)
            adaptive_scale_factor = 0.02 + 0.03 * (evaluations / self.budget)

            diversity_factor = np.std(swarm, axis=0) / (ub - lb)
            mutation_adjustment = (np.random.rand(self.dim) < diversity_factor).astype(float)

            for i in range(swarm_size):
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - swarm[i]) +
                                 self.c2 * np.random.rand(self.dim) * (global_best_position - swarm[i]))
                velocities[i] = np.clip(velocities[i], -0.5 * abs(ub - lb), 0.5 * abs(ub - lb))

                swarm[i] += velocities[i]
                self.crossover_prob = 0.5 + 0.5 * np.random.rand()
                self.mutation_prob = 0.1 + 0.2 * (1 - evaluations / self.budget)
                if np.random.rand() < self.mutation_prob:
                    swarm[i] += self.levy_flight(self.dim) * mutation_adjustment
                else:
                    swarm[i] = self.chaotic_local_search(swarm[i], lb, ub, adaptive_scale_factor)
                swarm[i] = np.clip(swarm[i], lb, ub)

                trial_score = func(swarm[i])
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = swarm[i]
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best_position = swarm[i]
                        global_best_score = trial_score
                        if evaluations >= self.budget:
                            break

            if evaluations % (self.budget // 10) == 0:
                swarm_size = max(self.min_swarm_size, swarm_size - 4)
                swarm = swarm[:swarm_size]
                velocities = velocities[:swarm_size]
                personal_best_positions = personal_best_positions[:swarm_size]
                personal_best_scores = personal_best_scores[:swarm_size]

            if evaluations % (self.budget // 4) == 0:
                elite_idx = np.argsort(personal_best_scores)[:5]
                swarm[:5] = personal_best_positions[elite_idx]
                swarm[5:] = np.random.uniform(lb, ub, (swarm_size - 5, self.dim))

        return global_best_position, global_best_score