import numpy as np
import scipy.special as sp

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

    def levy_flight(self, dim, scale):
        beta = 1.5
        sigma = (sp.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (sp.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = scale * u / abs(v)**(1 / beta)  # Change 1
        return step

    def chaotic_local_search(self, position, lb, ub, evaluations):
        a = 0.5
        b = 3.0
        amplitude = 0.01 * (1 + 0.5 * evaluations / self.budget)  # Change 2
        chaotic_sequence = a + (b - a) * np.random.rand(self.dim)
        new_position = np.clip(position + chaotic_sequence * (ub - lb) * amplitude, lb, ub)  # Change 3
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
            self.inertia_weight = 0.9 - 0.5 / (1 + np.exp(-(10 * evaluations / self.budget - 5)))
            self.mutation_factor = self.initial_mutation_factor + 0.2 * (evaluations / self.budget)

            for i in range(swarm_size):
                scale = 1 - evaluations / self.budget  # Additional parameter for adaptability
                if np.random.rand() < 0.5:
                    swarm[i] += self.levy_flight(self.dim, scale)
                else:
                    swarm[i] = self.chaotic_local_search(swarm[i], lb, ub, evaluations)
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
                swarm_size = max(self.min_swarm_size, swarm_size - 5)
                swarm = swarm[:swarm_size]
                velocities = velocities[:swarm_size]
                personal_best_positions = personal_best_positions[:swarm_size]
                personal_best_scores = personal_best_scores[:swarm_size]

        return global_best_position, global_best_score