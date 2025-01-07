import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 55
        self.w = 0.6
        self.c1 = 1.6
        self.c2 = 1.8
        self.temperature = 100
        self.cooling_rate = 0.98

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        swarm_velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm_positions)
        personal_best_scores = np.array([func(x) for x in swarm_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evals = self.population_size

        while evals < self.budget:
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            adaptive_c1 = self.c1 * (1 - evals / self.budget)
            adaptive_c2 = self.c2 * (evals / self.budget)
            swarm_velocities = (self.w * swarm_velocities +
                                adaptive_c1 * r1 * (personal_best_positions - swarm_positions) +
                                adaptive_c2 * r2 * (global_best_position - swarm_positions))
            swarm_positions = np.clip(swarm_positions + swarm_velocities, lb, ub)

            scores = np.array([func(x) for x in swarm_positions])
            evals += self.population_size

            improved = scores < personal_best_scores
            personal_best_positions[improved] = swarm_positions[improved]
            personal_best_scores[improved] = scores[improved]
            if min(scores) < global_best_score:
                global_best_score = min(scores)
                global_best_position = swarm_positions[np.argmin(scores)]

            for i in range(self.population_size):
                candidate_pos = swarm_positions[i] + np.random.uniform(-0.5, 0.5, self.dim) * self.temperature
                candidate_pos = np.clip(candidate_pos, lb, ub)
                candidate_score = func(candidate_pos)
                evals += 1
                if candidate_score < scores[i] or np.exp((scores[i] - candidate_score) / self.temperature) > np.random.rand():
                    swarm_positions[i] = candidate_pos
                    scores[i] = candidate_score
                    if candidate_score < personal_best_scores[i]:
                        personal_best_positions[i] = candidate_pos
                        personal_best_scores[i] = candidate_score
                        if candidate_score < global_best_score:
                            global_best_score = candidate_score
                            global_best_position = candidate_pos

            opposite_positions = lb + ub - swarm_positions
            opposite_positions = np.clip(opposite_positions, lb, ub)
            opposite_scores = np.array([func(x) for x in opposite_positions])
            evals += self.population_size

            replace = opposite_scores < scores
            swarm_positions[replace] = opposite_positions[replace]
            scores[replace] = opposite_scores[replace]

            self.temperature *= self.cooling_rate

        return global_best_position