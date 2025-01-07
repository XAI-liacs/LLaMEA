import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 55
        self.w_min = 0.4  # minimum inertia weight
        self.w_max = 0.9  # maximum inertia weight
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
        iteration = 0
        while evals < self.budget:
            iteration += 1

            # Dynamic inertia weight adjustment
            w = self.w_max - (iteration / (self.budget // self.population_size)) * (self.w_max - self.w_min)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            swarm_velocities = (w * swarm_velocities +
                                self.c1 * r1 * (personal_best_positions - swarm_positions) +
                                self.c2 * r2 * (global_best_position - swarm_positions))
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
                acceptance_probability = np.exp((scores[i] - candidate_score) / (self.temperature + 1e-9))
                if candidate_score < scores[i] or acceptance_probability > np.random.rand():
                    swarm_positions[i] = candidate_pos
                    scores[i] = candidate_score
                    if candidate_score < personal_best_scores[i]:
                        personal_best_positions[i] = candidate_pos
                        personal_best_scores[i] = candidate_score
                        if candidate_score < global_best_score:
                            global_best_score = candidate_score
                            global_best_position = candidate_pos

            self.temperature *= self.cooling_rate

        return global_best_position