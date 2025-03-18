import numpy as np

class AdaptiveSwarmGradientDescentEnhancedV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.sub_swarm_count = 3  # Change 1: Introduce multiple sub-swarms

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size
        chaos_factor = np.sin(np.arange(self.budget) / self.budget * np.pi)  # Change 2: Chaotic sequence

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.4 + 0.2 * chaos_factor[evaluations]  # Change 3: Chaotic inertia weight
            cognitive_coeff = 1.5 * adaptive_factor  # Change 4: Adjusted coefficient
            social_coeff = 1.9 + 0.2 * (1 - adaptive_factor)  # Change 5: Adjusted coefficient

            # Change 6: Multi-swarm dynamic adaptation
            for s in range(self.sub_swarm_count):
                start_idx = s * (self.population_size // self.sub_swarm_count)
                end_idx = start_idx + (self.population_size // self.sub_swarm_count)
                sub_swarm = swarm[start_idx:end_idx]
                sub_velocity = self.velocity[start_idx:end_idx]

                for i in range(len(sub_swarm)):
                    r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                    sub_velocity[i] = (inertia_weight * sub_velocity[i] +
                                       cognitive_coeff * r1 * (personal_best[start_idx + i] - sub_swarm[i]) +
                                       social_coeff * r2 * (global_best - sub_swarm[i]))
                    max_velocity = (ub - lb) * 0.1
                    sub_velocity[i] = np.clip(sub_velocity[i], -max_velocity, max_velocity)
                    sub_swarm[i] += sub_velocity[i]
                    sub_swarm[i] = np.clip(sub_swarm[i], lb, ub)

                    f_value = func(sub_swarm[i])
                    evaluations += 1
                    if f_value < personal_best_value[start_idx + i]:
                        personal_best[start_idx + i] = sub_swarm[i]
                        personal_best_value[start_idx + i] = f_value

                    if f_value < global_best_value:
                        global_best = sub_swarm[i]
                        global_best_value = f_value

                    if evaluations >= self.budget:
                        break

        return global_best, global_best_value