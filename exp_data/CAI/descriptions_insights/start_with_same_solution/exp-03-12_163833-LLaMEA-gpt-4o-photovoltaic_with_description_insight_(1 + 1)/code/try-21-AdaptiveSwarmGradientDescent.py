import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3  # Introducing multiple swarms
        self.population_size = (10 + 2 * int(np.sqrt(dim))) // self.num_swarms
        self.velocity = np.zeros((self.num_swarms, self.population_size, dim))
        self.hierarchical_learning_rate = 0.1  # New hierarchical learning rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarms = [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best = [swarm.copy() for swarm in swarms]
        personal_best_value = [np.array([func(x) for x in swarm]) for swarm in swarms]
        global_best = min((pb[np.argmin(pv)] for pb, pv in zip(personal_best, personal_best_value)), key=func)
        global_best_value = func(global_best)

        evaluations = sum(map(len, personal_best_value))

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            convergence_factor = np.log10(evaluations + 10) / np.log10(self.budget)
            inertia_weight = (0.9 * adaptive_factor + 0.1) * convergence_factor
            cognitive_coeff = 1.5 * adaptive_factor
            social_coeff = 1.5
            dynamic_lr = 0.05 + 0.45 * adaptive_factor
            velocity_scale = 1 + 0.5 * (1 - adaptive_factor)**2

            for swarm_idx in range(self.num_swarms):
                for i in range(self.population_size):
                    r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                    self.velocity[swarm_idx][i] = (inertia_weight * self.velocity[swarm_idx][i] +
                                                   cognitive_coeff * r1 * (personal_best[swarm_idx][i] - swarms[swarm_idx][i]) +
                                                   social_coeff * r2 * (global_best - swarms[swarm_idx][i]))
                    swarms[swarm_idx][i] += velocity_scale * dynamic_lr * self.velocity[swarm_idx][i]
                    swarms[swarm_idx][i] = np.clip(swarms[swarm_idx][i], lb, ub)

                    # Evaluate and update personal best
                    f_value = func(swarms[swarm_idx][i])
                    evaluations += 1
                    if f_value < personal_best_value[swarm_idx][i]:
                        personal_best[swarm_idx][i] = swarms[swarm_idx][i]
                        personal_best_value[swarm_idx][i] = f_value

                    # Update global best
                    if f_value < global_best_value:
                        global_best = swarms[swarm_idx][i]
                        global_best_value = f_value

                    if evaluations >= self.budget:
                        break

                # Hierarchical learning between swarms
                for other_swarm_idx in range(self.num_swarms):
                    if other_swarm_idx != swarm_idx:
                        for i in range(self.population_size):
                            diff = personal_best[other_swarm_idx][i] - personal_best[swarm_idx][i]
                            swarms[swarm_idx][i] += self.hierarchical_learning_rate * diff

        return global_best, global_best_value