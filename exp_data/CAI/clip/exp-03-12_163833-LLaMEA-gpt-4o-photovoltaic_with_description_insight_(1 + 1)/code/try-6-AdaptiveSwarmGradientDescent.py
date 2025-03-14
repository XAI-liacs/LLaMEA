import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.subpop_size = self.population_size // 2  # New attribute for subpopulation size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        # Constraint on function evaluations
        evaluations = self.population_size

        def subpop_refine(subpop, velocity):  # Refine subpopulation
            inertia_weight = 0.9
            cognitive_coeff = 1.5
            social_coeff = 1.5
            for i in range(self.subpop_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                velocity[i] = (inertia_weight * velocity[i] +
                               cognitive_coeff * r1 * (personal_best[i] - subpop[i]) +
                               social_coeff * r2 * (global_best - subpop[i]))
                subpop[i] += velocity[i]
                subpop[i] = np.clip(subpop[i], lb, ub)
                f_value = func(subpop[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = subpop[i]
                    personal_best_value[i] = f_value
                if f_value < global_best_value:
                    global_best = subpop[i]
                    global_best_value = f_value
                if evaluations >= self.budget:
                    break
            return evaluations

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            dynamic_lr = 0.1 + 0.5 * adaptive_factor
            subpop = swarm[:self.subpop_size].copy()
            evaluations = subpop_refine(subpop, self.velocity[:self.subpop_size])

            for i in range(self.subpop_size, self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (0.9 * adaptive_factor * self.velocity[i] +
                                    1.5 * adaptive_factor * r1 * (personal_best[i] - swarm[i]) +
                                    1.5 * r2 * (global_best - swarm[i]))
                swarm[i] += dynamic_lr * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value
                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value
                if evaluations >= self.budget:
                    break

        return global_best, global_best_value