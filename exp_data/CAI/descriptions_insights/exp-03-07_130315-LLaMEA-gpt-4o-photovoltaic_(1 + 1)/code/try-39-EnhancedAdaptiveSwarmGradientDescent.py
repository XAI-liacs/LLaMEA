import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        # Constraint on function evaluations
        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.6 * (0.9 - adaptive_factor)  # Changed line 1
            cognitive_coeff = 1.9 * adaptive_factor  # Changed line 2
            social_coeff = 1.6 + 0.3 * adaptive_factor  # Changed line 3

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                
                neighborhood_best = personal_best[np.random.randint(self.population_size)]
                gradient_attraction = 0.1 * (global_best - swarm[i])  # Changed line 4
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * gradient_attraction +  # Changed line 5
                                    np.random.random(self.dim) * (neighborhood_best - swarm[i]))
                velocity_scale = 0.6 + 0.4 * np.random.random()  # Changed line 6
                swarm[i] += velocity_scale * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate and update personal best
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = 0.9 * swarm[i] + 0.1 * personal_best[i]  # Changed line 7
                    personal_best_value[i] = f_value

                # Local search enhancement
                if evaluations < self.budget and np.random.rand() < 0.1:  # Changed line 8
                    local_search_point = swarm[i] + np.random.normal(0, 0.1, self.dim)  # Changed line 9
                    local_search_point = np.clip(local_search_point, lb, ub)  # Changed line 10
                    f_local = func(local_search_point)
                    evaluations += 1
                    if f_local < f_value:
                        swarm[i] = local_search_point
                        f_value = f_local

                # Update global best
                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value