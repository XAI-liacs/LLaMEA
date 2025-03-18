import numpy as np

class AdaptiveSwarmGradientDescent:
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
            inertia_weight = 0.9 - 0.5 * adaptive_factor
            improvement_factor = np.exp(-global_best_value / (np.min(personal_best_value) + 1e-10))
            cognitive_coeff = 1.7 * adaptive_factor * improvement_factor + 0.3 * np.random.random()
            social_coeff = 1.7 * improvement_factor + 0.3 * np.random.random()

            # Dynamic repulsion based on stagnation
            repulsion_coeff = 0.5 * (1 - improvement_factor) * np.random.random()

            velocity_scale = 0.5 + 0.5 * np.random.random()  # Modified line

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = velocity_scale * (  # Modified line
                                    inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) -
                                    repulsion_coeff * (swarm[i] - global_best))

                swarm[i] += self.velocity[i]
                # Dynamic boundary adjustment
                dynamic_lb = lb + (ub - lb) * 0.05 * (evaluations / self.budget) * adaptive_factor  # Modified line
                dynamic_ub = ub - (ub - lb) * 0.05 * (evaluations / self.budget) * adaptive_factor  # Modified line
                swarm[i] = np.clip(swarm[i], dynamic_lb, dynamic_ub)

                # Evaluate and update personal best
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                # Update global best
                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

            # Variable neighborhood search phase
            if evaluations < self.budget:
                local_search_radius = 0.1 * (ub - lb) * adaptive_factor
                for i in range(self.population_size):
                    candidate = global_best + np.random.uniform(-local_search_radius, local_search_radius, self.dim)
                    candidate = np.clip(candidate, lb, ub)
                    candidate_value = func(candidate)
                    evaluations += 1
                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value
                    if evaluations >= self.budget:
                        break

        return global_best, global_best_value