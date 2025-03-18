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

        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.5 + 0.5 * adaptive_factor  # Changed line
            cognitive_coeff = 2.0 * adaptive_factor  # Changed line
            social_coeff = 1.0 + adaptive_factor  # Changed line

            # Dynamic population size adjustment
            current_population_size = self.population_size + int(adaptive_factor * 5)

            elite_threshold = int(0.2 * current_population_size)  # New line
            elite_indices = np.argpartition(personal_best_value, elite_threshold)[:elite_threshold]  # New line
            elite_individuals = personal_best[elite_indices]  # New line

            for i in range(current_population_size):
                if i >= self.population_size:
                    new_member = np.random.uniform(lb, ub, self.dim)
                    self.velocity = np.random.uniform(-1, 1, (current_population_size, self.dim))
                else:
                    r1, r2, r3 = np.random.random(self.dim), np.random.random(self.dim), np.random.random(self.dim)  # Changed line
                    elite_influence = elite_individuals[np.random.randint(elite_individuals.shape[0])]  # New line
                    self.velocity[i] = (inertia_weight * self.velocity[i] +
                                        cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                        social_coeff * r2 * (global_best - swarm[i]) +  # Changed line
                                        0.5 * r3 * (elite_influence - swarm[i]))  # New line
                    swarm[i] += self.velocity[i]
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