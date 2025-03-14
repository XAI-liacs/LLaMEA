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
            inertia_weight = 0.8 - 0.4 * adaptive_factor  # Changed adaptive range for inertia weight
            cognitive_coeff = 1.5 * adaptive_factor * np.random.uniform(1.0, 2.0) 
            social_coeff = 1.7 * adaptive_factor  # Adjusted social coefficient

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
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

            # Periodic local search refinement
            if evaluations % (self.population_size * 5) == 0:  # Conduct local search periodically
                for j in range(self.population_size):
                    refined_solution = self.local_search(personal_best[j], func, lb, ub)
                    refined_value = func(refined_solution)
                    evaluations += 1
                    if refined_value < personal_best_value[j]:
                        personal_best[j] = refined_solution
                        personal_best_value[j] = refined_value
                        if refined_value < global_best_value:
                            global_best = refined_solution
                            global_best_value = refined_value

        return global_best, global_best_value

    def local_search(self, position, func, lb, ub):  # New local search function
        perturbation = np.random.uniform(-0.05, 0.05, self.dim) * (ub - lb)
        new_position = np.clip(position + perturbation, lb, ub)
        return new_position