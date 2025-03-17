import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.neigh_size = max(2, self.population_size // 5)  # Define a neighborhood size

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
            inertia_weight = 0.9 - 0.5 * adaptive_factor  # Adjust inertia weight formula
            cognitive_coeff = 1.7 * adaptive_factor
            social_coeff = 1.3 + 0.7 * adaptive_factor  

            for i in range(self.population_size):
                # Dynamic neighborhood topology
                neighbors_idx = np.random.choice(self.population_size, self.neigh_size, replace=False)
                neighborhood_best = personal_best[neighbors_idx[np.argmin(personal_best_value[neighbors_idx])]]
                
                r1, r2, r3 = np.random.random(self.dim), np.random.random(self.dim), np.random.random(self.dim)
                mutation_coeff = 0.1 * adaptive_factor  # Introduce adaptive mutation
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (neighborhood_best - swarm[i]) +
                                    mutation_coeff * r3 * (global_best - swarm[i]))
                self.velocity[i] = np.clip(self.velocity[i], -0.1 * (ub - lb), 0.1 * (ub - lb))  
                swarm[i] += 0.5 * self.velocity[i]  # Apply adaptive learning rate
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