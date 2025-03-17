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
            inertia_weight = 0.6 - 0.4 * adaptive_factor  # Adjusted dynamic inertia
            cognitive_coeff = 1.2 + 0.3 * adaptive_factor  # Slightly adjusted cognitive coefficient
            social_coeff = 1.8
            
            # Introduce neighborhood influence
            neighborhood_coeff = 0.5 + 0.3 * adaptive_factor

            adaptive_lr = 0.1 + 0.9 * adaptive_factor

            for i in range(self.population_size):
                r1, r2, r3 = np.random.random(self.dim), np.random.random(self.dim), np.random.random(self.dim)
                best_neighbor = personal_best[np.random.choice(self.population_size)]
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    neighborhood_coeff * r3 * (best_neighbor - swarm[i]))
                swarm[i] += adaptive_lr * self.velocity[i]
                
                # Adaptive mutation
                if np.random.rand() < 0.1:
                    mutation_strength = (ub - lb) * 0.02 * adaptive_factor
                    swarm[i] += np.random.normal(0, mutation_strength, self.dim)
                    
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                if f_value < global_best_value and np.random.rand() < 0.5:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value