import numpy as np

class AdaptiveSwarmGradientDescentEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        # Memory for previous best solutions
        self.previous_global_best = None  # Change 1

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
            inertia_weight = 0.4 + 0.3 * np.sin(evaluations / self.budget * np.pi)
            cognitive_coeff = 1.7 * adaptive_factor
            social_coeff = 1.7 + 0.3 * (1 - adaptive_factor)

            cooling_schedule = np.exp(-(evaluations / self.budget)**2)
            
            # Introduce adaptive mutation rate
            mutation_rate = 0.1 * (1.0 - global_best_value)  # Change 2

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                max_velocity = (ub - lb) * 0.1 * cooling_schedule
                self.velocity[i] = np.clip(self.velocity[i], -max_velocity, max_velocity)
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)
                
                # Apply mutation with adaptive rate
                if np.random.rand() < mutation_rate:  # Change 3
                    mutation_vector = np.random.uniform(-0.1, 0.1, self.dim)  # Change 4
                    swarm[i] += mutation_vector  # Change 5
                    swarm[i] = np.clip(swarm[i], lb, ub)  # Change 6

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    self.previous_global_best = global_best  # Change 7
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value