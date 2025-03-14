import numpy as np

class AdaptiveSwarmGradientDescentEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.base_population_size, dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.base_population_size
        swarm = np.random.uniform(lb, ub, (population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = population_size

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.5 + 0.2 * np.sin(evaluations / self.budget * np.pi) # Change 1
            cognitive_coeff = 1.5 * adaptive_factor # Change 2
            social_coeff = 1.8 - 0.2 * adaptive_factor # Change 3
            perturbation_strength = 0.05 * (1 - adaptive_factor) # Change 4
            
            cooling_schedule = np.exp(-(evaluations / self.budget)**2) 
            for i in range(population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                max_velocity = (ub - lb) * 0.1 * cooling_schedule
                self.velocity[i] = np.clip(self.velocity[i], -max_velocity, max_velocity)
                swarm[i] += self.velocity[i] + perturbation_strength * np.random.uniform(-1, 1, self.dim) # Change 5
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

            if evaluations % 10 == 0: # Change 6
                population_size = self.base_population_size + int(adaptive_factor * self.base_population_size) # Change 7
                self.velocity.resize((population_size, self.dim)) # Change 8
                swarm.resize((population_size, self.dim)) # Change 9
                personal_best.resize((population_size, self.dim)) # Change 10
                personal_best_value.resize(population_size) # Change 11

        return global_best, global_best_value