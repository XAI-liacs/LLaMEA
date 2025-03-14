import numpy as np

class AdaptiveSwarmGradientDescentEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.mutation_factor = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = 1 - (evaluations / self.budget)**2 # Change 1
            inertia_weight = 0.9 * np.exp(-0.5 * evaluations / self.budget) # Change 2
            cognitive_coeff = 1.5 + 0.1 * np.cos(evaluations / self.budget * np.pi) # Change 3
            social_coeff = 1.5 + 0.5 * np.sin(evaluations / self.budget * np.pi) # Change 4

            cooling_schedule = np.exp(-(evaluations / self.budget)**2)
            
            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                max_velocity = (ub - lb) * 0.1 * cooling_schedule
                self.velocity[i] = np.clip(self.velocity[i], -max_velocity, max_velocity)
                swarm[i] += self.velocity[i]
                
                # Introduce differential mutation
                idxs = np.random.choice(self.population_size, 3, replace=False) # Change 5
                mutant = personal_best[idxs[0]] + self.mutation_factor * (personal_best[idxs[1]] - personal_best[idxs[2]]) # Change 6
                mutant = np.clip(mutant, lb, ub) # Change 7
                trial = np.clip(swarm[i] + mutant - personal_best[i], lb, ub) # Change 8
                
                swarm[i] = trial if func(trial) < func(swarm[i]) else swarm[i] # Change 9

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