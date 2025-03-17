import numpy as np

class AdaptiveSwarmChaosMemory:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.memory_coeff = np.random.uniform(0.1, 0.3, self.population_size)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        chaotic_factor = np.random.random()  # Initialize with a random chaotic factor
        multi_chaotic = np.random.random()   # New multi-chaotic map factor

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.5 + 0.4 * np.cos(2 * np.pi * evaluations / self.budget)

            cognitive_coeff = 1.5 * adaptive_factor * chaotic_factor
            social_coeff = 1.5 * (1 + 0.1 * adaptive_factor) * np.sin(np.pi * evaluations / self.budget)

            for i in range(self.population_size):
                r1, r2, r3 = np.random.random(self.dim), np.random.random(self.dim), np.random.random(self.dim)
                damping_factor = 0.9 + 0.1 * (1 - adaptive_factor)
                
                # Update velocity with an adaptive learning rate
                self.velocity[i] = (damping_factor * inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    self.memory_coeff[i] * r3 * (swarm[np.random.randint(self.population_size)] - swarm[i]))
                
                exploration_noise = 0.02 * np.random.uniform(lb, ub, self.dim) * (adaptive_factor**3 + 0.01 * np.sin(2 * np.pi * evaluations / self.budget))
                swarm[i] += self.velocity[i] + exploration_noise
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

            chaotic_factor = 4 * chaotic_factor * (1 - chaotic_factor)
            multi_chaotic = 4 * multi_chaotic * (1 - multi_chaotic) * np.sin(np.pi * evaluations / self.budget)  # Apply multi-chaotic map

        return global_best, global_best_value