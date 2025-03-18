import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.initial_population_size, dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.initial_population_size
        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.4 + 0.5 * np.sin(2 * np.pi * adaptive_factor**2)  # Change 1
            cognitive_coeff = 1.2 * adaptive_factor  # Change 2
            social_coeff = 1.7  # Change 3
            learning_rate = 0.1 + 0.9 * adaptive_factor
            
            current_population_size = int(self.initial_population_size * (1 + 0.4 * adaptive_factor))  # Change 4
            self.velocity = np.resize(self.velocity, (current_population_size, self.dim))
            swarm = np.resize(swarm, (current_population_size, self.dim))
            
            neighborhood_size = int(0.1 * current_population_size)  # Change 5
            for i in range(current_population_size):
                neighbors = np.random.choice(current_population_size, neighborhood_size, replace=False)
                local_best = swarm[neighbors[np.argmin([func(swarm[n]) for n in neighbors])]]  # Change 6

                r1, r2, r3 = np.random.random(self.dim), np.random.random(self.dim), np.random.random()  # Change 7
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i % self.initial_population_size] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    r3 * (local_best - swarm[i]))  # Change 8
                swarm[i] += learning_rate * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i % self.initial_population_size]:
                    personal_best[i % self.initial_population_size] = swarm[i]
                    personal_best_value[i % self.initial_population_size] = f_value

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if np.random.rand() < 0.2:  # Change 9
                    if np.random.rand() < 0.5:
                        mutation = np.random.normal(0, 0.05 * adaptive_factor, self.dim)  # Change 10
                        swarm[i] = np.clip(swarm[i] + mutation, lb, ub)
                    else:
                        mutation = np.random.uniform(-0.05, 0.05, self.dim)  # Change 11
                        swarm[i] = np.clip(swarm[i] + mutation, lb, ub)

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value