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
            inertia_weight = 0.5 + 0.5 * np.sin(np.pi * adaptive_factor)  # Change 1
            cognitive_coeff = 1.5 * adaptive_factor + 0.5  # Change 2
            social_coeff = 1.7  # Change 3
            learning_rate = 0.1 + 0.6 * adaptive_factor  # Change 4
            
            current_population_size = int(self.initial_population_size * (1 + 0.5 * adaptive_factor))
            self.velocity = np.resize(self.velocity, (current_population_size, self.dim))
            swarm = np.resize(swarm, (current_population_size, self.dim))
            
            for i in range(current_population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i % self.initial_population_size] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
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

                if np.random.rand() < 0.2:  # Change 5
                    mutation = np.random.uniform(-0.05, 0.05, self.dim)  # Change 6
                    swarm[i] = np.clip(swarm[i] + mutation, lb, ub)

                if evaluations >= self.budget:
                    break

                # Elite selection and local search (Change 7-8)
                if i == 0 and evaluations < self.budget:
                    elite = personal_best[np.argmin(personal_best_value)]
                    local_search = elite + np.random.uniform(-0.01, 0.01, self.dim)
                    local_search = np.clip(local_search, lb, ub)
                    local_value = func(local_search)
                    evaluations += 1
                    if local_value < global_best_value:
                        global_best = local_search
                        global_best_value = local_value

        return global_best, global_best_value