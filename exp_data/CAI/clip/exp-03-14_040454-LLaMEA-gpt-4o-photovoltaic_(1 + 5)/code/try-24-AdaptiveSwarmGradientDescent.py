import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, dim))  # Stochastic initial velocity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        # Memory for local optima escape
        memory = np.full((self.population_size, self.dim), np.inf)
        escape_threshold = 50

        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = np.random.rand()  # Stochastic adaptive factor
            inertia_weight = 0.7 - 0.3 * adaptive_factor  # Dynamic inertia reversal
            cognitive_coeff = 1.5 * (0.5 + 0.5 * adaptive_factor)
            social_coeff = 1.5
            
            adaptive_lr = 0.1 + 0.9 * adaptive_factor

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += adaptive_lr * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                # Selective global best update
                if f_value < global_best_value and np.random.rand() < 0.5:
                    global_best = swarm[i]
                    global_best_value = f_value

                # Memory-augmented local optima escape
                if evaluations % escape_threshold == 0:
                    if np.all(memory[i] == swarm[i]):
                        swarm[i] = np.random.uniform(lb, ub, self.dim)  # Perturb to escape
                    memory[i] = swarm[i]

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value