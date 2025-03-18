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

        # Constraint on function evaluations
        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.9 + 0.1 * adaptive_factor  # Modified line
            cognitive_coeff = 1.5 * (0.5 + 0.5 * adaptive_factor)
            social_coeff = 1.5 * (1 - adaptive_factor)

            # Dynamically scale population size
            current_population_size = self.population_size - int(0.2 * self.population_size * evaluations / self.budget)

            for i in range(current_population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                
                # Apply velocity clamping
                self.velocity[i] = np.clip(self.velocity[i], -0.1 * (ub - lb), 0.1 * (ub - lb))

                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate and update personal best
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                # Update global best
                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break
            
            # Add diversity improvement: reintroduce random individuals if diversity is low
            if np.std(swarm) < 0.001:
                swarm[np.random.randint(current_population_size)] = np.random.uniform(lb, ub, self.dim)

        return global_best, global_best_value