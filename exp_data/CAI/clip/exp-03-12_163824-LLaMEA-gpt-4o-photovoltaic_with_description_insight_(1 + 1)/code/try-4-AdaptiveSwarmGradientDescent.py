import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.layer_annealing_factor = 0.95  # Added annealing factor for velocity damping per layer

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
            inertia_weight = 0.7 + 0.3 * adaptive_factor
            cognitive_coeff = 1.5 * adaptive_factor * (1 + np.std(personal_best_value))  # Modified cognitive coefficient
            social_coeff = 1.5 * np.std(swarm) / np.mean(swarm)  # Adjust social coefficient

            # Dynamic population resizing
            if evaluations % (self.budget // 10) == 0 and evaluations < self.budget * 0.8:
                self.population_size = min(self.population_size + 1, self.budget // self.dim)
                self.velocity = np.vstack((self.velocity, np.zeros((1, self.dim))))
                new_position = np.random.uniform(lb, ub, self.dim)
                swarm = np.vstack((swarm, new_position))
                personal_best = np.vstack((personal_best, new_position))
                personal_best_value = np.append(personal_best_value, func(new_position))
                evaluations += 1

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                
                # Apply layer-wise annealing factor to velocity for refined exploration
                self.velocity[i] *= self.layer_annealing_factor
                
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
        
        return global_best, global_best_value