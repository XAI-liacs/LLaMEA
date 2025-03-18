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

        # Logistic map parameters for chaotic sequence
        chaotic_sequence = np.random.rand(self.population_size)
        logistic_mu = 3.9

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.5 + 0.3 * adaptive_factor  # Modified inertia weight
            cognitive_coeff = 1.5 * adaptive_factor
            social_coeff = 1.6 * np.std(swarm) / np.mean(swarm)  # Adjusted social coefficient

            for i in range(self.population_size):
                # Update chaotic sequence
                chaotic_sequence[i] = logistic_mu * chaotic_sequence[i] * (1 - chaotic_sequence[i])

                r1, r2 = chaotic_sequence[i], np.random.random(self.dim)  # Use chaotic r1
                mutation_factor = 0.17 * chaotic_sequence[i] * adaptive_factor  # Use enhanced chaotic mutation factor 
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    mutation_factor) 
                
                if evaluations % 8 == 0:  # Add periodic chaotic perturbation
                    self.velocity[i] += 0.1 * chaotic_sequence[i] * adaptive_factor  # Increased perturbation factor
                
                self.velocity[i] *= self.layer_annealing_factor
                
                swarm[i] += self.velocity[i]
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