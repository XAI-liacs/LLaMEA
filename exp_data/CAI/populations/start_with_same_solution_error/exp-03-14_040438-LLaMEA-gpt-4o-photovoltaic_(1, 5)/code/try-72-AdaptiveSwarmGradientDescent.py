import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.q_coefficient = 1.5  # Quantum-inspired coefficient

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
            chaos_seq = 0.7 * (1 - 2 * np.abs(0.5 - np.random.rand()))
            inertia_weight = 0.9 + (chaos_seq * adaptive_factor * 1.05)  # Adjusted from 0.7 to 0.9

            # Dynamically scale learning factors for enhanced exploration
            cognitive_coeff = 1.5 * (0.5 + adaptive_factor)
            social_coeff = 1.5 * (0.5 + adaptive_factor)

            current_population_size = min(self.population_size + int(adaptive_factor * 5), self.budget - evaluations)

            for i in range(current_population_size):
                if i >= self.population_size:
                    new_member = lb + (ub - lb) * np.random.rand(self.dim) * (1 + np.abs(chaos_seq))
                    self.velocity = np.random.uniform(-1.5, 1.5, (self.population_size, self.dim))
                    swarm[i % self.population_size] = new_member
                else:
                    r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                    self.velocity[i] = (inertia_weight * self.velocity[i] +
                                        cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                        social_coeff * r2 * (global_best - swarm[i])) + 0.03 * np.random.randn(self.dim)  # Updated noise factor
                    swarm[i] += self.velocity[i]
                    swarm[i] = np.clip(swarm[i], lb, ub)

                    # Quantum-inspired influence
                    q_rand = 2 * np.random.rand(self.dim) - 1
                    swarm[i] += self.q_coefficient * adaptive_factor * q_rand * (global_best - swarm[i]) * 0.7  # Adjusted scaling factor

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