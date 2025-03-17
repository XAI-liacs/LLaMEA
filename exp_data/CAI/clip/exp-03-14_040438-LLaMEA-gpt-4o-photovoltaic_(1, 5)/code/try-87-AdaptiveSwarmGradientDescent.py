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
            chaos_seq = 0.8 * (1 - 2 * np.abs(0.5 - np.random.rand()))  # Refined chaos sequence
            inertia_weight = 0.7 + (0.4 * adaptive_factor ** 2)  # Non-linear inertia scaling

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
                                        social_coeff * r2 * (global_best - swarm[i])) + 0.02 * np.random.randn(self.dim)
                    swarm[i] += self.velocity[i]
                    swarm[i] = np.clip(swarm[i], lb, ub)

                    # Quantum-inspired influence
                    q_rand = 2 * np.random.rand(self.dim) - 1
                    swarm[i] += (self.q_coefficient + 0.25 * adaptive_factor) * adaptive_factor * q_rand * (global_best - swarm[i]) * 0.8  # Refined quantum influence

                    # Stochastic Opposition-based Learning
                    opposite = lb + ub - swarm[i] + np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                    opposite = np.clip(opposite, lb, ub)
                    f_value = func(opposite)
                    if f_value < personal_best_value[i]:
                        swarm[i] = opposite
                        personal_best[i] = opposite
                        personal_best_value[i] = f_value

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