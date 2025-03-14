import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))

    def levy_flight(self, L):
        return np.random.standard_cauchy(self.dim) * L

    def chaotic_map(self, x):
        return np.mod(3.99 * x * (1 - x) + 0.03, 1.0)  # Slightly adjusted

    def opposition_based_learning(self, swarm, lb, ub):
        return lb + ub - swarm

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size
        chaotic_param = 0.8  # Adjusted initial value

        while evaluations < self.budget:
            chaotic_param = self.chaotic_map(chaotic_param)
            adaptive_factor = chaotic_param * (1 - evaluations / self.budget)
            inertia_weight = 0.5 + 0.4 * (adaptive_factor ** 2) + 0.05 * np.sin(2 * np.pi * evaluations / self.budget)
            cognitive_coeff = 1.5 * adaptive_factor + 0.05 * np.sin(2 * np.pi * evaluations / self.budget)
            social_coeff = 1.5 * (1 - adaptive_factor)
            mutation_rate = 0.05 + 0.15 * adaptive_factor  # Adjusted mutation rate

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                opposition = self.opposition_based_learning(swarm[i], lb, ub)
                swarm[i] = np.where(np.random.rand(self.dim) < 0.5, swarm[i], opposition)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                if np.random.rand() < mutation_rate:
                    L = 0.1 * adaptive_factor
                    swarm[i] += self.levy_flight(L)

                # Modified crossover mechanism
                if np.random.rand() < 0.1:  # 10% crossover rate
                    partner_idx = np.random.randint(self.population_size)
                    crossover_point = np.random.randint(1, self.dim)
                    swarm[i][:crossover_point] = personal_best[partner_idx][:crossover_point]

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