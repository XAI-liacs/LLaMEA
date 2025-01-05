import numpy as np

class Adaptive_Hybrid_Swarm_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.6
        self.q_factor = 0.9
        self.initial_reset_chance = 0.04
        self.momentum_factor = 1.05
        self.adaptive_rate = 0.98
        self.leap_scale = 0.1
        
    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        return u / np.abs(v) ** (1 / beta)
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.w *= self.adaptive_rate
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))

                position[i] += velocity[i] + self.q_factor * self.levy_flight(self.dim)
                position[i] = np.clip(position[i], lb, ub)

                current_value = func(position[i])
                evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_position[i] = position[i]
                    personal_best_value[i] = current_value

                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

                if evaluations >= self.budget:
                    break

            # Greedy Local Selection
            local_indices = np.random.choice(self.population_size, 2, replace=False)
            if personal_best_value[local_indices[0]] < personal_best_value[local_indices[1]]:
                best_local_position = personal_best_position[local_indices[0]]
            else:
                best_local_position = personal_best_position[local_indices[1]]

            if np.random.rand() < self.initial_reset_chance * ((evaluations / self.budget) ** self.momentum_factor):
                random_index = np.random.randint(self.population_size)
                position[random_index] = best_local_position + np.random.uniform(lb, ub, self.dim) * self.leap_scale
                position[random_index] = np.clip(position[random_index], lb, ub)
                current_value = func(position[random_index])
                evaluations += 1

                if current_value < personal_best_value[random_index]:
                    personal_best_position[random_index] = position[random_index]
                    personal_best_value[random_index] = current_value

                if current_value < global_best_value:
                    global_best_position = position[random_index]
                    global_best_value = current_value

        return global_best_position, global_best_value