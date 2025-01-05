import numpy as np

class Adaptive_Quantum_Pso_Cross_Learning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.q_factor = 0.9
        self.learning_factor = 0.5
        self.mutation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(24)

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
                r_learning = np.random.rand()

                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))

                if r_learning < self.learning_factor:
                    partner_idx = np.random.randint(self.population_size)
                    position[i] = (self.q_factor * np.random.normal(loc=partner_idx, scale=abs(ub-lb)/2, size=self.dim))

                position[i] += velocity[i]
                position[i] = np.clip(position[i], lb, ub)

                if np.random.rand() < self.mutation_rate:
                    position[i] += np.random.normal(scale=(ub-lb)/10, size=self.dim)
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

        return global_best_position, global_best_value