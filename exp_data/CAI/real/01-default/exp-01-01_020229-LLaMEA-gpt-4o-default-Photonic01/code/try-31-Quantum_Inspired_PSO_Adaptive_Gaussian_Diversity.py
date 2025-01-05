import numpy as np

class Quantum_Inspired_PSO_Adaptive_Gaussian_Diversity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_initial = 0.9
        self.w_final = 0.4
        self.q_factor = 0.8
        self.gaussian_scale = 0.1
        self.momentum_factor = 1.0
        self.diversity_preservation = 0.15

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
                w = self.w_final + (self.w_initial - self.w_final) * (1 - evaluations / self.budget)
                velocity[i] = (w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                adaptive_gaussian_scale = self.gaussian_scale * (1 - evaluations / self.budget)
                position[i] += (velocity[i] + 
                                self.q_factor * np.random.normal(scale=adaptive_gaussian_scale, size=self.dim))
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

            # Diversity Preservation Mechanism
            diversity_index = np.random.choice(self.population_size, int(self.population_size * self.diversity_preservation), replace=False)
            for idx in diversity_index:
                if np.random.rand() < 0.5:
                    position[idx] = np.random.uniform(lb, ub, self.dim)
                else:
                    position[idx] += self.q_factor * np.random.normal(scale=self.gaussian_scale, size=self.dim)
                position[idx] = np.clip(position[idx], lb, ub)
                current_value = func(position[idx])
                evaluations += 1
                
                if current_value < personal_best_value[idx]:
                    personal_best_position[idx] = position[idx]
                    personal_best_value[idx] = current_value
                
                if current_value < global_best_value:
                    global_best_position = position[idx]
                    global_best_value = current_value

        return global_best_position, global_best_value