import numpy as np

class Quantum_Pso_Adaptive_Gauss_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.q_factor = 0.9
        self.gaussian_scale = 0.1
        self.reset_chance = 0.1

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
        iteration = 0
        max_iterations = self.budget // self.population_size

        while evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (iteration / max_iterations))
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (w * velocity[i] + 
                               self.c1 * r1 * (personal_best_position[i] - position[i]) + 
                               self.c2 * r2 * (global_best_position - position[i]))
                position[i] += velocity[i] + self.q_factor * np.random.normal(scale=self.gaussian_scale * (1 - iteration / max_iterations), size=self.dim)
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

            if np.random.rand() < self.reset_chance:
                random_index = np.random.randint(self.population_size)
                position[random_index] = np.random.uniform(lb, ub, self.dim)
                current_value = func(position[random_index])
                evaluations += 1
                
                if current_value < personal_best_value[random_index]:
                    personal_best_position[random_index] = position[random_index]
                    personal_best_value[random_index] = current_value
                
                if current_value < global_best_value:
                    global_best_position = position[random_index]
                    global_best_value = current_value

            iteration += 1

        return global_best_position, global_best_value