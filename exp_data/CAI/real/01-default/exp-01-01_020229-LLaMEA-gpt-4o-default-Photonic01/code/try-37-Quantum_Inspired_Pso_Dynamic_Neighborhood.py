import numpy as np

class Quantum_Inspired_Pso_Dynamic_Neighborhood:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.q_factor = 0.8
        self.gaussian_scale = 0.1
        self.reset_chance = 0.1
        self.adaptive_rate = 0.9
        self.neighbor_size = max(1, self.population_size // 5)
        
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
                self.w = 0.5 + 0.5 * np.cos(np.pi * evaluations / self.budget)  # Update inertia weight
                velocity[i] = (self.w * velocity[i] +
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

            # Dynamic Neighborhood Adaptation
            for _ in range(self.neighbor_size):
                random_index = np.random.randint(self.population_size)
                neighbor_indices = np.random.choice(self.population_size, self.neighbor_size, replace=False)
                local_best_index = neighbor_indices[np.argmin(personal_best_value[neighbor_indices])]
                if personal_best_value[local_best_index] < personal_best_value[random_index]:
                    personal_best_position[random_index] = personal_best_position[local_best_index]
                    personal_best_value[random_index] = personal_best_value[local_best_index]

            # Global Position Reset Mechanism
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

        return global_best_position, global_best_value