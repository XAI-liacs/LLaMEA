import numpy as np

class Quantum_Enhanced_PSO_Adaptive_Gaussian:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased for more robust exploration
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.8
        self.q_factor = 0.8
        self.gaussian_scale = 0.15  # Increased scale for initial exploration
        self.initial_reset_chance = 0.03  # Adjusted for better exploration
        self.momentum_factor = 1.1
        self.adaptive_rate = 0.98  # Slightly faster decrease of inertia weight
        self.temperature = 1.0
        
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
            neighborhood_size = int(self.population_size / 3)  # Smaller neighborhoods for diversity
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.w *= self.adaptive_rate
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                adaptive_gaussian_scale = self.gaussian_scale * np.exp(-evaluations / self.budget)
                annealing_factor = np.exp(-evaluations / (self.budget * self.temperature))
                
                position[i] += (velocity[i] + 
                                self.q_factor * np.random.normal(scale=adaptive_gaussian_scale, size=self.dim) +
                                annealing_factor * np.random.uniform(-1, 1, self.dim))
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

            # Stagnation-Triggered Reset Mechanism
            reset_chance = self.initial_reset_chance * ((evaluations / self.budget) ** self.momentum_factor)
            if np.random.rand() < reset_chance:
                stagnated_indices = np.random.choice(self.population_size, neighborhood_size, replace=False)
                for idx in stagnated_indices:
                    position[idx] = np.random.uniform(lb, ub, self.dim)
                    current_value = func(position[idx])
                    evaluations += 1
                    
                    if current_value < personal_best_value[idx]:
                        personal_best_position[idx] = position[idx]
                        personal_best_value[idx] = current_value
                    
                    if current_value < global_best_value:
                        global_best_position = position[idx]
                        global_best_value = current_value

            # Dynamic Neighborhood Restructuring
            if evaluations % 10 == 0 and neighborhood_size > 2:  # Prevent neighborhood size from becoming too small
                neighborhood_size = max(2, neighborhood_size - 1)

        return global_best_position, global_best_value