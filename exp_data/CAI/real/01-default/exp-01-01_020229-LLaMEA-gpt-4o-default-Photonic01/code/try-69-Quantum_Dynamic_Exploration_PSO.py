import numpy as np

class Quantum_Dynamic_Exploration_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Slightly increased for more samples
        self.c1 = 1.5             # Adjusted for balanced exploration-exploitation
        self.c2 = 1.5
        self.w = 0.6              # Lower starting inertia weight
        self.q_factor = 0.85
        self.gaussian_scale = 0.15
        self.initial_reset_chance = 0.02
        self.momentum_factor = 1.2
        self.adaptive_rate = 0.995
        self.temperature = 0.9    # Reduced for faster cooling
        self.velocity_clamp = 0.5 # New: clamp velocity to control exploration
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            neighborhood_size = int(self.population_size / 3)  # Tighter neighborhood
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.w *= self.adaptive_rate
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                velocity[i] = np.clip(velocity[i], -self.velocity_clamp, self.velocity_clamp)

                adaptive_gaussian_scale = self.gaussian_scale * (1 - evaluations / self.budget)
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

            # Adaptive Reset Mechanism based on stagnation
            reset_chance = self.initial_reset_chance * ((evaluations / self.budget) ** self.momentum_factor)
            if np.random.rand() < reset_chance:
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

            # Dynamic Neighborhood Restructuring
            if evaluations % 15 == 0:  # Adjust neighborhood size dynamically
                neighborhood_size = max(1, neighborhood_size - 1)

        return global_best_position, global_best_value