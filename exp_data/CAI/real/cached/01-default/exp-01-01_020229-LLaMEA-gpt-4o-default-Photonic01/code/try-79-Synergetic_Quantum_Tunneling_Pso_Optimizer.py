import numpy as np

class Synergetic_Quantum_Tunneling_Pso_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased population for better exploration
        self.c1 = 1.5  # Reduced cognitive factor to balance exploration and exploitation
        self.c2 = 2.5  # Increased social factor to emphasize global best attraction
        self.w = 0.9  # Start with higher inertia for exploration
        self.q_factor = 0.8  # Reduced for focused exploration
        self.gaussian_scale = 0.05  # Reduced for fine-tuned mutations
        self.initial_reset_chance = 0.05
        self.momentum_factor = 1.02  # Slight adjustment to improve reset chance increment
        self.adaptive_rate = 0.995  # Slower inertia reduction for sustained exploration
        self.initial_temperature = 5.0  # Higher initial temperature for aggressive exploration
        self.temperature_decay = 0.95  # Decay temperature slower for smoother transitions
        
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
        temperature = self.initial_temperature

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))

                # Quantum tunneling and temperature influence
                adaptive_gaussian_scale = self.gaussian_scale * (1 - evaluations / self.budget)
                position[i] += (velocity[i] + 
                                self.q_factor * np.random.normal(scale=adaptive_gaussian_scale, size=self.dim) +
                                temperature * np.random.uniform(-1, 1, self.dim))
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

            # Update inertia and temperature
            self.w *= self.adaptive_rate
            temperature *= self.temperature_decay

        return global_best_position, global_best_value