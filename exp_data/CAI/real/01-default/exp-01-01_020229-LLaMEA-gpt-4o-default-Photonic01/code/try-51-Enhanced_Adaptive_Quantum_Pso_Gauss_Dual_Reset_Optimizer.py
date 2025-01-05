import numpy as np

class Enhanced_Adaptive_Quantum_Pso_Gauss_Dual_Reset_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7  # Starting inertia weight
        self.q_factor = 0.9
        self.gaussian_scale = 0.1
        self.initial_reset_chance = 0.1
        self.momentum_factor = 1.05
        self.adaptive_rate = 0.95
        self.stagnation_threshold = 10  # New parameter for additional reset mechanism

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
        no_improvement_counter = 0  # Counter for tracking stagnation
        
        while evaluations < self.budget:
            improvement_made = False
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.w *= self.adaptive_rate  # Dynamically adjust inertia weight
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
                    improvement_made = True
                
                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value
                    improvement_made = True

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
                    improvement_made = True
                
                if current_value < global_best_value:
                    global_best_position = position[random_index]
                    global_best_value = current_value
                    improvement_made = True

            # Additional Reset Mechanism for stagnation
            if not improvement_made:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

            if no_improvement_counter >= self.stagnation_threshold:
                stagnation_index = np.random.randint(self.population_size)
                position[stagnation_index] = np.random.uniform(lb, ub, self.dim)
                current_value = func(position[stagnation_index])
                evaluations += 1
                
                if current_value < personal_best_value[stagnation_index]:
                    personal_best_position[stagnation_index] = position[stagnation_index]
                    personal_best_value[stagnation_index] = current_value
                    no_improvement_counter = 0  # Reset counter on improvement
                
                if current_value < global_best_value:
                    global_best_position = position[stagnation_index]
                    global_best_value = current_value

        return global_best_position, global_best_value