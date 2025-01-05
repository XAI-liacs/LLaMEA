import numpy as np

class Quantum_Pso_Adaptive_Gauss_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased population size for better coverage
        self.c1 = 1.5  # Reduced cognitive component for better diversification
        self.c2 = 2.5  # Increased social component to enhance convergence
        self.w = 0.9  # Higher initial inertia for exploration
        self.w_min = 0.4  # Minimum inertia weight to avoid excessive exploration
        self.q_factor = 0.8  # Adjusted quantum factor for balanced exploration
        self.gaussian_scale = 0.2  # Increased for more significant mutations
        self.reset_chance = 0.1  # Initial reset chance to maintain diversity
        self.adaptive_rate = 0.95  # Gradual decrease in inertia weight
        self.temperature_scaling = 0.5  # New scaling factor for annealing

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
                self.w = max(self.w * self.adaptive_rate, self.w_min)  # Adaptive inertia weight
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))

                adaptive_gaussian_scale = self.gaussian_scale * (1 - evaluations / self.budget)
                annealing_factor = np.exp(-self.temperature_scaling * evaluations / self.budget)
                
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

            # Reset Mechanism to prevent premature convergence
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