import numpy as np

class Quantum_Adaptive_Mutation_Gradient_Pso_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased for more diversity
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.6
        self.q_factor = 0.95
        self.gaussian_scale = 0.1
        self.mutation_rate = 0.2  # Introduced for adaptive mutation
        self.mutation_decay = 0.995  # Decay rate of mutation impact
        self.temperature = 1.0
        self.gradient_step = 0.01  # Step size for gradient-based exploration

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
                self.w *= self.mutation_decay
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                adaptive_gaussian_scale = self.gaussian_scale * (1 - evaluations / self.budget)
                annealing_factor = np.exp(-evaluations / (self.budget * self.temperature))
                
                position[i] += (velocity[i] +
                                self.q_factor * np.random.normal(scale=adaptive_gaussian_scale, size=self.dim) +
                                annealing_factor * np.random.uniform(-1, 1, self.dim))

                # Apply adaptive mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(scale=adaptive_gaussian_scale, size=self.dim)
                    position[i] += mutation_vector
                    self.mutation_rate *= self.mutation_decay

                # Gradient-based exploration (simple numerical gradient approximation)
                gradient = np.zeros(self.dim)
                eps = 1e-6
                for d in range(self.dim):
                    temp_pos = np.copy(position[i])
                    temp_pos[d] += eps
                    gradient[d] = (func(temp_pos) - func(position[i])) / eps

                position[i] -= self.gradient_step * gradient
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