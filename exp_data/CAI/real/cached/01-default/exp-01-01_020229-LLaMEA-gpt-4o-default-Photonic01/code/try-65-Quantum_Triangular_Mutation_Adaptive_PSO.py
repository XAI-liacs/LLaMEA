import numpy as np

class Quantum_Triangular_Mutation_Adaptive_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.9
        self.q_factor = 0.9
        self.triangular_scale = 0.1  # Scale for triangular distribution
        self.inertia_adaptive_rate = 0.98  # More dynamic inertia weight adaptation
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
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.w *= self.inertia_adaptive_rate
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                triangular_noise = np.random.triangular(-self.triangular_scale, 0, self.triangular_scale, self.dim)
                annealing_factor = np.exp(-evaluations / (self.budget * self.temperature))
                
                position[i] += (velocity[i] + 
                                self.q_factor * triangular_noise +
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
            if evaluations % (self.population_size // 2) == 0:
                random_indices = np.random.choice(self.population_size, 3, replace=False)
                for idx in random_indices:
                    position[idx] = np.random.uniform(lb, ub, self.dim)
                    current_value = func(position[idx])
                    evaluations += 1
                    
                    if current_value < personal_best_value[idx]:
                        personal_best_position[idx] = position[idx]
                        personal_best_value[idx] = current_value
                    
                    if current_value < global_best_value:
                        global_best_position = position[idx]
                        global_best_value = current_value

        return global_best_position, global_best_value