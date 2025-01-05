import numpy as np

class Quantum_Levy_Pso_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.levy_alpha = 1.5  # Levy flight exponent
        self.initial_levy_scale = 0.1
        self.q_factor = 0.8
        self.adaptive_rate = 0.98
        self.temperature = 1.0
        
    def levy_flight(self, scale, size):
        u = np.random.normal(0, 1, size) * scale
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1/self.levy_alpha))
        return step

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
                self.w *= self.adaptive_rate
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                adaptive_levy_scale = self.initial_levy_scale * (1 - evaluations / self.budget)
                annealing_factor = np.exp(-evaluations / (self.budget * self.temperature))
                
                position[i] += (velocity[i] + 
                                self.q_factor * self.levy_flight(adaptive_levy_scale, self.dim) +
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

        return global_best_position, global_best_value