import numpy as np

class Enhanced_Quantum_Pso_Neighborhood_Informed_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.q_factor = 0.9
        self.gaussian_scale = 0.1
        self.temperature = 1.0
        self.neighborhood_size = 5  # New: Local neighborhood size
        self.adaptive_rate = 0.95

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
                # Get neighborhood best
                distances = np.linalg.norm(position - position[i], axis=1)
                neighbors_idx = np.argsort(distances)[:self.neighborhood_size]
                neighborhood_best_value = np.min(personal_best_value[neighbors_idx])
                neighborhood_best_position = personal_best_position[neighbors_idx[np.argmin(personal_best_value[neighbors_idx])]]
                
                r1, r2 = np.random.rand(2)
                self.w *= self.adaptive_rate
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (neighborhood_best_position - position[i]))
                
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

        return global_best_position, global_best_value