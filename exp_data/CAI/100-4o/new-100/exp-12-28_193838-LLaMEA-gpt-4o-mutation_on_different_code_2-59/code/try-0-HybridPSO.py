import numpy as np

class HybridPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.mutation_factor = 0.8
        self.f_opt = np.Inf
        self.x_opt = None
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize swarm
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([np.Inf] * self.population_size)
        global_best_position = None
        global_best_value = np.Inf
        evaluations = 0
        
        while evaluations < self.budget:
            # Evaluate the swarm
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                value = func(positions[i])
                evaluations += 1
                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = positions[i]
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = positions[i]
            
            # Update velocities and positions
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (
                self.inertia_weight * velocities +
                self.cognitive_coef * r1 * (personal_best_positions - positions) +
                self.social_coef * r2 * (global_best_position - positions)
            )
            positions += velocities
            
            # Apply boundary constraints
            positions = np.clip(positions, lb, ub)
            
            # Differential mutation
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                mutant_vector = positions[indices[0]] + self.mutation_factor * (positions[indices[1]] - positions[indices[2]])
                mutant_vector = np.clip(mutant_vector, lb, ub)
                if evaluations < self.budget:
                    mutant_value = func(mutant_vector)
                    evaluations += 1
                    if mutant_value < personal_best_values[i]:
                        personal_best_values[i] = mutant_value
                        personal_best_positions[i] = mutant_vector
                        if mutant_value < global_best_value:
                            global_best_value = mutant_value
                            global_best_position = mutant_vector
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt