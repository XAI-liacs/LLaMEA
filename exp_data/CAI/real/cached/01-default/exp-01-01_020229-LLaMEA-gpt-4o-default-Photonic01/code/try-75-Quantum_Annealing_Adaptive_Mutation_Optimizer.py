import numpy as np

class Quantum_Annealing_Adaptive_Mutation_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Larger for better initial exploration
        self.initial_temperature = 10.0
        self.cooling_rate = 0.95
        self.mutation_scale = 0.1
        self.q_factor = 0.8
        self.mutation_adaptive_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(24)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size
        temperature = self.initial_temperature

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum-inspired position update
                quantum_jump = self.q_factor * np.random.uniform(-1, 1, self.dim)
                new_position = position[i] + quantum_jump
                new_position = np.clip(new_position, lb, ub)

                current_value = func(new_position)
                evaluations += 1

                # Simulated Annealing acceptance criteria
                if current_value < personal_best_value[i] or \
                   np.random.rand() < np.exp((personal_best_value[i] - current_value) / temperature):
                    position[i] = new_position
                    personal_best_position[i] = new_position
                    personal_best_value[i] = current_value
                
                if current_value < global_best_value:
                    global_best_position = new_position
                    global_best_value = current_value

                # Adaptive mutation
                if np.random.rand() < self.mutation_adaptive_rate:
                    mutation = self.mutation_scale * np.random.normal(size=self.dim)
                    mutated_position = position[i] + mutation
                    mutated_position = np.clip(mutated_position, lb, ub)
                    
                    mutated_value = func(mutated_position)
                    evaluations += 1
                    
                    if mutated_value < personal_best_value[i]:
                        position[i] = mutated_position
                        personal_best_position[i] = mutated_position
                        personal_best_value[i] = mutated_value
                    
                    if mutated_value < global_best_value:
                        global_best_position = mutated_position
                        global_best_value = mutated_value

                if evaluations >= self.budget:
                    break

            # Cooling schedule
            temperature *= self.cooling_rate

        return global_best_position, global_best_value