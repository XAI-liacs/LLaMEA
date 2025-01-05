import numpy as np

class Adaptive_Quantum_Simulated_Annealing_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_temperature = 100.0
        self.cooling_rate = 0.99
        self.min_temperature = 1e-3
        self.quantum_tunneling_factor = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        best_position = np.copy(position)
        best_value = np.array([func(p) for p in best_position])
        global_best_position = best_position[np.argmin(best_value)]
        global_best_value = np.min(best_value)

        temperature = self.initial_temperature
        evaluations = self.population_size

        while evaluations < self.budget and temperature > self.min_temperature:
            for i in range(self.population_size):
                current = position[i]
                new_position = current + np.random.normal(scale=temperature, size=self.dim)
                new_position = np.clip(new_position, lb, ub)
                
                current_value = func(current)
                new_value = func(new_position)
                evaluations += 2

                if new_value < current_value or np.exp((current_value - new_value) / temperature) > np.random.rand():
                    position[i] = new_position
                    if new_value < best_value[i]:
                        best_position[i] = new_position
                        best_value[i] = new_value
                        if new_value < global_best_value:
                            global_best_position = new_position
                            global_best_value = new_value

                if evaluations >= self.budget:
                    break

            # Quantum Tunneling Mechanism
            if np.random.rand() < self.quantum_tunneling_factor:
                random_index = np.random.randint(self.population_size)
                tunneling_position = np.random.uniform(lb, ub, self.dim)
                tunneling_value = func(tunneling_position)
                evaluations += 1
                
                if tunneling_value < best_value[random_index]:
                    position[random_index] = tunneling_position
                    best_position[random_index] = tunneling_position
                    best_value[random_index] = tunneling_value
                    if tunneling_value < global_best_value:
                        global_best_position = tunneling_position
                        global_best_value = tunneling_value

            temperature *= self.cooling_rate

        return global_best_position, global_best_value