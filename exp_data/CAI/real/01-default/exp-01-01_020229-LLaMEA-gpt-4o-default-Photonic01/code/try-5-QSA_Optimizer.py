import numpy as np

class QSA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_temp = 1000
        self.final_temp = 1
        self.alpha = 0.95  # Cooling rate
        self.quantum_amplitude = 0.1  # Quantum fluctuation control

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize position randomly
        current_position = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_position)
        best_position = np.copy(current_position)
        best_value = current_value

        temperature = self.initial_temp
        evaluations = 1

        while evaluations < self.budget and temperature > self.final_temp:
            # Quantum-inspired transition with superposition
            quantum_step = np.random.uniform(-1, 1, self.dim) * self.quantum_amplitude * temperature
            candidate_position = current_position + quantum_step
            candidate_position = np.clip(candidate_position, lb, ub)

            # Evaluate candidate solution
            candidate_value = func(candidate_position)
            evaluations += 1

            # Acceptance criteria based on simulated annealing
            delta_value = candidate_value - current_value
            acceptance_probability = np.exp(-delta_value / temperature) if delta_value > 0 else 1

            if np.random.rand() < acceptance_probability:
                current_position = candidate_position
                current_value = candidate_value

                # Update best found solution
                if current_value < best_value:
                    best_position = current_position
                    best_value = current_value

            # Cooling schedule
            temperature *= self.alpha

        return best_position, best_value