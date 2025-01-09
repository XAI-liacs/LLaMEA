import numpy as np

class QuantumAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.temperature = 1.0
        self.tunneling_prob = 0.1

    def _initialize_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
    
    def _neighbor_solution(self, current_solution):
        return current_solution + np.random.uniform(-0.1, 0.1, self.dim)
    
    def _quantum_tunneling(self, current_solution):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def __call__(self, func):
        current_solution = self._initialize_solution()
        best_solution = current_solution
        best_value = func(current_solution)
        
        evaluations = 1

        while evaluations < self.budget:
            if np.random.rand() < self.tunneling_prob:
                candidate_solution = self._quantum_tunneling(current_solution)
            else:
                candidate_solution = self._neighbor_solution(current_solution)

            candidate_value = func(candidate_solution)
            evaluations += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            acceptance_prob = np.exp((best_value - candidate_value) / self.temperature)
            if candidate_value < best_value or np.random.rand() < acceptance_prob:
                current_solution = candidate_solution

            # Gradually cool down the temperature
            self.temperature *= 0.99

        return best_solution