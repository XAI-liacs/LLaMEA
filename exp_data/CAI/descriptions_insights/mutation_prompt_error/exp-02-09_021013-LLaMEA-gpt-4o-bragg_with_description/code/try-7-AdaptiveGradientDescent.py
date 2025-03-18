import numpy as np

class AdaptiveGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        best_solution = None
        best_value = float('-inf')
        evaluations = 0
        
        # Define step size and adaptive parameters
        step_size = 0.1
        decay_rate = 0.95
        exploration_factor = 0.05  # Reduced exploration factor for finer search

        # Initialize random solution within bounds
        current_solution = np.random.uniform(
            func.bounds.lb, func.bounds.ub, self.dim
        )

        while evaluations < self.budget:
            # Evaluate current solution
            current_value = func(current_solution)
            evaluations += 1

            # Update best solution if current is better
            if current_value > best_value:
                best_value = current_value
                best_solution = current_solution.copy()

            # Generate stochastic gradient approximation
            gradient = np.zeros(self.dim)
            for i in range(self.dim):
                perturbation = np.zeros(self.dim)
                perturbation[i] = exploration_factor * np.random.randn()
                perturbed_solution = np.clip(
                    current_solution + perturbation,
                    func.bounds.lb, func.bounds.ub
                )
                gradient[i] = (func(perturbed_solution) - current_value) / perturbation[i]
                evaluations += 1
                if evaluations >= self.budget:
                    break

            # Adaptive step update
            current_solution += step_size * gradient
            current_solution = np.clip(current_solution, func.bounds.lb, func.bounds.ub)
            
            # Decay step size with slight boost
            step_size *= decay_rate * 1.05  # Slight boost in decay to quickly respond to gradients

            # Ensure not exceeding budget
            if evaluations >= self.budget:
                break

        return best_solution, best_value