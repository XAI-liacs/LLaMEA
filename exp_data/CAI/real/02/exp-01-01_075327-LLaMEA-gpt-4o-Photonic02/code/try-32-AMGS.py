import numpy as np

class AMGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')

    def __call__(self, func):
        # Initialize the search
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        current_solution = np.random.uniform(bounds[0], bounds[1], self.dim)
        current_value = func(current_solution)
        self.best_solution = current_solution
        self.best_value = current_value

        # Define parameters
        memory_size = max(5, self.dim)
        step_size = 0.1 * (bounds[1] - bounds[0])
        memory = []

        # Main optimization loop
        for _ in range(self.budget - 1):
            # Adaptive memory check
            if len(memory) >= memory_size:
                memory.pop(0)

            # Estimate gradients using memory if possible
            perturbation = np.random.randn(self.dim) * step_size
            trial_solution = current_solution + perturbation
            trial_solution = np.clip(trial_solution, bounds[0], bounds[1])
            trial_value = func(trial_solution)
            memory.append((trial_solution, trial_value))

            # Compute pseudo-gradient
            gradients = np.zeros(self.dim)
            if len(memory) > 1:  # Ensure enough memory for gradient estimation
                for sol, val in memory:
                    diff = sol - current_solution
                    gradients += (val - current_value) * diff / (np.linalg.norm(diff) + 1e-8)

            # Update current solution
            gradients /= (np.linalg.norm(gradients) + 1e-6)  # Normalize with L2 norm (changed from 1e-8 to 1e-6)
            step_size *= 0.985  # Adapt learning rate (changed from 0.98 to 0.985)
            new_solution = current_solution - step_size * gradients
            new_solution = np.clip(new_solution, bounds[0], bounds[1])
            new_value = func(new_solution)

            # Update best found solution
            if new_value < self.best_value:
                self.best_value = new_value
                self.best_solution = new_solution

            # Prepare for next iteration
            current_solution = new_solution
            current_value = new_value

        return self.best_solution, self.best_value