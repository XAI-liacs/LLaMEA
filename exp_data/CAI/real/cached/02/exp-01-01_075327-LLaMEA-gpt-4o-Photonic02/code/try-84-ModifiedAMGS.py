import numpy as np

class ModifiedAMGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        current_solution = np.random.uniform(bounds[0], bounds[1], self.dim)
        current_value = func(current_solution)
        self.best_solution = current_solution
        self.best_value = current_value

        memory_size = max(5, self.dim)
        step_size = 0.1 * (bounds[1] - bounds[0])
        memory = []

        for _ in range(self.budget - 1):
            if len(memory) >= memory_size:
                memory.pop(0)

            perturbation = np.random.standard_normal(self.dim) * step_size
            trial_solution = current_solution + perturbation
            trial_solution = np.clip(trial_solution, bounds[0], bounds[1])
            trial_value = func(trial_solution)
            
            memory.append((trial_solution, trial_value))

            gradients = np.zeros(self.dim)
            weights = np.linspace(1, 0.5, len(memory))  # Decreasing weights
            weighted_grads = []

            for i, (sol, val) in enumerate(memory):
                diff = sol - current_solution
                if np.linalg.norm(diff) > 1e-8:
                    grad = (val - current_value) * diff / (np.linalg.norm(diff) + 1e-8)
                    weighted_grads.append(weights[i] * grad)
                    
            if weighted_grads:
                gradients = np.sum(weighted_grads, axis=0)

            if np.linalg.norm(gradients) > 1e-6:
                gradients /= (np.linalg.norm(gradients) + 1e-6)

            step_size *= (0.985 + np.random.uniform(-0.005, 0.005))
            direction = np.random.choice([-1, 1], size=self.dim)  # Directional mutation
            new_solution = current_solution - step_size * gradients * direction
            new_solution = np.clip(new_solution, bounds[0], bounds[1])
            new_value = func(new_solution)

            if new_value < self.best_value:
                self.best_value = new_value
                self.best_solution = new_solution

            current_solution = new_solution
            current_value = new_value

        return self.best_solution, self.best_value