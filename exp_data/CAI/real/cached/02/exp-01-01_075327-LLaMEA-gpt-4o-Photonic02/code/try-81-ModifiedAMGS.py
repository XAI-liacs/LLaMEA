import numpy as np

class ModifiedAMGS:
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

            # Differential mutation strategy
            if len(memory) > 2:
                idxs = np.random.choice(len(memory), 3, replace=False)
                a, b, c = memory[idxs[0]][0], memory[idxs[1]][0], memory[idxs[2]][0]
                mutant = np.clip(a + 0.5 * (b - c), bounds[0], bounds[1])
            else:
                mutant = np.random.standard_normal(self.dim) * step_size

            # Crossover
            crossover_rate = 0.7
            trial_solution = np.where(np.random.rand(self.dim) < crossover_rate, mutant, current_solution)
            trial_value = func(trial_solution)
            memory.append((trial_solution, trial_value))

            # Compute pseudo-gradient
            gradients = np.zeros(self.dim)
            for sol, val in memory:
                diff = sol - current_solution
                if np.linalg.norm(diff) > 1e-8:
                    gradients += (val - current_value) * diff / (np.linalg.norm(diff) + 1e-8)

            # Update current solution
            if np.linalg.norm(gradients) > 1e-6:
                gradients /= (np.linalg.norm(gradients) + 1e-6)  # Normalize with L2 norm
            step_size *= (0.985 + np.random.uniform(-0.005, 0.005))  # More adaptive learning rate
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