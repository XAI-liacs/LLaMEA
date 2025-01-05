import numpy as np

class AdvancedSAMC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')

    def __call__(self, func):
        # Initialize the search
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        num_agents = 3  # Number of agents
        solutions = [np.random.uniform(bounds[0], bounds[1], self.dim) for _ in range(num_agents)]
        values = [func(sol) for sol in solutions]
        
        # Initialize best solution
        self.best_solution = solutions[np.argmin(values)]
        self.best_value = np.min(values)

        # Parameters
        memory_size = max(5, self.dim)
        step_size = 0.1 * (bounds[1] - bounds[0])
        memory = [[] for _ in range(num_agents)]

        for _ in range(self.budget - num_agents):
            for i in range(num_agents):
                # Adaptive memory check per agent
                if len(memory[i]) >= memory_size:
                    memory[i].pop(0)

                # Dynamic perturbation strategy
                perturbation = np.random.standard_normal(self.dim) * step_size
                trial_solution = solutions[i] + perturbation
                trial_solution = np.clip(trial_solution, bounds[0], bounds[1])
                trial_value = func(trial_solution)
                memory[i].append((trial_solution, trial_value))

                # Pseudo-gradient calculation
                gradients = np.zeros(self.dim)
                for sol, val in memory[i]:
                    diff = sol - solutions[i]
                    if np.linalg.norm(diff) > 1e-8:
                        gradients += (val - values[i]) * diff / (np.linalg.norm(diff) + 1e-8)

                if np.linalg.norm(gradients) > 1e-6:
                    gradients /= (np.linalg.norm(gradients) + 1e-6)

                step_size *= (0.99 + np.random.uniform(-0.01, 0.01))  # Enhanced learning rate adaptability
                new_solution = solutions[i] - step_size * gradients
                new_solution = np.clip(new_solution, bounds[0], bounds[1])
                new_value = func(new_solution)

                # Update best found solution
                if new_value < self.best_value:
                    self.best_value = new_value
                    self.best_solution = new_solution

                # Prepare for next iteration
                solutions[i] = new_solution
                values[i] = new_value

        return self.best_solution, self.best_value