import numpy as np

class DolphinEcholocationOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_dolphins = 15
        self.best_solution = None
        self.best_obj = float('inf')
        self.learning_rate = 0.05  # social learning rate
        self.echolocate_radius = 0.1  # initial echolocation radius

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.num_dolphins, self.dim))
        self.lb, self.ub = bounds.lb, bounds.ub

    def echolocate(self, dolphin):
        direction = np.random.uniform(-1, 1, self.dim)
        distance = np.random.uniform(0, self.echolocate_radius)
        return np.clip(dolphin + direction * distance, self.lb, self.ub)

    def social_learning(self, dolphin, global_best):
        movement = self.learning_rate * (global_best - dolphin)
        return np.clip(dolphin + movement, self.lb, self.ub)

    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            new_population = []
            for dolphin in self.population:
                echolocation_solution = self.echolocate(dolphin)
                social_solution = self.social_learning(dolphin, self.best_solution if self.best_solution is not None else dolphin)
                solutions = [echolocation_solution, social_solution]
                
                # Evaluate both solutions
                objs = [func(sol) for sol in solutions]
                evaluations += len(solutions)
                
                # Choose the better solution
                best_idx = np.argmin(objs)
                new_population.append(solutions[best_idx])
                
                # Update global best
                if objs[best_idx] < self.best_obj:
                    self.best_obj = objs[best_idx]
                    self.best_solution = solutions[best_idx]
                
                # Early stopping if budget exceeded
                if evaluations >= self.budget:
                    break

            # Update the population
            self.population = np.array(new_population)
            # Dynamic adjustment of echolocation radius
            self.echolocate_radius *= 0.99

        return self.best_solution