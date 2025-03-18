import numpy as np

class StochasticGradientDescentALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        self.evaluations = 0  # Reset evaluations counter
        bounds = func.bounds
        initial_points = self.uniform_sampling(bounds, 10)
        best_solution = None
        best_score = np.inf

        for point in initial_points:
            solution, score = self.optimize_from_point(func, point, bounds)
            if score < best_score:
                best_solution = solution
                best_score = score

            if self.evaluations >= self.budget:
                break

        return best_solution, best_score

    def uniform_sampling(self, bounds, num_samples):
        lb, ub = bounds.lb, bounds.ub
        samples = [lb + np.random.rand(self.dim) * (ub - lb) for _ in range(num_samples)]
        return samples

    def optimize_from_point(self, func, start_point, bounds):
        current_solution = np.array(start_point)
        learning_rate = 0.1
        decay_rate = 0.95
        tolerance = 1e-6
        
        while self.evaluations < self.budget:
            gradient = self.estimate_gradient(func, current_solution)
            new_solution = current_solution - learning_rate * gradient
            
            # Ensure the new solution is within bounds
            new_solution = np.clip(new_solution, bounds.lb, bounds.ub)
            
            score = func(new_solution)
            self.evaluations += 1

            if np.linalg.norm(gradient) < tolerance:
                break
            
            learning_rate *= decay_rate

            if score < func(current_solution):
                current_solution = new_solution

        return current_solution, func(current_solution)

    def estimate_gradient(self, func, point, epsilon=1e-8):
        gradient = np.zeros(self.dim)
        fx = func(point)
        for i in range(self.dim):
            point_eps = np.copy(point)
            point_eps[i] += epsilon
            gradient[i] = (func(point_eps) - fx) / epsilon
            self.evaluations += 1
            if self.evaluations >= self.budget:
                break
        return gradient