import numpy as np

class SpiralOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0

    def random_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def compute_spiral_trajectory(self, center, radius, theta, scale_factor=0.9):
        delta = radius * np.sin(theta)
        direction = np.random.uniform(-1, 1, self.dim)
        direction /= np.linalg.norm(direction)
        return np.clip(center + delta * direction, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        center = self.random_solution()
        best_solution = center
        best_value = func(center)
        
        radius = (self.upper_bound - self.lower_bound) / 2
        theta = np.pi / 4
        
        while self.evaluations < self.budget:
            next_solution = self.compute_spiral_trajectory(center, radius, theta)
            self.evaluations += 1
            current_value = func(next_solution)
            
            if current_value < best_value:
                best_value = current_value
                best_solution = next_solution
                center = next_solution
                radius *= 0.95  # Reduce radius to exploit
            else:
                theta += np.pi / 12  # Increment angle to explore
                
            if self.evaluations % (self.budget / 10) == 0:
                radius = max(radius, 0.1)  # Reset radius periodically to ensure exploration
        
        return best_solution