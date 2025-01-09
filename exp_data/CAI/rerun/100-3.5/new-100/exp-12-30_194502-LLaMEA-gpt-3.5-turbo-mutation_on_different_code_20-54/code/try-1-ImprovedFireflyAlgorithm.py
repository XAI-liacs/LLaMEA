import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y, iter_count):
        beta = self.beta0 * np.exp(-self.gamma * np.linalg.norm(x - y)**2)
        return beta + (1 / (1 + iter_count))  # Adaptive attractiveness update

    def move_firefly(self, x, y, iter_count):
        beta = self.attractiveness(x, y, iter_count)
        return x + beta * (y - x) + self.alpha * (np.random.rand(self.dim) - 0.5)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        intensities = np.array([func(firefly) for firefly in fireflies])
        
        for i in range(self.budget):
            for j in range(self.budget):
                if intensities[j] < intensities[i]:  # Brighter firefly found
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], i)
                    intensities[i] = func(fireflies[i])
            
            # Update global best
            min_index = np.argmin(intensities)
            if intensities[min_index] < self.f_opt:
                self.f_opt = intensities[min_index]
                self.x_opt = fireflies[min_index]
                
        return self.f_opt, self.x_opt