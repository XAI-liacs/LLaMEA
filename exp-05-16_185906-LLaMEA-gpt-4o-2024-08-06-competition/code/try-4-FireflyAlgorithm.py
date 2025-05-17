import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta_min=0.2, gamma=1.0, population_size=20):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha  
        self.beta_min = beta_min  
        self.gamma = gamma  
        self.population_size = population_size

    def __call__(self, func):
        # Initialize fireflies
        fireflies = np.random.uniform(-100, 100, (self.population_size, self.dim))
        intensities = np.array([func(fly) for fly in fireflies])
        best_idx = np.argmin(intensities)
        
        n_eval = self.population_size
        
        while n_eval < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if intensities[j] < intensities[i]:
                        # Calculate attractiveness
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.beta_min + (1.0 - self.beta_min) * np.exp(-self.gamma * r ** 2)
                        
                        # Update position
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + self.alpha * (np.random.rand(self.dim) - 0.5) * 100
                        
                        # Ensure within bounds
                        fireflies[i] = np.clip(fireflies[i], -100, 100)
                        
                        # Evaluate new position
                        intensity = func(fireflies[i])
                        n_eval += 1
                        
                        # Update intensity
                        if intensity < intensities[i]:
                            intensities[i] = intensity
                            if intensity < intensities[best_idx]:
                                best_idx = i
                                
                        # Break if budget is exceeded
                        if n_eval >= self.budget:
                            break

        return intensities[best_idx], fireflies[best_idx]