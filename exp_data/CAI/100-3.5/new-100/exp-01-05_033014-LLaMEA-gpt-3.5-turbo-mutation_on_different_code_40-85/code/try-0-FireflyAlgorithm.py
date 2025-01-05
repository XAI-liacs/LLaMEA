import numpy as np
from scipy.spatial.distance import cdist

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    beta = self.beta0 * np.exp(-self.gamma * np.linalg.norm(fireflies[j] - fireflies[i])**2)
                    fireflies[i] += self.alpha * (fireflies[j] - fireflies[i]) * beta
                    fireflies[i] = np.clip(fireflies[i], func.bounds.lb, func.bounds.ub)
        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        
        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)
        
        f_values = [func(f) for f in fireflies]
        best_idx = np.argmin(f_values)
        
        self.f_opt = f_values[best_idx]
        self.x_opt = fireflies[best_idx]
        
        return self.f_opt, self.x_opt