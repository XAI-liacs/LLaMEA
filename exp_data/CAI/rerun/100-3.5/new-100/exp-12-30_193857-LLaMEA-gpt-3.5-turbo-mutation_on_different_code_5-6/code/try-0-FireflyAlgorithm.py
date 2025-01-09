import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=2.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def attractiveness(r):
            return self.beta0 * np.exp(-self.gamma * r**2)
        
        def move_firefly(x, intensity, x_other):
            r = np.linalg.norm(x - x_other)
            beta = attractiveness(r)
            return x + beta * (x_other - x) + self.alpha * intensity * (np.random.rand(self.dim) - 0.5)
        
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[i]) < func(fireflies[j]):
                    intensity = 1 / (1 + func(fireflies[i]) - func(fireflies[j]))
                    fireflies[i] = move_firefly(fireflies[i], intensity, fireflies[j])
            
        idx = np.argmin([func(f) for f in fireflies])
        self.f_opt = func(fireflies[idx])
        self.x_opt = fireflies[idx]
        
        return self.f_opt, self.x_opt