import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y)**2)

    def move_firefly(self, x_i, x_j, beta):
        return x_i + beta * (x_j - x_i) + self.alpha * np.random.normal(size=self.dim)

    def __call__(self, func):
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        f_values = np.apply_along_axis(func, 1, x)
        
        for _ in range(self.budget):
            for i in range(len(x)):
                for j in range(len(x)):
                    if f_values[j] < f_values[i]:
                        beta = self.beta0 * self.attractiveness(x[i], x[j])
                        x[i] = self.move_firefly(x[i], x[j], beta)
                        f_values[i] = func(x[i])
                        
                        if f_values[i] < self.f_opt:
                            self.f_opt = f_values[i]
                            self.x_opt = x[i]
        
        return self.f_opt, self.x_opt