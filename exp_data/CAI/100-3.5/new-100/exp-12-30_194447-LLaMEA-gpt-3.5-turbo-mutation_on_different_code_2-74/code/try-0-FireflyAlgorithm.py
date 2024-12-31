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

    def attractiveness(self, x_i, x_j):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x_i - x_j))

    def __call__(self, func):
        # Initialization
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        f = func(x)
        self.f_opt = f
        self.x_opt = x

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(x) < func(self.x_opt):
                        self.x_opt = x

                    x += self.alpha * (self.attractiveness(self.x_opt, x) * (self.x_opt - x) + np.random.uniform(-1, 1, size=(self.dim,)))
        
        return self.f_opt, self.x_opt