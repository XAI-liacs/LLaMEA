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

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_fireflies(self, x, f, fireflies):
        for fly in fireflies:
            if f < fly[1]:
                r = np.linalg.norm(x - fly[0])
                beta = self.attractiveness(r)
                x = x + beta * (fly[0] - x) + self.alpha * (np.random.rand(self.dim) - 0.5)
        return x

    def __call__(self, func):
        fireflies = [(np.random.uniform(func.bounds.lb, func.bounds.ub), np.Inf) for _ in range(self.budget)]
        
        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[i][0]) < fireflies[j][1]:
                    fireflies[j] = (self.move_fireflies(fireflies[j][0], func(fireflies[j][0]), fireflies), func(fireflies[j][0]))
            
            if fireflies[i][1] < self.f_opt:
                self.f_opt = fireflies[i][1]
                self.x_opt = fireflies[i][0]
                
        return self.f_opt, self.x_opt