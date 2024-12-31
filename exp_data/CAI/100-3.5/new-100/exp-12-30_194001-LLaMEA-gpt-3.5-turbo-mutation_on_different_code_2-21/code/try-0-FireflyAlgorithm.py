import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_fireflies(self, current_x, current_f, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if func(current_x[j]) < current_f[i]:
                    current_x[j] += self.alpha * (current_x[i] - current_x[j]) + self.attractiveness(np.linalg.norm(current_x[i] - current_x[j])) * np.random.uniform(-1, 1, self.dim)
                    current_f[j] = func(current_x[j])
            min_index = np.argmin(current_f)
            if current_f[min_index] < self.f_opt:
                self.f_opt = current_f[min_index]
                self.x_opt = current_x[min_index]

    def __call__(self, func):
        current_x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        current_f = np.array([func(x) for x in current_x])

        self.move_fireflies(current_x, current_f, func)
        
        return self.f_opt, self.x_opt