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
        return self.beta0 * np.exp(-self.gamma * r ** 2)

    def move_firefly(self, x, attractiveness, x_other):
        r = np.linalg.norm(x - x_other)
        beta = self.attractiveness(r)
        return x + attractiveness * (x_other - x) + self.alpha * (np.random.rand(self.dim) - 0.5)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        intensities = np.array([func(firefly) for firefly in fireflies])

        for i in range(self.budget):
            for j in range(self.budget):
                if intensities[j] < intensities[i]:
                    attractiveness_ij = self.attractiveness(np.linalg.norm(fireflies[i] - fireflies[j]))
                    fireflies[i] = self.move_firefly(fireflies[i], attractiveness_ij, fireflies[j])
                    intensities[i] = func(fireflies[i])
            
            best_index = np.argmin(intensities)
            if intensities[best_index] < self.f_opt:
                self.f_opt = intensities[best_index]
                self.x_opt = fireflies[best_index]
        
        return self.f_opt, self.x_opt