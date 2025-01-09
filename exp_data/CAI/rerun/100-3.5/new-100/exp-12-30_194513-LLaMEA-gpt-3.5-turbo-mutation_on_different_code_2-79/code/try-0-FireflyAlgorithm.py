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

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)
    
    def move_fireflies(self, current_pos, best_pos, func):
        for i in range(len(current_pos)):
            for j in range(len(current_pos)):
                if func(current_pos[j]) < func(current_pos[i]):
                    r = np.linalg.norm(current_pos[j] - current_pos[i])
                    beta = self.attractiveness(r)
                    current_pos[i] = current_pos[i] + beta * (current_pos[j] - current_pos[i]) + self.alpha * (np.random.rand(self.dim) - 0.5)
        
        return current_pos

    def __call__(self, func):
        current_pos = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        
        for i in range(self.budget):
            current_pos = self.move_fireflies(current_pos, current_pos, func)
            
            for j in range(len(current_pos)):
                f = func(current_pos[j])
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = current_pos[j]
                
        return self.f_opt, self.x_opt