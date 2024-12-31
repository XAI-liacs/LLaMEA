import numpy as np

class GreyWolfOptimization:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 5
        self.max_iter = self.budget // self.pop_size
        self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
    
    def __call__(self, func):
        alpha, beta, delta = self.positions
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                fitness = func(self.positions[i])
                if fitness < func(alpha):
                    alpha = self.positions[i]
                elif fitness < func(beta) and fitness > func(alpha):
                    beta = self.positions[i]
                elif fitness < func(delta) and fitness > func(beta):
                    delta = self.positions[i]
            a = 2 - 2 * _ / self.max_iter
            for i in range(self.pop_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                A1, A2, A3 = 2 * a * r1 - a, 2 * a * r2 - a, 2 * np.random.random(self.dim)
                C1, C2, C3 = 2 * r1, 2 * r2, r1
                D_alpha = np.abs(C1 * alpha - self.positions[i])
                D_beta = np.abs(C2 * beta - self.positions[i])
                D_delta = np.abs(C3 * delta - self.positions[i])
                X1 = alpha - A1 * D_alpha
                X2 = beta - A2 * D_beta
                X3 = delta - A3 * D_delta
                self.positions[i] = (X1 + X2 + X3) / 3
        f_opt = np.min([func(alpha), func(beta), func(delta)])
        x_opt = self.positions[np.argmin([func(alpha), func(beta), func(delta)])]
        
        return f_opt, x_opt