import numpy as np

class AdaptiveCMA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lambda_ = int(4 + np.floor(3 * np.log(dim)))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))
        self.sigma = 0.3
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.eye(dim)
        self.C = self.B @ self.D @ (self.B @ self.D).T
        self.f_opt = np.Inf
        self.x_opt = np.random.uniform(-5, 5, dim)
        
    def __call__(self, func):
        remaining_budget = self.budget
        
        while remaining_budget > 0:
            arz = np.random.randn(self.lambda_, self.dim)
            y = arz @ np.diag(np.diag(self.D))
            x = self.x_opt + self.sigma * y @ self.B.T
            
            fitness = np.asarray([func(xi) for xi in x])
            remaining_budget -= self.lambda_
            
            indices = np.argsort(fitness)
            x = x[indices]
            z = arz[indices]
            self.x_opt = np.dot(self.weights, x[:self.mu])
            y_w = np.dot(self.weights, z[:self.mu])
            
            self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * y_w
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.c_sigma)**(2 * remaining_budget / self.budget)) < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.c_c) * self.pc + hsig * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * np.dot(self.weights, y[:self.mu])
            
            C_new = np.dot(y[:self.mu].T, np.diag(self.weights) @ y[:self.mu])
            self.C = (1 - self.c1 - self.c_mu) * self.C + self.c1 * np.outer(self.pc, self.pc) + self.c_mu * C_new
            
            self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.c_sigma)**(2 * remaining_budget / self.budget)) - 1))
            
            if remaining_budget > 0:
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(self.D)
            
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = x[0]
        
        return self.f_opt, self.x_opt