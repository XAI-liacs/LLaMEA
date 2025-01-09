import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.sigma = 0.5  # Adjusted initial sigma for better exploration
        self.lambda_ = int(4 + np.floor(3 * np.log(self.dim)))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff/self.dim) / (self.dim + 4 + 2 * self.mueff/self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1)/(self.dim + 1)) - 1) + self.cs
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = np.eye(self.dim)
        self.invsqrtC = np.eye(self.dim)
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.mean = np.random.uniform(-5, 5, self.dim)

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            arz = np.random.randn(self.lambda_, self.dim)
            arx = self.mean + self.sigma * (arz @ (self.B * self.D).T)
            arx = np.clip(arx, -5, 5)
            arfitness = np.apply_along_axis(func, 1, arx)
            evals += self.lambda_

            indices = np.argsort(arfitness)
            arfitness = arfitness[indices]
            arz = arz[indices]
            arx = arx[indices]
            
            if arfitness[0] < self.f_opt:
                self.f_opt = arfitness[0]
                self.x_opt = arx[0]
            
            self.mean = arx[:self.mu].T @ self.weights

            zmean = arz[:self.mu].T @ self.weights
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.invsqrtC @ zmean)
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * evals / self.lambda_)) / self.chiN < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.B @ self.D @ zmean)
            artmp = (arz[:self.mu].T @ np.diag(self.weights / self.sigma)) @ (arz[:self.mu])
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * artmp
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1) * 0.9)  # Adaptive damping for step-size
            
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            E, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(E)
            self.invsqrtC = self.B @ np.diag(1/self.D) @ self.B.T

        return self.f_opt, self.x_opt