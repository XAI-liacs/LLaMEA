import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.lambda_ = 4 + int(3 * np.log(dim))  # Population size
        self.mu = self.lambda_ // 2  # Number of parents
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C = np.eye(dim)
        self.invsqrtC = np.eye(dim)
        self.eigenval = 0
        self.chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
    
    def __call__(self, func):
        xmean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        sigma = 0.3
        counteval = 0

        while counteval < self.budget:
            arz = np.random.randn(self.lambda_, self.dim)
            arx = xmean + sigma * (arz @ self.B.T) * self.D
            arx_valid = np.clip(arx, func.bounds.lb, func.bounds.ub)
            arfitness = np.array([func(x) for x in arx_valid])
            counteval += self.lambda_
            idx = np.argsort(arfitness)
            arx = arx_valid[idx]
            xmean_old = xmean
            xmean = np.dot(self.weights, arx[:self.mu])
            zmean = np.dot(self.weights, arz[idx[:self.mu]])
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.invsqrtC @ zmean)
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * counteval / self.lambda_)) / self.chiN < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (xmean - xmean_old) / sigma
            artmp = (arx[:self.mu] - xmean_old) / sigma
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]) + self.cmu * (artmp.T @ np.diag(self.weights) @ artmp)
            sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            if counteval - self.eigenval > self.lambda_ / (self.c1 + self.cmu) / self.dim / 10:
                self.eigenval = counteval
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(np.maximum(self.D, 0))
                self.invsqrtC = self.B @ np.diag(1 / self.D) @ self.B.T
            if arfitness[0] < self.f_opt:
                self.f_opt = arfitness[0]
                self.x_opt = arx[0]
                
        return self.f_opt, self.x_opt