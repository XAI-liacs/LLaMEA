import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.lambda_ = 4 + int(3 * np.log(self.dim))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.weights += 0.1 * (1 - self.weights)  # Enhance exploration early
        self.mueff = 1 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = np.eye(self.dim)
        self.invsqrtC = np.eye(self.dim)
        self.eigen_eval = 0
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        self.sigma = 0.3
        self.success_rate = 0.2  # Initialize success rate

    def __call__(self, func):
        xmean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        for generation in range(self.budget // self.lambda_):
            arz = np.random.randn(self.dim, self.lambda_)
            arx = xmean[:, None] + self.sigma * (self.B @ (self.D * arz))
            arx = np.clip(arx, func.bounds.lb, func.bounds.ub)
            arfitness = np.array([func(x) for x in arx.T])
            sorted_indices = np.argsort(arfitness)
            xold = xmean
            xmean = arx[:, sorted_indices[:self.mu]] @ self.weights

            zmean = arz[:, sorted_indices[:self.mu]] @ self.weights
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.invsqrtC @ zmean)
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * generation / self.lambda_)) / self.chiN < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (xmean - xold) / self.sigma

            artmp = (1 / self.sigma) * (arx[:, sorted_indices[:self.mu]] - xold[:, None])
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) \
                    + self.cmu * artmp @ np.diag(self.weights) @ artmp.T

            if generation % (self.lambda_ / self.dim / 10) < 1:
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(self.D)
                self.invsqrtC = self.B @ np.diag(1 / self.D) @ self.B.T

            self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)
            if arfitness[sorted_indices[0]] < self.f_opt:
                self.f_opt = arfitness[sorted_indices[0]]
                self.x_opt = arx[:, sorted_indices[0]]
                self.success_rate = 0.8 * self.success_rate + 0.2  # Update success rate

            if generation % 10 == 0:  # Adjust sigma based on success rate every 10 generations
                if self.success_rate < 0.2:
                    self.sigma *= 0.9
                elif self.success_rate > 0.3:
                    self.sigma *= 1.1
                self.success_rate = 0.2  # Reset for next interval

        return self.f_opt, self.x_opt