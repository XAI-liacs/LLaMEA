import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.sigma_initial = 0.3 * (self.upper_bound - self.lower_bound)
        self.lamda = 4 + int(3 * np.log(dim))
        self.mu = self.lamda // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C = np.eye(dim)
        self.eigen_updated = 0
        self.chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    def __call__(self, func):
        xmean = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        sigma = self.sigma_initial
        eval_count = 0

        while eval_count < self.budget:
            if eval_count + self.lamda > self.budget:
                break

            arz = np.random.randn(self.lamda, self.dim)
            ary = arz @ np.diag(self.D)
            arz = arz @ self.B
            arx = xmean + sigma * ary
            arx = np.clip(arx, self.lower_bound, self.upper_bound)
            arfitness = np.array([func(x) for x in arx])
            eval_count += self.lamda

            arindex = np.argsort(arfitness)
            arfitness = arfitness[arindex]
            xold = xmean
            xmean = self.weights @ arx[arindex[:self.mu]]

            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (xmean - xold) / sigma
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * eval_count / self.lamda)) / self.chiN < (1.4 + 2 / (self.dim + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (xmean - xold) / sigma
            artmp = (arx[arindex[:self.mu]] - xold) / sigma
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc @ self.pc.T + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * artmp.T @ (np.diag(self.weights) @ artmp)

            sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)

            if eval_count - self.eigen_updated > self.lamda / (self.c1 + self.cmu) / self.dim / 10:
                self.eigen_updated = eval_count
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(self.D)