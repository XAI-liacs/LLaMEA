import numpy as np

class CMAESInspiredOptimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lambda_ = 4 + int(3 * np.log(dim))  # Population size
        self.mu = self.lambda_ // 2  # Number of parents
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
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
        self.eigeneval = 0
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        self.xmean = np.random.uniform(-5, 5, self.dim)
        self.sigma = 0.3  # Step size
        self.f_opt = np.Inf
        self.x_opt = None
        self.lr = 0.2  # Learning rate for adaptive step size

    def __call__(self, func):
        for iteration in range(self.budget // self.lambda_):
            arz = np.random.randn(self.lambda_, self.dim)
            arx = self.xmean + self.sigma * np.dot(arz, self.B * self.D)
            arx = np.clip(arx, func.bounds.lb, func.bounds.ub)
            arfitness = np.array([func(x) for x in arx])

            if np.min(arfitness) < self.f_opt:
                self.f_opt = np.min(arfitness)
                self.x_opt = arx[np.argmin(arfitness)]

            arindex = np.argsort(arfitness)
            arx = arx[arindex]
            arz = arz[arindex]

            self.xmean = np.dot(self.weights, arx[:self.mu])
            zmean = np.dot(self.weights, arz[:self.mu])
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(self.B, zmean)

            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * iteration / self.lambda_)) / self.chiN < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * np.dot(self.B, np.dot(self.D, zmean))

            artmp = (1 / self.sigma) * (arx[:self.mu] - self.xmean)
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * np.dot((self.weights * artmp).T, artmp)

            if iteration - self.eigeneval > self.lambda_ / (self.c1 + self.cmu) / self.dim / 10:
                self.eigeneval = iteration
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(self.D)
                self.invB = np.linalg.inv(self.B)
                self.invsqrtC = np.dot(self.B, np.dot(np.diag(1 / self.D), self.invB))

            self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)
            self.sigma *= np.exp(self.lr * (self.f_opt - np.min(arfitness)))  # Adaptive scaling

        return self.f_opt, self.x_opt