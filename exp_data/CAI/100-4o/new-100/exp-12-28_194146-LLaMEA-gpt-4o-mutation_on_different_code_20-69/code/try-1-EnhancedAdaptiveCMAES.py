import numpy as np

class EnhancedAdaptiveCMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.sigma = 0.3
        self.lam = int(4 + np.floor(3 * np.log(self.dim)))
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = self.B @ np.diag(self.D**2) @ self.B.T
        self.eigen_update_freq = int(1. / (self.c1 + self.cmu) / self.dim / 10.)
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        self.restart_threshold = 1e-8 * self.dim

    def __call__(self, func):
        x_mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        f_opt = np.inf
        x_opt = None
        evaluations = 0

        while evaluations < self.budget:
            arz = np.random.randn(self.lam, self.dim)
            ary = np.dot(arz, np.diag(self.D))
            arz = np.dot(ary, self.B.T)
            arx = x_mean + self.sigma * arz

            for arxi in arx:
                arxi[:] = np.clip(arxi, func.bounds.lb, func.bounds.ub)

            fitness = np.array([func(x) for x in arx])
            evaluations += self.lam

            indices = np.argsort(fitness)
            x_mean = np.dot(self.weights, arx[indices[:self.mu]])

            y = np.dot(self.weights, ary[indices[:self.mu]])
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * y / self.sigma

            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * evaluations / self.lam)) / self.chiN < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y / self.sigma
            
            artmp = (1 / self.sigma) * ary[indices[:self.mu]]
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * artmp.T @ np.diag(self.weights) @ artmp
            
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

            if evaluations % self.eigen_update_freq == 0:
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(np.clip(self.D, 1e-10, None))

            if f_opt > fitness[indices[0]]:
                f_opt = fitness[indices[0]]
                x_opt = arx[indices[0]]

            if np.min(self.D) < self.restart_threshold:
                x_mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                self.sigma *= 1.5  # Increase mutation strength on restart
                self.D = np.ones(self.dim)
                self.B = np.eye(self.dim)
                self.C = self.B @ np.diag(self.D**2) @ self.B.T

        return f_opt, x_opt