import numpy as np

class CMAESDifferentialMutation:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = int(4 + np.floor(3 * np.log(self.dim)))
        self.sigma = 0.5
        self.lambda_ = self.pop_size
        self.mu = self.pop_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = self.B @ np.diag(self.D**2) @ self.B.T
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        self.xmean = np.random.uniform(-5.0, 5.0, self.dim)
        self.evaluations = 0

    def __call__(self, func):
        best_solution = None
        while self.evaluations < self.budget:
            arz = np.random.randn(self.lambda_, self.dim)
            arx = self.xmean + self.sigma * (arz @ self.B @ np.diag(self.D))
            arx = np.clip(arx, -5.0, 5.0)
            arfitness = np.array([func(x) for x in arx])
            self.evaluations += self.lambda_
            sorted_indices = np.argsort(arfitness)
            arfitness = arfitness[sorted_indices]
            arz = arz[sorted_indices]
            arx = arx[sorted_indices]

            if arfitness[0] < self.f_opt:
                self.f_opt = arfitness[0]
                best_solution = arx[0]

            xold = self.xmean
            self.xmean = np.dot(self.weights, arx[:self.mu])

            zmean = np.dot(self.weights, arz[:self.mu])
            ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ zmean)
            self.ps = ps
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.evaluations / self.lambda_)) / self.chiN < (1.4 + 2 / (self.dim + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.xmean - xold) / self.sigma
            artmp = (arz[:self.mu] - zmean).T @ np.diag(self.weights)
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * artmp @ artmp.T
            self.sigma *= np.exp((np.linalg.norm(ps) / self.chiN - 1) * self.cs / self.damps)

            if self.evaluations % self.dim == 0:
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                eigvals, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(eigvals)
                self.B = self.B[:, eigvals.argsort()]
                self.D = self.D[eigvals.argsort()]

        return self.f_opt, best_solution