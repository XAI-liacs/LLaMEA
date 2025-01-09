import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.sigma = 0.5
        self.lambda_ = 4 + int(3 * np.log(self.dim))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.eigen_eval = 0
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = np.eye(self.dim)
        self.invsqrtC = np.eye(self.dim)
        self.arx = np.zeros((self.dim, self.lambda_))
        self.arfitness = np.zeros(self.lambda_)
        self.elitist_archive = []  # Added line
        self.counteval = 0
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        xmean = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        while self.counteval < self.budget:
            for k in range(self.lambda_):
                self.arx[:, k] = xmean + self.sigma * self.B @ (self.D * np.random.randn(self.dim))
                self.arx[:, k] = np.clip(self.arx[:, k], self.bounds[0], self.bounds[1])
                self.arfitness[k] = func(self.arx[:, k])
                self.counteval += 1
            
            arindex = np.argsort(self.arfitness)
            arx_sorted = self.arx[:, arindex]
            x_old = xmean
            xmean = np.dot(arx_sorted[:, :self.mu], self.weights)

            if self.arfitness[arindex[0]] < self.f_opt:
                self.f_opt = self.arfitness[arindex[0]]
                self.x_opt = arx_sorted[:, 0]
                self.elitist_archive.append((self.f_opt, self.x_opt))  # Added line

            Z = (arx_sorted[:, :self.mu] - x_old[:, None]) / self.sigma
            C_new = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc)) + self.cmu * (Z * self.weights).dot(Z.T)
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ (xmean - x_old) / self.sigma
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.counteval / self.lambda_)) / 1.4 < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (xmean - x_old) / self.sigma
            self.C = C_new

            if self.counteval - self.eigen_eval > np.floor(self.lambda_ / (self.c1 + self.cmu) / self.dim / 10):
                self.eigen_eval = self.counteval
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(self.D)
                self.invsqrtC = self.B @ np.diag(1/self.D) @ self.B.T

            self.sigma *= np.exp((np.linalg.norm(self.ps)/self.chi_n - 1) * self.cs / self.damps)

        return self.f_opt, self.x_opt