import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lam = 4 + int(3 * np.log(dim))  # population size
        self.mu = self.lam // 2  # number of selected offspring
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C = np.eye(dim)
        self.sigma = 0.3
        self.xmean = np.random.uniform(-5, 5, dim)
        self.counteval = 0
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        while self.counteval < self.budget:
            # Generate offspring
            arz = np.random.randn(self.lam, self.dim)
            ary = np.dot(arz, self.B * self.D)
            arx = self.xmean + self.sigma * ary
            arx = np.clip(arx, -5, 5)
            
            # Evaluate offspring
            arfitness = np.array([func(x) for x in arx])
            self.counteval += self.lam
            
            # Sort by fitness and update the best
            arindex = np.argsort(arfitness)
            self.x_opt = arx[arindex[0]]
            self.f_opt = arfitness[arindex[0]]
            
            # Update population mean
            self.xmean = np.dot(self.weights, arx[arindex[:self.mu]])
            
            # Update evolution paths
            zmean = np.dot(self.weights, arz[arindex[:self.mu]])
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(self.B, zmean)
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lam)) / np.sqrt(self.dim) < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * self.sigma * np.dot(self.B, np.dot(self.D, zmean))
            
            # Adapt covariance matrix
            artmp = (ary[arindex[:self.mu]].T / self.sigma).T
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * np.dot((self.weights[:, np.newaxis] * artmp).T, artmp)
            
            # Adapt step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
            
            # Decomposition of C
            self.B, self.D = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D)
        
        return self.f_opt, self.x_opt