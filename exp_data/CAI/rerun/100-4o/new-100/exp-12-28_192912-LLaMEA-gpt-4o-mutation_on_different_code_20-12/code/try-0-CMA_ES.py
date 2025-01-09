import numpy as np

class CMA_ES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lmbda = 4 + int(3 * np.log(dim))  # Population size
        self.mu = self.lmbda // 2  # Number of parents
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1)/(self.dim + 1)) - 1) + self.cs
        self.pcov = np.zeros(self.dim)
        self.psig = np.zeros(self.dim)
        self.sigma = 0.3
        self.B = np.eye(self.dim)
        self.D = np.eye(self.dim)
        self.C = np.eye(self.dim)
        self.eigen_eval = 0
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        xmean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        counteval = 0
        
        while counteval < self.budget:
            arx = np.array([xmean + self.sigma * np.dot(self.B, np.dot(self.D, np.random.randn(self.dim))) for _ in range(self.lmbda)])
            arfitness = np.array([func(x) for x in arx])
            counteval += self.lmbda

            indices = np.argsort(arfitness)
            arx = arx[indices]
            xmean = np.dot(arx[:self.mu].T, self.weights)
            
            self.psig = (1 - self.cs) * self.psig + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(self.B, np.dot(self.D, np.linalg.solve(self.B, xmean - xmean)))
            hsig = np.linalg.norm(self.psig) / np.sqrt(1 - (1 - self.cs)**(2 * counteval/self.lmbda)) < (1.4 + 2/(self.dim+1))
            self.pcov = (1 - self.cc) * self.pcov + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (xmean - xmean)
            
            artmp = (arx[:self.mu] - xmean) / self.sigma
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pcov, self.pcov) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * np.dot(artmp.T * self.weights, artmp)
            
            if counteval - self.eigen_eval > self.lmbda / (self.c1 + self.cmu) / self.dim / 10:
                self.eigen_eval = counteval
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(self.D)
            
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.psig) / np.linalg.norm(np.random.randn(self.dim)) - 1))
            
            if arfitness[0] < self.f_opt:
                self.f_opt = arfitness[0]
                self.x_opt = arx[0]
        
        return self.f_opt, self.x_opt