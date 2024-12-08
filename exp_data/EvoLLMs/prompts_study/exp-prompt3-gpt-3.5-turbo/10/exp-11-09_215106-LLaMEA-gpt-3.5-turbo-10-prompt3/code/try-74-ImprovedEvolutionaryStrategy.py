import numpy as np

class ImprovedEvolutionaryStrategy(EvolutionaryStrategy):
    def __call__(self, func):
        mu = self.mu
        lambda_ = self.lambda_
        sigma = self.sigma
        pc = self.pc
        ps = self.ps
        C = self.C
        D = self.D
        invsqrtC = self.invsqrtC
        
        x_mean = np.random.uniform(-5.0, 5.0, self.dim)
        x = np.random.uniform(-5.0, 5.0, (mu, self.dim))
        fitness = np.array([func(x_i) for x_i in x])
        
        for _ in range(self.budget // lambda_):
            x_old = x.copy()
            fitness_old = fitness.copy()
            
            for i in range(lambda_):
                z = np.random.normal(0, 1, self.dim)
                x[i] = x_mean + sigma * (D * (invsqrtC @ z))
                fitness[i] = func(x[i])
            
            idx = np.argsort(fitness)
            x = x[idx[:mu]]
            x_mean = np.mean(x, axis=0)
            
            z = np.random.normal(0, 1, self.dim)
            ps = (1 - 0.1) * ps + np.sqrt(0.1 * (2 - 0.1)) * (z < 0)
            pc = (1 - 0.4) * pc + np.sqrt(0.4 * (2 - 0.4)) * (z >= 0)
            
            cSigma = (2 * pc * np.sqrt(1 - (1 - pc)**2)) / np.sqrt(self.dim)
            C = np.dot(C, np.dot(np.diagflat(1 - cSigma), C)) + np.outer(cSigma, cSigma)
            D = D * np.exp(0.0873 * (np.linalg.norm(ps) / np.sqrt(self.dim)) - 1)
            invsqrtC = np.linalg.inv(np.linalg.cholesky(C).T)
            
            sigma = sigma * np.exp((np.linalg.norm(ps) - 0.2) / 0.3)
            
            if np.random.rand() < 0.1:
                x = np.random.uniform(-5.0, 5.0, (mu, self.dim))
                fitness = np.array([func(x_i) for x_i in x])
            
        return x[0]