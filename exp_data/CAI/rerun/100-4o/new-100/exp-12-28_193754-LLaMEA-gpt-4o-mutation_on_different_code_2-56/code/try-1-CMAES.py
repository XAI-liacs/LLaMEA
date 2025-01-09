import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lambda_ = 4 + int(3 * np.log(self.dim))  # Population size
        mu = lambda_ // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        sigma = 0.3 * (func.bounds.ub - func.bounds.lb) * (1 / np.sqrt(self.dim))  # Initial step-size
        C = np.eye(self.dim)  # Covariance matrix
        pc = np.zeros(self.dim)  # Evolution path for C
        ps = np.zeros(self.dim)  # Evolution path for sigma
        B = np.eye(self.dim)  # B defines the coordinate system
        D = np.ones(self.dim)  # Diagonal of D defines the scaling
        invsqrtC = B @ np.diag(1 / D) @ B.T

        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_s = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_s

        xmean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)  # Initial guess
        eigensystem_eval_interval = self.dim * 10

        for iteration in range(0, self.budget, lambda_):
            arz = np.random.randn(lambda_, self.dim)
            arx = np.array([xmean + sigma * B @ (D * arz[k]) for k in range(lambda_)])
            fitnesses = np.array([func(x) for x in arx])
            
            if np.min(fitnesses) < self.f_opt:
                self.f_opt = np.min(fitnesses)
                self.x_opt = arx[np.argmin(fitnesses)]

            if iteration + lambda_ >= self.budget:
                break

            indices = np.argsort(fitnesses)
            arx = arx[indices]
            arz = arz[indices]
            xmean_old = xmean
            xmean = np.dot(weights, arx[:mu])
            zmean = np.dot(weights, arz[:mu])
            
            ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * invsqrtC @ zmean
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp((c_s / damps) * (norm_ps / np.sqrt(1 - (1 - c_s)**(2 * iteration / lambda_)) - 1))
            
            hsig = int((norm_ps / np.sqrt(1 - (1 - c_s)**(2 * iteration / lambda_))) / np.sqrt(1 + c1 - c1) < (1.4 + 2 / (self.dim + 1)))
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (xmean - xmean_old) / sigma
            
            artmp = (1 / sigma) * (arx[:mu] - xmean_old)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C) + cmu * np.dot((weights * artmp.T), artmp)
            
            if iteration % eigensystem_eval_interval < lambda_:
                C = np.triu(C) + np.triu(C, 1).T
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.diag(D))
                invsqrtC = B @ np.diag(1 / D) @ B.T

        return self.f_opt, self.x_opt