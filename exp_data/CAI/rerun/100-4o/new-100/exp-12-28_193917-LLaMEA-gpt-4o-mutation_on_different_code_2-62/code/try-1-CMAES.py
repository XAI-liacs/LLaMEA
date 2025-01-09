import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 4 + int(3 * np.log(dim))
        self.sigma = 0.5  # Adjusted from 0.3 to 0.5
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        mean = np.random.uniform(lb, ub)
        C = np.eye(self.dim)
        B = np.eye(self.dim)
        D = np.ones(self.dim)
        inv_sqrt_C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))

        mu = self.population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = np.sum(weights)**2 / np.sum(weights**2)

        c1 = 2 / ((self.dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((self.dim + 2)**2 + mueff))
        cs = (mueff + 2) / (self.dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1)/ (self.dim + 1)) - 1) + cs
        cc = (4 + mueff/self.dim) / (self.dim + 4 + 2*mueff/self.dim)

        for iteration in range(0, self.budget, self.population_size):
            arz = np.random.randn(self.population_size, self.dim)
            arx = mean + self.sigma * (arz @ B @ np.diag(D))
            arx = np.clip(arx, lb, ub)

            fitness = np.array([func(x) for x in arx])
            if np.min(fitness) < self.f_opt:
                self.f_opt = np.min(fitness)
                self.x_opt = arx[np.argmin(fitness)]

            if iteration + self.population_size >= self.budget:
                break

            indices = np.argsort(fitness)
            arx = arx[indices]
            arz = arz[indices]
            best_arx = arx[:mu]
            best_arz = arz[:mu]

            mean_old = mean
            mean = np.dot(weights, best_arx)

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C @ (mean - mean_old) / self.sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) / chiN < 1.4 + 2/(self.dim + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - mean_old) / self.sigma

            artmp = best_arz.T @ np.diag(weights)
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * artmp @ artmp.T
            C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry

            B, D = np.linalg.eigh(C)
            D = np.sqrt(D)
            inv_sqrt_C = B @ np.diag(1/D) @ B.T

        return self.f_opt, self.x_opt