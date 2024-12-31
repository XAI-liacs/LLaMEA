import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        x_mean = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        sigma = 0.5  # Adjusted initial step size
        lambda_ = 4 + int(3 * np.log(self.dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)

        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff)
        d_sigma = 1.5 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma  # Adjusted damping factor

        p_c = np.zeros(self.dim)
        p_sigma = np.zeros(self.dim)
        C = np.eye(self.dim)
        B, D = np.linalg.eigh(C)
        inv_sqrt_C = np.dot(B, np.dot(np.diag(1 / np.sqrt(D)), B.T))
        
        restart_factor = 5 # New restart mechanism
        budget_step = self.budget // (lambda_ * restart_factor) # Adjusted budget distribution

        for _ in range(budget_step):
            arz = np.random.randn(lambda_, self.dim)
            ary = np.dot(arz, B * D) * sigma
            arx = x_mean + ary
            arx = np.clip(arx, func.bounds.lb, func.bounds.ub)
            
            fitness = np.apply_along_axis(func, 1, arx)
            indices = np.argsort(fitness)
            fitness = fitness[indices]
            arx = arx[indices]

            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = arx[0]

            x_old = x_mean
            x_mean = np.dot(weights, arx[:mu])
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.dot(inv_sqrt_C, x_mean - x_old) / sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (_ + 1))) < (1.4 + 2 / (self.dim + 1))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma

            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C))
            for k in range(mu):
                C += c_mu * weights[k] * np.outer(arx[k] - x_old, arx[k] - x_old) / sigma**2

            B, D = np.linalg.eigh(C)
            inv_sqrt_C = np.dot(B, np.dot(np.diag(1 / np.sqrt(D)), B.T))
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (_ + 1))) - 1))

        return self.f_opt, self.x_opt