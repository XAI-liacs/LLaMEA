class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=0.2):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_firefly(self, x_i, x_j, f_i, f_j):
        r = np.linalg.norm(x_i - x_j)
        beta = self.attractiveness(r)
        epsilon = self.alpha * (np.random.rand(self.dim) - 0.5)
        x_new = x_i + beta * (x_j - x_i) + epsilon
        f_new = func(x_new)
        if f_new < f_i:
            return x_new, f_new
        else:
            return x_i, f_i

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        intensities = np.array([func(ff) for ff in fireflies])
        
        for i in range(self.budget):
            for j in range(self.budget):
                if intensities[j] < intensities[i]:
                    fireflies[i], intensities[i] = self.move_firefly(fireflies[i], fireflies[j], intensities[i], intensities[j])
            
            self.alpha = max(0.05, self.alpha * 0.99)  # Dynamic alpha update

        best_idx = np.argmin(intensities)
        self.f_opt = intensities[best_idx]
        self.x_opt = fireflies[best_idx]
        
        return self.f_opt, self.x_opt