class ImprovedFireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return np.exp(-self.beta0 * np.linalg.norm(x - y)**2)

    def move_firefly(self, x, y, alpha, beta):
        beta = beta * np.exp(-alpha * np.linalg.norm(x - y)**2)
        return x + beta * (y - x) + 0.01 * np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(self.budget):
                alpha = self.alpha / (1 + i)  # Dynamic alpha update
                beta = self.beta0 / (1 + i)   # Dynamic beta update
                if func(population[j]) < func(population[i]):
                    population[i] = self.move_firefly(population[i], population[j], alpha, beta)
            
        f_best = np.min([func(x) for x in population])
        x_best = population[np.argmin([func(x) for x in population])]
        
        return f_best, x_best