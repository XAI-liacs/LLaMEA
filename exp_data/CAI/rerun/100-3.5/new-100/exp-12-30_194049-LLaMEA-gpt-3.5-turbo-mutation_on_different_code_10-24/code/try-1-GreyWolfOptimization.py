class GreyWolfOptimization:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.alpha = np.zeros(dim)
        self.beta = np.zeros(dim)
        self.delta = np.zeros(dim)
        self.a = 2.0
    
    def update_parameter_a(self, t):
        self.a = 2 * np.cos(t * np.pi / 2 / self.budget)  # Cosine-based adaptation
        
    def __call__(self, func):
        population = self.initialize_population()
        self.update_alpha_beta_delta(population, func)

        for t in range(1, self.budget + 1):
            self.update_parameter_a(t)
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)
            r3 = np.random.uniform(0, 1)

            for i in range(self.dim):
                A1 = 2 * self.a * r1 - self.a
                C1 = 2 * r2
                A2 = 2 * self.a * r3 - self.a
                C2 = 2 * r2

                if np.abs(A1) < 1:
                    D_alpha = np.abs(C1 * self.alpha[i] - population[0][i])
                    X1 = self.alpha[i] - A1 * D_alpha
                elif np.abs(A1) >= 1:
                    rand_alpha = np.random.randint(0, 3)
                    D_alpha = np.abs(C1 * self.alpha[i] - population[rand_alpha][i])
                    X1 = self.alpha[i] - A1 * D_alpha

                if np.abs(A2) < 1:
                    D_beta = np.abs(C2 * self.beta[i] - population[1][i])
                    X2 = self.beta[i] - A2 * D_beta
                elif np.abs(A2) >= 1:
                    rand_beta = np.random.randint(0, 3)
                    D_beta = np.abs(C2 * self.beta[i] - population[rand_beta][i])
                    X2 = self.beta[i] - A2 * D_beta

                D_delta = np.abs(C2 * self.delta[i] - population[2][i])
                X3 = self.delta[i] - A2 * D_delta

                population[0][i] = np.clip(X1, self.lb, self.ub)
                population[1][i] = np.clip(X2, self.lb, self.ub)
                population[2][i] = np.clip(X3, self.lb, self.ub)

            self.update_alpha_beta_delta(population, func)

        f_opt = func(self.alpha)
        return f_opt, self.alpha