import numpy as np

class GreyWolfOptimization:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.alpha = np.zeros(dim)
        self.beta = np.zeros(dim)
        self.delta = np.zeros(dim)
    
    def initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (3, self.dim))

    def update_alpha_beta_delta(self, population, func):
        fitness = [func(individual) for individual in population]
        alpha_index = np.argmin(fitness)
        beta_index = np.argsort(fitness)[1]
        delta_index = np.argsort(fitness)[2]

        self.alpha = population[alpha_index]
        self.beta = population[beta_index]
        self.delta = population[delta_index]

    def __call__(self, func):
        population = self.initialize_population()
        self.update_alpha_beta_delta(population, func)

        for _ in range(self.budget):
            a = 2 - 2 * _ / self.budget  # a decreases linearly from 2 to 0
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)
            r3 = np.random.uniform(0, 1)

            for i in range(self.dim):
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                A2 = 2 * a * r3 - a
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