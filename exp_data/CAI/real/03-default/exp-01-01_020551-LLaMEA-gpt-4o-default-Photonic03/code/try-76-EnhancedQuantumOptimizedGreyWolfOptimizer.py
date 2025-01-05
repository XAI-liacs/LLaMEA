import numpy as np

class EnhancedQuantumOptimizedGreyWolfOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.n_wolves = max(5, dim + 2)
        self.alpha, self.beta, self.delta = None, None, None
        self.qbeta = 0.25  # Quantum behavior probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        wolves = np.random.uniform(lb, ub, (self.n_wolves, self.dim))
        scores = np.array([func(wolves[i]) for i in range(self.n_wolves)])
        evaluations = self.n_wolves

        while evaluations < self.budget:
            sorted_indices = np.argsort(scores)
            self.alpha, self.beta, self.delta = wolves[sorted_indices[0]], wolves[sorted_indices[1]], wolves[sorted_indices[2]]

            a = 2 - evaluations * (2 / self.budget)  # Linearly decreases from 2 to 0

            new_wolves = np.empty_like(wolves)
            for i in range(self.n_wolves):
                A1, C1 = 2 * a * np.random.random(self.dim) - a, 2 * np.random.random(self.dim)
                A2, C2 = 2 * a * np.random.random(self.dim) - a, 2 * np.random.random(self.dim)
                A3, C3 = 2 * a * np.random.random(self.dim) - a, 2 * np.random.random(self.dim)

                D_alpha = np.abs(C1 * self.alpha - wolves[i])
                D_beta = np.abs(C2 * self.beta - wolves[i])
                D_delta = np.abs(C3 * self.delta - wolves[i])

                X1 = self.alpha - A1 * D_alpha
                X2 = self.beta - A2 * D_beta
                X3 = self.delta - A3 * D_delta

                new_position = (X1 + X2 + X3) / 3

                # Quantum-inspired update
                if np.random.rand() < self.qbeta:
                    q = np.random.randn(self.dim) * 0.1  # Controlled quantum step
                    new_position += q * (ub - lb)

                new_position = np.clip(new_position, lb, ub)
                new_wolves[i] = new_position

            new_scores = np.array([func(new_wolves[i]) for i in range(self.n_wolves)])
            evaluations += self.n_wolves

            improved = new_scores < scores
            wolves[improved] = new_wolves[improved]
            scores[improved] = new_scores[improved]

        best_index = np.argmin(scores)
        return wolves[best_index], scores[best_index]