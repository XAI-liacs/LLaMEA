import numpy as np
from scipy.special import gamma

class ALFGWO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = None
        self.beta = None
        self.delta = None
        self.best_value = float('inf')
        self.wolf_pack_size = 20
        self.wolves = []

    def levy_flight(self, L):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return L * step

    def initialize_wolves(self, lb, ub):
        wolves = []
        for _ in range(self.wolf_pack_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            wolves.append({'position': position, 'fitness': float('inf')})
        return wolves

    def update_positions(self, lb, ub):
        a = 2 - 2 * (self.evaluations / self.budget)
        
        for wolf in self.wolves:
            A1, A2, A3 = 2 * a * np.random.rand(self.dim) - a, 2 * a * np.random.rand(self.dim) - a, 2 * a * np.random.rand(self.dim) - a
            C1, C2, C3 = 2 * np.random.rand(self.dim), 2 * np.random.rand(self.dim), 2 * np.random.rand(self.dim)

            D_alpha = np.abs(C1 * self.alpha['position'] - wolf['position'])
            D_beta = np.abs(C2 * self.beta['position'] - wolf['position'])
            D_delta = np.abs(C3 * self.delta['position'] - wolf['position'])

            X1 = self.alpha['position'] - A1 * D_alpha
            X2 = self.beta['position'] - A2 * D_beta
            X3 = self.delta['position'] - A3 * D_delta

            new_position = (X1 + X2 + X3) / 3
            flight = self.levy_flight(0.01)
            wolf['position'] = np.clip(new_position + flight, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        self.wolves = self.initialize_wolves(lb, ub)
        self.evaluations = 0
        
        while self.evaluations < self.budget:
            for wolf in self.wolves:
                wolf['fitness'] = func(wolf['position'])
                self.evaluations += 1
                
                if wolf['fitness'] < self.best_value:
                    self.best_value = wolf['fitness']
                    self.alpha, self.beta, self.delta = sorted(self.wolves, key=lambda x: x['fitness'])[:3]

                if self.evaluations >= self.budget:
                    break

            self.update_positions(lb, ub)

        return self.alpha['position'], self.best_value