import numpy as np

class DynamicStepSizeACS(AdaptiveCuckooSearch):
    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta
        s = np.random.normal(0, 1, self.dim)
        u = np.random.normal(0, sigma, self.dim)
        v = s / (abs(u) ** (1 / beta))
        step = 0.01 * v
        step_size = 0.1 + np.random.rand() * 0.9  # Dynamic step size adaptation
        return step * step_size