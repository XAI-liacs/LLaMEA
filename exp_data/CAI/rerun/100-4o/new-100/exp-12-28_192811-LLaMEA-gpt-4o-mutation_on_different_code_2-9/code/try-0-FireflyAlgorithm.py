import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.2  # randomness parameter
        self.beta_min = 0.2  # minimum attractiveness
        self.gamma = 1.0  # absorption coefficient
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta_min * np.exp(-self.gamma * r**2)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        n_fireflies = int(np.sqrt(self.budget))  # number of fireflies
        fireflies = np.random.uniform(lb, ub, (n_fireflies, self.dim))
        intensities = np.array([func(x) for x in fireflies])

        evaluations = n_fireflies
        while evaluations < self.budget:
            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    if intensities[j] < intensities[i]:
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.attractiveness(r)
                        fireflies[i] += beta * (fireflies[j] - fireflies[i])
                        fireflies[i] += self.alpha * (np.random.rand(self.dim) - 0.5)
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        new_intensity = func(fireflies[i])
                        evaluations += 1
                        if new_intensity < intensities[i]:
                            intensities[i] = new_intensity
                            if new_intensity < self.f_opt:
                                self.f_opt = new_intensity
                                self.x_opt = fireflies[i]
                        if evaluations >= self.budget:
                            break
                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt