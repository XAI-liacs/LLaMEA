import numpy as np

class AdaptiveCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pa = 0.25
        self.alpha = 1.5

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        s = np.random.normal(0, 1, self.dim)
        u = np.random.normal(0, sigma, self.dim)
        v = s / (abs(u) ** (1 / beta))
        step = 0.01 * v
        return step

    def __call__(self, func):
        def generate_cuckoo(nest):
            new_nest = nest + self.levy_flight()
            return np.clip(new_nest, -5.0, 5.0)

        def replace_worst(nests, new_nest):
            idx = np.argmax([func(nest) for nest in nests])
            nests[idx] = new_nest
            return nests

        nests = np.random.uniform(-5.0, 5.0, (self.alpha*10, self.dim))
        nests_fit = np.array([func(nest) for nest in nests])
        best_nest = nests[np.argmin(nests_fit)]

        for _ in range(self.budget):
            idx = np.random.randint(self.alpha*10)
            new_nest = generate_cuckoo(nests[idx])
            new_fit = func(new_nest)
            if new_fit < nests_fit[idx]:
                nests_fit[idx] = new_fit
                nests = replace_worst(nests, new_nest)
                best_nest = new_nest if new_fit < func(best_nest) else best_nest
                
        return best_nest