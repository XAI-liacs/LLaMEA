class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 1.0 / np.sqrt(1.0 + np.arange(self.budget))  # Dynamic alpha adaptation
        self.beta0 = 1.0
        self.gamma = 0.2
        self.f_opt = np.Inf
        self.x_opt = None