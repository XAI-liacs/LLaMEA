class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0, step_size=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.step_size = step_size
        self.f_opt = np.Inf
        self.x_opt = None

    def move_firefly(self, x, attractiveness, x_other):
        r = np.linalg.norm(x - x_other)
        beta = self.attractiveness(r)
        return x + self.step_size * attractiveness * (x_other - x) + self.alpha * (np.random.rand(self.dim) - 0.5)