class ImprovedFireflyOptimization(FireflyOptimization):
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta_min=0.2):
        super().__init__(budget, dim, alpha, beta_min)
        self.gamma = 1.0 / np.sqrt(dim)  # Update gamma dynamically based on dimensionality