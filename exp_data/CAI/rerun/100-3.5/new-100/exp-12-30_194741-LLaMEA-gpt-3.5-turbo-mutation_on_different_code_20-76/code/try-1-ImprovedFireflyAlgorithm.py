class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0, step_size=0.1):
        super().__init__(budget, dim, alpha, beta0, gamma)
        self.step_size = step_size

    def move_fireflies(self, fireflies, light_intensity):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if light_intensity[j] < light_intensity[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = self.attractiveness(light_intensity[j])
                    step = self.step_size / (1 + r**2)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) * step

        return fireflies