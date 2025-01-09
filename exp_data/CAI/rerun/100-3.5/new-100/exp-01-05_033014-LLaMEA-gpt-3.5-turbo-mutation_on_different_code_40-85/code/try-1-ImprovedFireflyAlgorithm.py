class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    beta = self.beta0 * np.exp(-self.gamma * np.linalg.norm(fireflies[j] - fireflies[i])**2)
                    rand_move = np.random.uniform(-1, 1, self.dim)
                    fireflies[i] += self.alpha * (fireflies[j] - fireflies[i]) * beta + rand_move
                    fireflies[i] = np.clip(fireflies[i], func.bounds.lb, func.bounds.ub)
        return fireflies