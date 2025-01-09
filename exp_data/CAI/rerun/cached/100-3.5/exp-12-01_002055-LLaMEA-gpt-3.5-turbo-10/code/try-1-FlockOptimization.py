import numpy as np

class FlockOptimization:
    def __init__(self, budget, dim, num_birds=20, alpha=1.0, beta=1.0, gamma=1.0, step_size=0.1):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.step_size = step_size

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        birds = np.random.uniform(lower_bound, upper_bound, (self.num_birds, self.dim))
        best_position = np.random.uniform(lower_bound, upper_bound, self.dim)
        best_fitness = np.inf

        for _ in range(self.budget):
            for i in range(self.num_birds):
                fitness = func(birds[i])
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_position = birds[i]

                random_bird = np.random.randint(self.num_birds)
                birds[i] += self.alpha * (best_position - birds[i]) + self.beta * (birds[random_bird] - birds[i]) + self.gamma * np.random.uniform(-1, 1, self.dim)
                birds[i] += np.random.normal(0, self.step_size, self.dim)
                birds[i] = np.clip(birds[i], lower_bound, upper_bound)

        return best_position