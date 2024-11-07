import numpy as np
import random
import copy

class ImprovedLeverageEntropy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0
        self.entropy_history = []
        self.exploitation_rate = 0.2
        self.adaptive_dim = False

    def __call__(self, func):
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0
        self.entropy_history = []

        for _ in range(self.budget):
            # Randomly select a dimension to leverage
            dim = random.randint(0, self.dim - 1)

            # Generate a random point in the search space
            x = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

            # Calculate the entropy of the current point
            entropy = 0.0
            for i in range(self.dim):
                if x[i]!= self.lower_bound and x[i]!= self.upper_bound:
                    entropy += 1 / np.log(2 * np.pi * np.sqrt(1 + (x[i] - self.lower_bound) ** 2))

            # Update the entropy
            self.entropy += entropy
            self.entropy_history.append(self.entropy)

            # Evaluate the function at the current point
            f = func(x)

            # Update the best solution if the current solution is better
            if self.f_best is None or f < self.f_best:
                self.f_best = f
                self.x_best = x

            # If the current solution is close to the best solution, reduce the entropy
            if self.f_best_val - f < 1e-3:
                self.entropy -= entropy / 2

            # Balance exploration and exploitation
            if random.random() < self.exploitation_rate:
                # Randomly select a dimension to leverage
                dim = random.randint(0, self.dim - 1)

                # Generate a random point in the search space
                x = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

                # Calculate the entropy of the current point
                entropy = 0.0
                for i in range(self.dim):
                    if x[i]!= self.lower_bound and x[i]!= self.upper_bound:
                        entropy += 1 / np.log(2 * np.pi * np.sqrt(1 + (x[i] - self.lower_bound) ** 2))

                # Update the entropy
                self.entropy += entropy
                self.entropy_history.append(self.entropy)

            # Adaptive dimensionality
            if _ % (self.budget // 5) == 0 and self.adaptive_dim:
                self.dim *= 2
                self.adaptive_dim = False

        # Reduce the entropy to maintain the balance between exploration and exploitation
        self.entropy = max(0.0, self.entropy - 0.1)

        # Update the best solution if the current solution is better
        if self.f_best_val > self.f_best:
            self.f_best = self.f_best
            self.x_best = self.x_best

        return self.f_best

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
improved_leverage_entropy = ImprovedLeverageEntropy(budget, dim)
for _ in range(100):
    print(improved_leverage_entropy(func))
