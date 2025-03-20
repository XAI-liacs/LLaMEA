import numpy as np

class AdaptiveMomentumFirefly:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.5  # Initial randomization parameter
        self.gamma = 1    # Light absorption coefficient
        self.beta_min = 0.2  # Minimum attractiveness
        self.momentum = 0.9  # Momentum coefficient

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize fireflies and their velocities
        fireflies = np.random.uniform(lb, ub, size=(self.budget, self.dim))
        velocities = np.zeros_like(fireflies)
        intensities = np.array([func(ff) for ff in fireflies])

        best_idx = np.argmin(intensities)
        best_firefly = fireflies[best_idx]

        evaluations = len(fireflies)

        while evaluations < self.budget:
            for i in range(len(fireflies)):
                for j in range(len(fireflies)):
                    if intensities[j] < intensities[i]:
                        distance = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * distance**2)

                        attraction = beta * (fireflies[j] - fireflies[i])
                        random_step = self.alpha * (np.random.rand(self.dim) - 0.5)
                        velocities[i] = self.momentum * velocities[i] + attraction + random_step
                        fireflies[i] += velocities[i]
                        
                        # Ensure the fireflies remain within bounds
                        fireflies[i] = np.clip(fireflies[i], lb, ub)

                # Evaluate the new position of firefly i
                new_intensity = func(fireflies[i])
                evaluations += 1

                # Update the intensity if improvement is found
                if new_intensity < intensities[i]:
                    intensities[i] = new_intensity

                # Update the global best firefly
                if new_intensity < intensities[best_idx]:
                    best_idx = i
                    best_firefly = fireflies[i]
                    self.momentum = 0.95  # Increase momentum for better convergence

                # Break if we reach the budget
                if evaluations >= self.budget:
                    break

            # Gradually decrease alpha
            self.alpha *= 0.95

        return best_firefly