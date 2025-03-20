import numpy as np

class AdaptiveMomentumFirefly:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.5  # Initial randomization parameter
        self.gamma = 0.5  # Light absorption coefficient, adjusted for better exploration
        self.beta_min = 0.2  # Minimum attractiveness
        self.momentum = 0.9  # Momentum coefficient

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize fireflies closer to lb
        fireflies = np.random.uniform(lb, (lb + ub) / 2, size=(self.budget, self.dim))
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
                        beta = (self.beta_min + distance * (1 - self.beta_min)) * np.exp(-0.8 * self.gamma * distance**2 * (1 - evaluations/self.budget))  # Implement adaptive beta scaling

                        attraction = beta * (fireflies[j] - fireflies[i])
                        random_step = self.alpha * 1.05 * (np.random.rand(self.dim) - 0.5)  # Slightly increase randomization factor
                        velocities[i] = self.momentum * velocities[i] + attraction + random_step
                        fireflies[i] += velocities[i]
                        
                        # Ensure the fireflies remain within bounds
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        fireflies[i] = np.where(fireflies[i] < lb, lb + (lb - fireflies[i]), fireflies[i])  # Reflection

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
                    self.momentum = 0.95 + 0.1 * (1 - new_intensity)  # Slightly refine momentum dynamics

                # Break if we reach the budget
                if evaluations >= self.budget:
                    break

            # Gradually decrease alpha
            self.alpha *= 0.97 + 0.03 * (evaluations / self.budget)  # Slightly adapt alpha faster

        return best_firefly