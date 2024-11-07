import numpy as np
from scipy.optimize import differential_evolution
import random
import matplotlib.pyplot as plt

class CrowdSourcedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0])  # Search space between -5.0 and 5.0
        self.mean = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)  # Initialize mean
        self.covariance = np.eye(self.dim) * 1.0  # Initialize covariance matrix
        self.covariance_update_rate = 0.5  # Update covariance at 50% of the budget
        self.population_size = 20  # Increase population size for better exploration
        self.adaptive_covariance = True  # Use adaptive covariance update
        self.differential_evolution_seed = 42  # Seed for differential evolution
        self.learning_rate = 0.1  # Adaptive learning rate

    def __call__(self, func):
        # Initialize a list to store the convergence history
        convergence_history = []

        for i in range(int(self.budget * self.covariance_update_rate)):
            # Perform evolution strategy to update mean
            new_mean = self.mean + np.random.normal(0, 1.0, size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space
            new_mean = self.sample_strategy(new_mean)  # Apply sampling strategy

            # Calculate the learning rate based on the current convergence
            learning_rate = max(0.01, 0.1 - (i / (self.budget * self.covariance_update_rate)))

            # Perform genetic drift to update covariance
            new_covariance = self.covariance + learning_rate * np.random.normal(0, 0.1, size=(self.dim, self.dim))
            new_covariance = np.clip(new_covariance, 0, 1.0)  # Clip values to avoid negative covariance matrix

            # Evaluate function at new mean
            f_new = func(new_mean)

            # Update mean and covariance
            self.mean = new_mean
            self.covariance = new_covariance

            # Store the current convergence history
            convergence_history.append(f_new)

        # Perform final optimization using differential evolution
        bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=self.mean, seed=self.differential_evolution_seed)
        print(f"Final best solution: x = {res.x}, f(x) = {res.fun}")

        # Plot the convergence history
        plt.plot(convergence_history)
        plt.xlabel("Iteration")
        plt.ylabel("Convergence")
        plt.title("Convergence History")
        plt.show()

    def sample_strategy(self, mean):
        # Apply adaptive sampling strategy
        if self.adaptive_covariance:
            # Calculate the variance of the mean
            variance = np.var(mean)
            # Sample from a normal distribution with mean and variance
            new_mean = mean + np.random.normal(0, variance, size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space
        else:
            # Randomly sample from the mean
            new_mean = mean + np.random.normal(0, 1.0, size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space
        return new_mean

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

crowd_sourced = CrowdSourcedMetaheuristic(budget=100, dim=2)
crowd_sourced(func)