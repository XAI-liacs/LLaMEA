import numpy as np

class ACSGI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.sigma = 0.3  # Initial step size
        self.learning_rate = 0.1  # Learning rate for covariance adaptation

    def __call__(self, func):
        # Initialize variables
        x_best = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        f_best = func(x_best)
        eval_count = 1
        
        # Covariance matrix initialized to identity
        cov_matrix = np.eye(self.dim)
        
        while eval_count < self.budget:
            # Sample a set of candidate solutions
            num_samples = min(10, self.budget - eval_count)  # Adjust number of samples based on remaining budget
            samples = np.random.multivariate_normal(x_best, self.sigma**2 * cov_matrix, num_samples)
            samples = np.clip(samples, self.lower_bound, self.upper_bound)
            
            # Evaluate all samples
            f_values = np.array([func(x) for x in samples])
            eval_count += num_samples
            
            # Select the best sample
            idx_best = np.argmin(f_values)
            if f_values[idx_best] < f_best:
                x_best = samples[idx_best]
                f_best = f_values[idx_best]

            # Calculate approximate gradient
            gradient = np.mean((samples - x_best) * (f_values[:, np.newaxis] - f_best), axis=0)

            # Update covariance matrix
            cov_matrix = (1 - self.learning_rate) * cov_matrix + self.learning_rate * np.outer(gradient, gradient)
            
            # Adjust step size based on performance
            self.sigma *= 0.85 if f_values[idx_best] >= f_best else 1.05

        return x_best, f_best