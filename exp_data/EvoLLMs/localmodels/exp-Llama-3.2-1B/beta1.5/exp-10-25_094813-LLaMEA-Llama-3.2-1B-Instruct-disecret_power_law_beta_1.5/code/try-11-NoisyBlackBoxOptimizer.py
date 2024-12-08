import numpy as np
import matplotlib.pyplot as plt

class NoisyBlackBoxOptimizer:
    def __init__(self, budget, dim, max_iter=1000):
        self.budget = budget
        self.dim = dim
        self.max_iter = max_iter
        self.explore_eviction = False
        self.current_dim = 0
        self.func = None

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            return func
        else:
            # Hierarchical clustering-based gradient descent for efficient exploration-ejection
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.explore_eviction = False
                else:
                    # Perform hierarchical clustering-based gradient descent
                    if self.current_dim == 0:
                        # Hierarchical clustering-based gradient descent without hierarchical clustering
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
                    else:
                        # Hierarchical clustering-based gradient descent
                        cluster_labels = np.argpartition(func, self.current_dim)[-1]
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]])
                        self.current_dim += 1
                        if self.budget == 0:
                            break
                self.budget -= 1
            return self.func

    def func(self, x):
        return np.array([func(x) for func in self.func])

def noisy_black_box_optimization():
    # Initialize the NoisyBlackBoxOptimizer with a budget of 1000 evaluations
    optimizer = NoisyBlackBoxOptimizer(1000, 10)
    
    # Evaluate the BBOB test suite 24 times
    for _ in range(24):
        func = np.array([np.random.uniform(-5.0, 5.0, 10) for _ in range(10)])
        optimizer.func(func)
    
    # Print the final best function and its score
    print("Final Best Function:", optimizer.func[np.argmax(optimizer.func)])
    print("Score:", np.max(optimizer.func))

noisy_black_box_optimization()