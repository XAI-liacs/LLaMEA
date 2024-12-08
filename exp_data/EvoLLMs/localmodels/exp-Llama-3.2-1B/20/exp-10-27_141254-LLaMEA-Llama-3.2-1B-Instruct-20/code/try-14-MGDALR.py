import numpy as np
import random

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

class Individual:
    def __init__(self, dim):
        self.dim = dim
        self.fitness = 0
        self.lines = []

    def evaluate_fitness(self, func):
        x = np.array([-5.0] * self.dim)
        for _ in range(self.budget):
            y = func(x)
            self.fitness += y
            x = np.array([x[-1] + random.uniform(-1, 1) for _ in range(self.dim)])
            self.lines.append((x, y))
        return self.fitness

class MetaHeuristic:
    def __init__(self, algorithm, budget, dim):
        self.algorithm = algorithm
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        individual = Individual(self.dim)
        individual.fitness = 0
        individual.lines = []
        return self.algorithm.__call__(func, individual)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 