import numpy as np

class AGE_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        # PSO parameters
        self.population_size = 20
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.w = 0.7   # Inertia weight
        self.gradient_learning_rate = 0.1
        
        self.X = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.V = np.zeros((self.population_size, self.dim))
        self.p_best = self.X.copy()
        self.p_best_values = np.full(self.population_size, np.inf)
        self.g_best = None
        self.g_best_value = np.inf
        self.evaluations = 0
        
    def gradient_estimation(self, func, x):
        epsilon = 1e-8
        grad = np.zeros_like(x)
        fx = func(x)
        for i in range(len(x)):
            x_temp = x.copy()
            x_temp[i] += epsilon
            grad[i] = (func(x_temp) - fx) / epsilon
        return grad
    
    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.X[i])
                self.evaluations += 1
                
                if fitness < self.p_best_values[i]:
                    self.p_best_values[i] = fitness
                    self.p_best[i] = self.X[i]
                
                if fitness < self.g_best_value:
                    self.g_best_value = fitness
                    self.g_best = self.X[i]
                
            for i in range(self.population_size):
                gradient = self.gradient_estimation(func, self.X[i])
                self.V[i] = (self.w * self.V[i] +
                             self.c1 * np.random.rand(self.dim) * (self.p_best[i] - self.X[i]) +
                             self.c2 * np.random.rand(self.dim) * (self.g_best - self.X[i]) -
                             self.gradient_learning_rate * gradient)
                
                self.X[i] = np.clip(self.X[i] + self.V[i], self.lower_bound, self.upper_bound)
                
        return self.g_best