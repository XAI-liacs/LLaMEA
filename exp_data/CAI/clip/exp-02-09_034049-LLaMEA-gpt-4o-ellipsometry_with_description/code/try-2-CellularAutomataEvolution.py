import numpy as np

class CellularAutomataEvolution:
    def __init__(self, budget, dim, grid_size=10):
        self.budget = budget
        self.dim = dim
        self.grid_size = grid_size
        self.population = np.random.uniform(size=(grid_size, grid_size, dim))
        self.scores = np.full((grid_size, grid_size), np.inf)
        self.best_position = np.zeros(dim)
        self.best_score = np.inf
        self.neighborhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Von Neumann neighborhood

    def optimize(self, func):
        num_evaluations = 0
        
        while num_evaluations < self.budget:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Ensure positions are within bounds
                    self.population[i, j] = np.clip(self.population[i, j], func.bounds.lb, func.bounds.ub)
                    
                    # Evaluate the function
                    score = func(self.population[i, j])
                    num_evaluations += 1
                    self.scores[i, j] = score
                    
                    # Update global best
                    if score < self.best_score:
                        self.best_score = score
                        self.best_position = self.population[i, j].copy()
                    
                    # Cellular automata-based local competition
                    neighbors = [(i + dx, j + dy) for dx, dy in self.neighborhood]
                    valid_neighbors = [(x % self.grid_size, y % self.grid_size) for x, y in neighbors]
                    for x, y in valid_neighbors:
                        if self.scores[x, y] < self.scores[i, j]:
                            # Evolve to neighbor's position with a mutation
                            mutation = np.random.normal(0, 0.1, size=self.dim)
                            self.population[i, j] = self.population[x, y] + mutation

    def __call__(self, func):
        self.optimize(func)
        return self.best_position, self.best_score