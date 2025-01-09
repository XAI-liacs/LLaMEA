import numpy as np

class ABC_DNS_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.limit = 0.6
        self.max_trials = 100
        self.neighborhood_size = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def __call__(self, func):
        def random_solution():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        
        def levy_flight():
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.abs(v) ** (1 / beta)
            return step
        
        def fitness(solution):
            return func(solution)
        
        best_solution = random_solution()
        best_fitness = fitness(best_solution)
        
        population = [random_solution() for _ in range(self.pop_size)]
        trials = np.zeros(self.pop_size)
        
        for _ in range(self.budget):
            for i, solution in enumerate(population):
                new_solution = solution + levy_flight()
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                new_fitness = fitness(new_solution)
                
                if new_fitness < fitness(solution):
                    population[i] = new_solution
                    trials[i] = 0
                    if new_fitness < best_fitness:
                        best_solution = new_solution
                        best_fitness = new_fitness
                else:
                    trials[i] += 1
                    if trials[i] >= self.max_trials:
                        trials[i] = 0
                        population[i] = random_solution()
            
            # Dynamic neighborhood search
            for i in range(self.pop_size):
                neighborhood = np.random.choice(population, self.neighborhood_size, replace=False)
                best_neighbor = min(neighborhood, key=lambda x: fitness(x))
                if fitness(best_neighbor) < fitness(population[i]):
                    population[i] = best_neighbor
            
        return best_solution