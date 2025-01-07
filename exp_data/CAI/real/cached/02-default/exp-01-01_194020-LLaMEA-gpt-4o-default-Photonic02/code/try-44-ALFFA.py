import numpy as np

class ALFFA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 30
        self.population = []
        self.alpha = 1.0
        self.beta0 = 1.0
        self.gamma = 1.0

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'value': float('inf')})
        return population

    def levy_flight(self, step):
        # Levy flight step calculation using Mantegna's algorithm
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=step.shape)
        v = np.random.normal(0, 1, size=step.shape)
        return u / (abs(v) ** (1 / beta))

    def update_firefly(self, firefly_i, firefly_j, lb, ub):
        distance = np.linalg.norm(firefly_i['position'] - firefly_j['position'])
        beta = self.beta0 * np.exp(-self.gamma * distance ** 2)
        attraction = beta * (firefly_j['position'] - firefly_i['position'])
        random_walk = self.alpha * self.levy_flight(firefly_i['position'])
        firefly_i['position'] += attraction + random_walk
        firefly_i['position'] = np.clip(firefly_i['position'], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        
        while evaluations < self.budget:
            # Evaluate all fireflies
            for firefly in self.population:
                firefly['value'] = func(firefly['position'])
                evaluations += 1
                
                if firefly['value'] < self.best_value:
                    self.best_value = firefly['value']
                    self.best_solution = firefly['position'].copy()

                if evaluations >= self.budget:
                    break
            
            # Update fireflies based on pairwise attraction
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self.population[j]['value'] < self.population[i]['value']:
                        self.update_firefly(self.population[i], self.population[j], lb, ub)
        
        return self.best_solution, self.best_value