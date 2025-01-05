import numpy as np

class MQGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 30
        self.population = []
        self.phi = np.pi / 4  # Quantum rotation angle
        self.local_search_prob = 0.3
    
    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            fitness = float('inf')
            population.append({'position': position, 'fitness': fitness})
        return population
    
    def quantum_crossover(self, parent1, parent2, lb, ub):
        child_position = np.zeros(self.dim)
        for i in range(self.dim):
            r = np.random.rand()
            theta = self.phi if r < 0.5 else -self.phi
            child_position[i] = (parent1['position'][i] + parent2['position'][i]) / 2 + \
                                np.sin(theta) * (parent1['position'][i] - parent2['position'][i]) / 2
            if child_position[i] < lb[i] or child_position[i] > ub[i]:
                child_position[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()  # Re-initialize if out of bounds
        
        child_position = np.clip(child_position, lb, ub)
        return {'position': child_position, 'fitness': float('inf')}
    
    def local_search(self, individual, func, lb, ub):
        position = individual['position'].copy()
        for i in range(self.dim):
            perturbation = 0.1 * (ub[i] - lb[i]) * (np.random.rand() - 0.5)
            position[i] += perturbation
        position = np.clip(position, lb, ub)
        fitness = func(position)
        if fitness < individual['fitness']:
            individual['position'] = position
            individual['fitness'] = fitness
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        
        while evaluations < self.budget:
            # Evaluate fitness
            for individual in self.population:
                if individual['fitness'] == float('inf'):
                    fitness = func(individual['position'])
                    evaluations += 1
                    individual['fitness'] = fitness
                    if fitness < self.best_value:
                        self.best_value = fitness
                        self.best_solution = individual['position'].copy()
                    if evaluations >= self.budget:
                        break
            
            # Create new population
            new_population = []
            np.random.shuffle(self.population)
            for i in range(0, self.population_size, 2):
                if i + 1 >= self.population_size:
                    break
                parent1, parent2 = self.population[i], self.population[i + 1]
                child = self.quantum_crossover(parent1, parent2, lb, ub)
                new_population.append(child)
            
            # Apply local search
            for individual in new_population:
                if np.random.rand() < self.local_search_prob:
                    self.local_search(individual, func, lb, ub)
            
            # Replace old population with new population
            self.population = new_population
        
        return self.best_solution, self.best_value