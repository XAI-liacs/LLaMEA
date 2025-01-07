import numpy as np

class AMGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.memory_size = 5
        self.mutation_rate = 0.1
        self.memory = []
        self.best_solution = None
        self.best_value = float('inf')

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            individual = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': individual, 'value': float('inf')})
        return population

    def evaluate(self, population, func):
        for individual in population:
            if 'value' not in individual or np.isinf(individual['value']):
                individual['value'] = func(individual['position'])
        return population

    def select_parents(self, population):
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        return population[idx1] if population[idx1]['value'] < population[idx2]['value'] else population[idx2]

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        child = alpha * parent1['position'] + (1 - alpha) * parent2['position']
        return child

    def mutate(self, individual, lb, ub):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = lb + (ub - lb) * np.random.rand(self.dim)
            individual['position'] += mutation_vector * (np.random.rand(self.dim) - 0.5)
        individual['position'] = np.clip(individual['position'], lb, ub)

    def update_memory(self, individual):
        if len(self.memory) < self.memory_size:
            self.memory.append(individual)
        else:
            worst_index = np.argmax([ind['value'] for ind in self.memory])
            if individual['value'] < self.memory[worst_index]['value']:
                self.memory[worst_index] = individual

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)
        
        while evaluations < self.budget:
            population = self.evaluate(population, func)
            
            # Update global best solution
            for individual in population:
                if individual['value'] < self.best_value:
                    self.best_value = individual['value']
                    self.best_solution = individual['position'].copy()
                    self.update_memory(individual)
            
            # Generate new population
            new_population = []
            while len(new_population) < self.population_size and evaluations < self.budget:
                parent1 = self.select_parents(population)
                parent2 = self.select_parents(population)
                child_position = self.crossover(parent1, parent2)
                child = {'position': child_position, 'value': float('inf')}
                self.mutate(child, lb, ub)
                new_population.append(child)
                evaluations += 1
            
            population = new_population[:self.population_size]
        
        return self.best_solution, self.best_value