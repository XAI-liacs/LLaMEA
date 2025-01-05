import numpy as np

class QGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 30
        self.population = []
        self.phi = np.pi / 4  # Quantum rotation angle

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            individual = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': individual, 'fitness': float('inf')})
        return population

    def quantum_rotation(self, individual, global_best):
        for i in range(self.dim):
            theta = self.phi * (np.random.rand() - 0.5) * 2
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            delta = global_best[i] - individual['position'][i]
            individual['position'][i] += rotation @ np.array([delta, individual['position'][i]])[0]
        individual['position'] = np.clip(individual['position'], self.lb, self.ub)

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        offspring = alpha * parent1 + (1 - alpha) * parent2
        return offspring

    def mutate(self, individual):
        mutation_strength = 0.1
        mutation = (np.random.rand(self.dim) - 0.5) * 2 * mutation_strength
        individual += mutation
        return np.clip(individual, self.lb, self.ub)

    def __call__(self, func):
        self.lb, self.ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(self.lb, self.ub)
        
        while evaluations < self.budget:
            # Evaluate fitness
            for individual in self.population:
                individual['fitness'] = func(individual['position'])
                evaluations += 1
                if individual['fitness'] < self.best_value:
                    self.best_value = individual['fitness']
                    self.best_solution = individual['position'].copy()
                if evaluations >= self.budget:
                    break
            
            # Apply quantum rotation based on global best
            for individual in self.population:
                self.quantum_rotation(individual, self.best_solution)
            
            # Selection, Crossover, and Mutation
            sorted_population = sorted(self.population, key=lambda x: x['fitness'])
            new_population = sorted_population[:self.population_size // 2]  # Select the top half
            
            while len(new_population) < self.population_size:
                parent1, parent2 = np.random.choice(new_population, 2, replace=False)
                offspring_position = self.crossover(parent1['position'], parent2['position'])
                offspring_position = self.mutate(offspring_position)
                new_population.append({'position': offspring_position, 'fitness': float('inf')})
            
            self.population = new_population

        return self.best_solution, self.best_value