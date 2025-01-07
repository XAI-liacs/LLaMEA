import numpy as np

class BioSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.population = []

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'best_position': position, 'best_value': float('inf')})
        return population

    def crossover(self, parent1, parent2):
        alpha = np.random.uniform(0, 1, self.dim)
        child = alpha * parent1['position'] + (1 - alpha) * parent2['position']
        return np.clip(child, lb, ub)

    def mutate(self, individual, mutation_rate=0.1):
        if np.random.rand() < mutation_rate:
            mutation_vector = np.random.normal(0, 1, self.dim)
            individual['position'] += mutation_vector
            individual['position'] = np.clip(individual['position'], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            new_population = []

            for i in range(0, self.population_size, 2):
                parent1 = self.population[i]
                parent2 = self.population[(i+1) % self.population_size]

                child1_position = self.crossover(parent1, parent2)
                child2_position = self.crossover(parent2, parent1)

                new_population.append({'position': child1_position, 'best_position': child1_position, 'best_value': float('inf')})
                new_population.append({'position': child2_position, 'best_position': child2_position, 'best_value': float('inf')})

            # Mutation
            for individual in new_population:
                self.mutate(individual)

            # Evaluate new population
            for individual in new_population:
                value = func(individual['position'])
                evaluations += 1
                
                if value < individual['best_value']:
                    individual['best_value'] = value
                    individual['best_position'] = individual['position'].copy()

                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = individual['position'].copy()

                if evaluations >= self.budget:
                    break

            # Set new population for the next generation
            self.population = new_population

        return self.best_solution, self.best_value