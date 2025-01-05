import numpy as np

class HQGA:
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
            individual = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': individual, 'best_position': individual, 'best_value': float('inf')})
        return population

    def quantum_update(self, individual, global_best, lb, ub, beta):
        r1 = np.random.rand(self.dim)
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        
        mean_best = (individual['best_position'] + global_best) / 2
        individual['position'] = mean_best + beta * (r1 - 0.5) * np.abs(global_best - individual['position']) * np.tan(phi) * direction
        individual['position'] = np.clip(individual['position'], lb, ub)

    def genetic_crossover_and_mutation(self, parent1, parent2, lb, ub):
        crossover_point = np.random.randint(0, self.dim)
        child = np.concatenate((parent1['position'][:crossover_point], parent2['position'][crossover_point:]))
        
        # Mutate with small probability
        if np.random.rand() < 0.1:
            mutation_vector = (ub - lb) * (np.random.rand(self.dim) - 0.5) * 0.1
            child += mutation_vector
        
        return np.clip(child, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        global_best = None
        global_best_value = float('inf')
        
        while evaluations < self.budget:
            # Evaluate fitness and update bests
            for individual in self.population:
                value = func(individual['position'])
                evaluations += 1
                
                if value < individual['best_value']:
                    individual['best_value'] = value
                    individual['best_position'] = individual['position'].copy()
                
                if value < global_best_value:
                    global_best_value = value
                    global_best = individual['position'].copy()

                if evaluations >= self.budget:
                    break

            beta = 1.0 - evaluations / self.budget

            # Update individuals with quantum-inspired approach
            for individual in self.population:
                self.quantum_update(individual, global_best, lb, ub, beta)

            # Apply genetic operations
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = np.random.choice(self.population, 2, replace=False)
                child1_position = self.genetic_crossover_and_mutation(parent1, parent2, lb, ub)
                new_population.append({'position': child1_position, 'best_position': child1_position, 'best_value': float('inf')})
                
                child2_position = self.genetic_crossover_and_mutation(parent2, parent1, lb, ub)
                new_population.append({'position': child2_position, 'best_position': child2_position, 'best_value': float('inf')})
            
            self.population = new_population

        return global_best, global_best_value