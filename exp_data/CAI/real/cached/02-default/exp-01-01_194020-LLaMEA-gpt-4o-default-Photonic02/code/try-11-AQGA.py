import numpy as np

class AQGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 30
        self.genes = []
        self.phi = np.pi / 6  # Quantum rotation angle
        self.mutation_rate = 0.1

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'fitness': float('inf')})
        return population

    def quantum_crossover(self, parent1, parent2, lb, ub):
        child_position = np.empty(self.dim)
        for i in range(self.dim):
            r = np.random.rand()
            theta = self.phi if r < 0.5 else -self.phi
            child_position[i] = parent1['position'][i] * np.cos(theta) + parent2['position'][i] * np.sin(theta)
            if child_position[i] < lb[i] or child_position[i] > ub[i]:
                child_position[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
        return np.clip(child_position, lb, ub)

    def mutate(self, individual, lb, ub):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual['position'][i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
        individual['position'] = np.clip(individual['position'], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.genes = self.initialize_population(lb, ub)
        
        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size // 2):
                parent1, parent2 = np.random.choice(self.genes, 2, replace=False)
                child1_position = self.quantum_crossover(parent1, parent2, lb, ub)
                child2_position = self.quantum_crossover(parent2, parent1, lb, ub)

                for child_position in [child1_position, child2_position]:
                    child = {'position': child_position, 'fitness': func(child_position)}
                    evaluations += 1

                    if child['fitness'] < self.best_value:
                        self.best_value = child['fitness']
                        self.best_solution = child['position'].copy()

                    self.mutate(child, lb, ub)
                    new_population.append(child)

                    if evaluations >= self.budget:
                        break
                if evaluations >= self.budget:
                    break

            self.genes = sorted(new_population, key=lambda x: x['fitness'])[:self.population_size]

        return self.best_solution, self.best_value