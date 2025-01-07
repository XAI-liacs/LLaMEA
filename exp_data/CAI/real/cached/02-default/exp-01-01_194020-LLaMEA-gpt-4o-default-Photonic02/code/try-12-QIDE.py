import numpy as np

class QIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.populations = []
        self.phi = np.pi / 4  # Quantum rotation angle
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'best_position': position, 'best_value': float('inf')})
        return population

    def quantum_mutation(self, target_idx, lb, ub):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.populations[a]['position'] + self.F * (self.populations[b]['position'] - self.populations[c]['position'])
        mutant = np.clip(mutant, lb, ub)
        return mutant

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < self.CR or i == np.random.randint(self.dim):
                trial[i] = mutant[i]
        return trial

    def quantum_update(self, position, global_best, lb, ub):
        new_position = np.copy(position)
        for i in range(self.dim):
            r = np.random.rand()
            theta = self.phi if r < 0.5 else -self.phi
            new_position[i] = new_position[i] * np.cos(theta) + (global_best[i] - new_position[i]) * np.sin(theta)
            new_position[i] = np.clip(new_position[i], lb[i], ub[i])
        return new_position

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.populations = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            for idx, individual in enumerate(self.populations):
                mutant = self.quantum_mutation(idx, lb, ub)
                trial = self.crossover(individual['position'], mutant)
                trial = self.quantum_update(trial, self.best_solution if self.best_solution is not None else lb + (ub - lb) * 0.5, lb, ub)
                
                value = func(trial)
                evaluations += 1

                if value < individual['best_value']:
                    individual['best_value'] = value
                    individual['best_position'] = trial
                
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = trial

                if evaluations >= self.budget:
                    break
        
        return self.best_solution, self.best_value