import numpy as np

class GQIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.f = 0.5  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.phi = np.pi / 4  # Quantum rotation angle
        self.population = []

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'best_value': float('inf')})
        return population

    def quantum_crossover(self, target, mutant, lb, ub):
        for i in range(self.dim):
            r = np.random.rand()
            theta = self.phi if r < 0.5 else -self.phi
            target['position'][i] = target['position'][i] * np.cos(theta) + mutant[i] * np.sin(theta)
            if target['position'][i] < lb[i] or target['position'][i] > ub[i]:
                target['position'][i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()  # Re-initialize if out of bounds

        target['position'] = np.clip(target['position'], lb, ub)

    def mutate(self, idx, lb, ub):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a]['position'] + self.f * (self.population[b]['position'] - self.population[c]['position'])
        return np.clip(mutant, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i, lb, ub)

                trial = {'position': np.copy(target['position']), 'best_value': float('inf')}
                self.quantum_crossover(trial, mutant, lb, ub)

                trial_value = func(trial['position'])
                evaluations += 1

                if trial_value < target['best_value']:
                    self.population[i]['position'] = trial['position']
                    self.population[i]['best_value'] = trial_value

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial['position'].copy()

                if evaluations >= self.budget:
                    break

        return self.best_solution, self.best_value