import numpy as np

class ADEQT:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'value': float('inf')})
        return population

    def mutate(self, target_idx, population, lb, ub):
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = population[np.random.choice(indices)], population[np.random.choice(indices)], population[np.random.choice(indices)]
        mutant = a['position'] + self.mutation_factor * (b['position'] - c['position'])
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(crossover_mask, mutant, target['position'])
        return trial

    def quantum_tunneling(self, best_position, lb, ub):
        shift = (ub - lb) * (np.random.rand(self.dim) - 0.5) * 0.01
        tunneled_position = best_position + shift
        return np.clip(tunneled_position, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)
        
        while evaluations < self.budget:
            for idx, target in enumerate(population):
                mutant = self.mutate(idx, population, lb, ub)
                trial = self.crossover(target, mutant)
                trial_value = func(trial)
                evaluations += 1

                if trial_value < target['value']:
                    target['position'] = trial
                    target['value'] = trial_value

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget:
                tunneled_position = self.quantum_tunneling(self.best_solution, lb, ub)
                tunneled_value = func(tunneled_position)
                evaluations += 1

                if tunneled_value < self.best_value:
                    self.best_value = tunneled_value
                    self.best_solution = tunneled_position

        return self.best_solution, self.best_value