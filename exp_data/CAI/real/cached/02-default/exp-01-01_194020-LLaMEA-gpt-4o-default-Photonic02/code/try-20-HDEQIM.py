import numpy as np

class HDEQIM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.population = []
        self.F = 0.7  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.phi = np.pi / 4  # Quantum mutation angle
        self.best_solution = None
        self.best_value = float('inf')

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.pop_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'value': float('inf')})
        return population

    def quantum_mutation(self, position, best_position):
        mutated_position = position.copy()
        for i in range(self.dim):
            r = np.random.rand()
            theta = self.phi if r < 0.5 else -self.phi
            mutated_position[i] = position[i] * np.cos(theta) + (best_position[i] - position[i]) * np.sin(theta)
        return mutated_position

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            new_population = []
            for target in self.population:
                indices = np.random.choice(range(self.pop_size), 3, replace=False)
                a, b, c = [self.population[idx]['position'] for idx in indices]

                # Differential Evolution Mutation
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                trial = np.copy(target['position'])

                # Crossover
                for i in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[i] = mutant[i]

                # Quantum Mutation
                trial = self.quantum_mutation(trial, self.best_solution if self.best_solution is not None else lb + (ub - lb) / 2)

                # Selection
                trial_value = func(trial)
                evaluations += 1

                if trial_value < target['value']:
                    new_population.append({'position': trial, 'value': trial_value})

                    if trial_value < self.best_value:
                        self.best_value = trial_value
                        self.best_solution = trial
                else:
                    new_population.append(target)

                if evaluations >= self.budget:
                    break

            self.population = new_population

        return self.best_solution, self.best_value