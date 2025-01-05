import numpy as np

class AQDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.scaling_factor = 0.8
        self.cr_rate = 0.9
        self.population = []

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def quantum_update(self, target, best, lb, ub):
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        return best + (target - best) * np.tan(phi) * direction

    def differential_mutation(self, idx, lb, ub):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.scaling_factor * (self.population[b] - self.population[c])
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        trial = np.array([mutant[i] if np.random.rand() < self.cr_rate else target[i] for i in range(self.dim)])
        return trial

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        self.population = self.initialize_population(lb, ub)
        evaluations = 0
        best_value = float('inf')
        best_solution = None

        while evaluations < self.budget:
            new_population = []
            for idx, target in enumerate(self.population):
                mutant = self.differential_mutation(idx, lb, ub)
                trial = self.crossover(target, mutant)
                
                trial_value = func(trial)
                evaluations += 1

                if trial_value < best_value:
                    best_value = trial_value
                    best_solution = trial.copy()

                if trial_value < func(target):
                    new_population.append(trial)
                else:
                    new_population.append(target)

                if evaluations >= self.budget:
                    break

            self.population = new_population
            # Quantum-inspired update
            for idx, target in enumerate(self.population):
                self.population[idx] = self.quantum_update(target, best_solution, lb, ub)
                self.population[idx] = np.clip(self.population[idx], lb, ub)

        return best_solution, best_value