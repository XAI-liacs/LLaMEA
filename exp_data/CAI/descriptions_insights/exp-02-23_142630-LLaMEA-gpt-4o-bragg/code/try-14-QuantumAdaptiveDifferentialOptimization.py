import numpy as np

class QuantumAdaptiveDifferentialOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.history = []

    def quantum_bit_flip(self, solution, best):
        q = np.random.rand(self.dim)
        flip = np.random.rand(self.dim) < q
        return np.where(flip, best, solution)

    def differential_mutation(self, population, best):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = population[idxs]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def __call__(self, func):
        self.bounds = func.bounds
        population = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        best_idx = np.argmin(scores)
        best_solution = population[best_idx]

        self.history.extend(scores)

        evaluations = self.population_size
        while evaluations < self.budget:
            new_population = []
            if evaluations > self.budget / 2:  # Dynamic population scaling
                self.population_size = max(10, int(self.population_size * 0.9))
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    mutant = self.differential_mutation(population, best_solution)
                    trial_solution = self.quantum_bit_flip(mutant, best_solution)
                else:
                    trial_solution = population[i]

                trial_score = func(trial_solution)
                evaluations += 1

                if trial_score < scores[i]:
                    new_population.append(trial_solution)
                    scores[i] = trial_score
                else:
                    new_population.append(population[i])

                if trial_score < scores[best_idx]:
                    best_idx = i
                    best_solution = trial_solution

            population = np.array(new_population)
            self.history.extend(scores)

        return best_solution, scores[best_idx], self.history