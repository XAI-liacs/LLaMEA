import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, self.budget // 2)
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_value = float('inf')

    def mutate(self, idx, F):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        x1, x2, x3 = self.population[np.random.choice(candidates, 3, replace=False)]
        mutant = x1 + F * (x2 - x3)
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant, cr=0.7):
        jrand = np.random.randint(self.dim)
        trial = np.array([mutant[j] if np.random.rand() < cr or j == jrand else target[j] for j in range(self.dim)])
        return trial

    def simulated_annealing_acceptance(self, current_value, candidate_value, temperature):
        if candidate_value < current_value:
            return True
        else:
            return np.random.rand() < np.exp((current_value - candidate_value) / temperature)

    def __call__(self, func):
        evaluations = 0
        temperature = 1.0
        cooling_rate = 0.99

        while evaluations < self.budget:
            F = 0.5 + 0.3 * np.cos(2 * np.pi * evaluations / self.budget)  # Dynamic mutation factor
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self.mutate(i, F)
                trial = self.crossover(target, mutant)

                trial_value = func(trial)
                evaluations += 1

                if trial_value < self.best_value:
                    self.best_solution = trial
                    self.best_value = trial_value

                if self.simulated_annealing_acceptance(func(target), trial_value, temperature):
                    self.population[i] = trial

            temperature *= cooling_rate

        return self.best_solution