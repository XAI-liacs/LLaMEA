import numpy as np

class HybridDEAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutation(self, population, best_idx):
        indices = np.arange(self.population_size)
        for i in range(self.population_size):
            indices = np.delete(indices, np.where(indices == i))
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
            trial = np.copy(population[i])
            crossover = np.random.rand(self.dim) < self.crossover_rate
            trial[crossover] = mutant[crossover]
            if np.random.rand() < 1.0 / self.dim:
                trial[np.random.randint(0, self.dim)] = mutant[np.random.randint(0, self.dim)]
            yield trial

    def _annealing_acceptance(self, current_value, new_value):
        if new_value < current_value:
            return True
        return np.random.rand() < np.exp((current_value - new_value) / self.temperature)

    def __call__(self, func):
        population = self._initialize_population()
        values = np.array([func(ind) for ind in population])
        best_idx = np.argmin(values)
        best_value = values[best_idx]
        best_solution = population[best_idx]
        evals = self.population_size

        while evals < self.budget:
            for i, trial in enumerate(self._mutation(population, best_idx)):
                if evals >= self.budget:
                    break
                trial_value = func(trial)
                evals += 1
                if self._annealing_acceptance(values[i], trial_value):
                    population[i] = trial
                    values[i] = trial_value
                    if trial_value < best_value:
                        best_value = trial_value
                        best_solution = trial
            
            self.temperature *= self.cooling_rate
            best_idx = np.argmin(values)
            best_value = values[best_idx]

        return best_solution, best_value