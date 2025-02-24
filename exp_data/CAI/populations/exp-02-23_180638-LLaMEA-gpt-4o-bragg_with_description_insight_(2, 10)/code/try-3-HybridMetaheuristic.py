import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.population = None
        self.lb = None
        self.ub = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('-inf')

    def quasi_oppositional_init(self):
        midpoint = (self.lb + self.ub) / 2
        self.population = midpoint + np.random.rand(self.population_size, self.dim) * (self.ub - self.lb) / 2
        opposite_population = midpoint - (self.population - midpoint)
        self.population = np.vstack((self.population, opposite_population))
        self.population = np.clip(self.population, self.lb, self.ub)

    def evaluate_population(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()

    def differential_evolution_step(self, func):  # Changed line
        for i in range(self.population_size):
            idxs = [idx for idx in range(2 * self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + 0.8 * (b - c), self.lb, self.ub)
            trial_vector = np.where(np.random.rand(self.dim) < 0.9, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            if trial_fitness > self.fitness[i]:
                self.population[i] = trial_vector
                self.fitness[i] = trial_fitness
                if trial_fitness > self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector.copy()

    def periodicity_enforcement(self):
        for i in range(self.population_size):
            period = np.random.randint(1, self.dim // 2)
            for j in range(0, self.dim, period):
                self.population[i][j:j+period] = np.mean(self.population[i][j:j+period])

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.quasi_oppositional_init()
        self.evaluate_population(func)
        evaluations = 2 * self.population_size

        while evaluations < self.budget:
            self.differential_evolution_step(func)
            self.periodicity_enforcement()
            self.evaluate_population(func)
            evaluations += self.population_size

        return self.best_solution