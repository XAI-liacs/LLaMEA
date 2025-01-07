import numpy as np

class QuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.population = None
        self.best_solution = None
        self.best_score = np.inf

    def _initialize_population(self, lb, ub):
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb

    def _quantum_operator(self, target, trial):
        return target + np.random.uniform(-0.1, 0.1, self.dim) * (trial - target)
    
    def _mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        return self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                mutant = self._mutate(i)
                trial = np.copy(self.population[i])
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                trial[cross_points] = mutant[cross_points]
                trial = np.clip(trial, self.lb, self.ub)

                trial_quantum = self._quantum_operator(self.population[i], trial)
                score = func(trial_quantum)
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = trial_quantum

                if score < func(self.population[i]):
                    new_population[i] = trial_quantum

            self.population = new_population

        return self.best_solution, self.best_score