import numpy as np

class AdaptiveDifferentialEvolutionWithLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.7
        self.history = []
        self.population = None
        self.best_solution = None
        self.best_score = np.inf

    def _initialize_population(self, lb, ub):
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb

    def _mutate(self, i, lb, ub):
        indices = [idx for idx in range(self.population_size) if idx != i]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def _select(self, target_idx, offspring, func):
        target = self.population[target_idx]
        target_score = func(target)
        offspring_score = func(offspring)

        if offspring_score < target_score:
            self.population[target_idx] = offspring
            return offspring_score, True
        return target_score, False

    def _adapt_parameters(self):
        success_ratio = sum(self.history[-self.population_size:]) / self.population_size
        self.mutation_factor = 0.5 + 0.3 * success_ratio
        self.crossover_probability = 0.7 + 0.2 * (1 - success_ratio)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            success_count = 0
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                mutant = self._mutate(i, self.lb, self.ub)
                offspring = self._crossover(self.population[i], mutant)
                score, success = self._select(i, offspring, func)

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = np.copy(self.population[i])

                eval_count += 1
                success_count += success
            
            self.history.append(success_count)
            self._adapt_parameters()

        return self.best_solution, self.best_score