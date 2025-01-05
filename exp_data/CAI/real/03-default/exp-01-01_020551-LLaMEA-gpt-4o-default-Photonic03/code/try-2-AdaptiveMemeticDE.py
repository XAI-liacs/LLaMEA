import numpy as np

class AdaptiveMemeticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.f = 0.5     # Initial differential weight
        self.cr = 0.9    # Crossover probability
        self.mutation_strategies = ['rand', 'best']
        self.strategy_weights = np.array([0.5, 0.5])  # Start with equal strategy weights

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            new_scores = np.empty(self.population_size)

            for i in range(self.population_size):
                strategy_idx = np.random.choice(len(self.mutation_strategies), p=self.strategy_weights)
                if self.mutation_strategies[strategy_idx] == 'rand':
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    mutant = np.clip(x0 + self.f * (x1 - x2), lb, ub)
                else:  # 'best' strategy
                    best_idx = np.argmin(scores)
                    x1, x2 = population[np.random.choice(self.population_size, 2, replace=False)]
                    mutant = np.clip(population[best_idx] + self.f * (x1 - x2), lb, ub)

                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])
                # Local search (e.g., simple gradient step)
                if np.random.rand() < 0.1 and evaluations < self.budget:
                    gradient = np.random.uniform(-0.01, 0.01, self.dim)
                    trial = np.clip(trial + gradient, lb, ub)
                    evaluations += 1

                trial_score = func(trial)
                evaluations += 1

                if trial_score < scores[i]:
                    new_population[i] = trial
                    new_scores[i] = trial_score
                    self.strategy_weights[strategy_idx] += 0.1  # Boost successful strategy
                else:
                    new_population[i] = population[i]
                    new_scores[i] = scores[i]

            # Normalize strategy weights
            self.strategy_weights /= self.strategy_weights.sum()
            population, scores = new_population, new_scores

        best_index = np.argmin(scores)
        return population[best_index], scores[best_index]