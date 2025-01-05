import numpy as np

class ReinforcementLearningDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_strategies = [0.5, 0.7, 1.0]
        self.strategy_probs = np.ones(len(self.mutation_strategies)) / len(self.mutation_strategies)
        self.cross_prob = 0.7
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(self.population_size)

            for i in range(self.population_size):
                # Select mutation strategy using RL policy-based probabilities
                strategy_idx = np.random.choice(len(self.mutation_strategies), p=self.strategy_probs)
                F = self.mutation_strategies[strategy_idx]

                # Differential Evolution mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[indices]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.cross_prob
                trial = np.where(crossover_mask, mutant, pop[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = trial_fitness
                    # Reward the chosen strategy
                    self.strategy_probs[strategy_idx] += 0.1
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]

            # Normalize strategy probabilities
            self.strategy_probs /= np.sum(self.strategy_probs)

            # Update population and fitness
            pop, fitness = new_pop, new_fitness

            # Update global best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(best_global):
                best_global = pop[best_idx].copy()

            # Save the history of best solutions
            self.history.append(best_global)

        return best_global