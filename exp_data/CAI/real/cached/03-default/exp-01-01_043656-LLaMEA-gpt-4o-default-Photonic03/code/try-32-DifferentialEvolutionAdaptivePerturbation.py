import numpy as np

class DifferentialEvolutionAdaptivePerturbation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.mutation_factor_initial = 0.8
        self.mutation_factor_final = 0.2
        self.crossover_probability = 0.9

    def adaptive_mutation_factor(self, eval_count):
        lambda_factor = eval_count / self.budget
        return self.mutation_factor_initial * (1 - lambda_factor) + self.mutation_factor_final * lambda_factor

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Select three random, distinct indices
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[indices]
                mutation_factor = self.adaptive_mutation_factor(eval_count)

                # Differential mutation
                mutant_vector = np.clip(a + mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True  # ensure at least one crossover
                trial_vector = np.where(crossover_mask, mutant_vector, pop[i])
                
                # Selection
                trial_fitness = func(trial_vector)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial_vector
                    fitness[i] = trial_fitness
                
                if eval_count >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return pop[best_index]