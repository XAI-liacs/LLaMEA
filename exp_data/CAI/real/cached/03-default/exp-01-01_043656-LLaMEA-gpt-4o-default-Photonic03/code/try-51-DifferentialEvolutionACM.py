import numpy as np

class DifferentialEvolutionACM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.initial_crossover_rate = 0.9
        self.final_crossover_rate = 0.3
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.4

    def adapt_parameters(self, diversity, eval_count):
        lambda_factor = eval_count / self.budget
        crossover_rate = (self.initial_crossover_rate - self.final_crossover_rate) * (1 - lambda_factor) + self.final_crossover_rate
        mutation_factor = (self.initial_mutation_factor - self.final_mutation_factor) * (1 - diversity) + self.final_mutation_factor
        return crossover_rate, mutation_factor

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            diversity = np.mean(np.std(pop, axis=0) / (bounds[:, 1] - bounds[:, 0]))
            crossover_rate, mutation_factor = self.adapt_parameters(diversity, eval_count)

            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                
                trial = np.copy(pop[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < crossover_rate or j == jrand:
                        trial[j] = mutant[j]
                
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx]