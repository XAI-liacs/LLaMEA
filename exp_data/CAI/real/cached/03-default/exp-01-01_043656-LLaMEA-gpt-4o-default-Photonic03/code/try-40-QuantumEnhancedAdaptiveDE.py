import numpy as np

class QuantumEnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.f_scale_initial = 0.9
        self.f_scale_final = 0.5
        self.cr_initial = 0.9
        self.cr_final = 0.4
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_mutation(self, target, candidates, eval_count):
        indices = np.random.choice(len(candidates), 3, replace=False)
        a, b, c = candidates[indices]
        lambda_factor = (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        
        mutant = a + quantum_factor * (b - c)
        return mutant

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            f_scale = self.f_scale_initial * (1 - eval_count / self.budget) + self.f_scale_final * (eval_count / self.budget)
            cr = self.cr_initial * (1 - eval_count / self.budget) + self.cr_final * (eval_count / self.budget)

            new_population = np.empty_like(population)
            for i in range(self.population_size):
                mutant = self.quantum_mutation(population[i], population, eval_count)
                crossover = np.random.rand(self.dim) < cr
                trial = np.where(crossover, mutant, population[i])
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

                if eval_count >= self.budget:
                    break

            population = new_population
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

        return best_solution