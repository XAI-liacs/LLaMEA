import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=None, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population_size = population_size or 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        # Optimization loop
        while eval_count < self.budget:
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Mutation with adaptive F
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                adaptive_F = np.random.uniform(0.5, 1.0)  # Adaptive F
                mutant = population[a] + adaptive_F * (population[b] - population[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover with adaptive CR
                adaptive_CR = np.random.uniform(0.5, 1.0)  # Adaptive CR
                crossover = np.random.rand(self.dim) < adaptive_CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Selection with elitism
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
            
            # Preserve best solution
            best_idx = np.argmin(fitness)
            new_population[np.argmax(new_fitness)] = population[best_idx]
            new_fitness[np.argmax(new_fitness)] = fitness[best_idx]
            population, fitness = new_population, new_fitness

        return self.f_opt, self.x_opt