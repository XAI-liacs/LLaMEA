import numpy as np
from scipy.optimize import minimize

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Adjusted based on problem size
        self.f_initial = 0.8  # Initial differential weight
        self.cr_initial = 0.9  # Initial crossover probability
        self.f_final = 0.6  # Final differential weight
        self.cr_final = 0.5  # Final crossover probability
        self.migration_interval = 0.2  # Fraction of budget before migration occurs

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        population = self.introduce_periodicity(population)
        fitness = np.array([func(ind) for ind in population])
        
        eval_count = self.population_size
        
        while eval_count < self.budget:
            f = self.adaptive_parameter(self.f_initial, self.f_final, eval_count)
            cr = self.adaptive_parameter(self.cr_initial, self.cr_final, eval_count)
            
            new_population = np.zeros_like(population)

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + f * (b - c), bounds[0], bounds[1])
                
                crossover = np.random.rand(self.dim) < cr
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, population[i])
                trial = self.introduce_periodicity(trial.reshape(1, -1))[0]
                
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness > fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

                if eval_count >= self.budget:
                    break
            
            population = new_population
            
            if eval_count / self.budget > self.migration_interval:
                population = self.periodic_migration(population, func)

        best_index = np.argmax(fitness)
        best_solution = population[best_index]
        
        result = minimize(func, best_solution, bounds=bounds.T, method='L-BFGS-B')
        
        return result.x

    def introduce_periodicity(self, population):
        period = self.dim // 2
        for i in range(len(population)):
            for j in range(0, self.dim, period):
                segment = population[i, j:j + period]
                population[i, j:j + period] = np.mean(segment)
        return population

    def adaptive_parameter(self, initial, final, current_eval):
        return initial + (final - initial) * (current_eval / self.budget)

    def periodic_migration(self, population, func):
        # Encourage diversity by periodically reintroducing global search patterns
        num_migrants = self.population_size // 5
        migrants = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_migrants, self.dim))
        population[-num_migrants:] = self.introduce_periodicity(migrants)
        return population