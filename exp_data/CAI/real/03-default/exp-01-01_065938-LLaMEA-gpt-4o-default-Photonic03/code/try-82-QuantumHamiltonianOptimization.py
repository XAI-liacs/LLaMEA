import numpy as np

class QuantumHamiltonianOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha_min, self.alpha_max = 0.5, 1.0
        self.beta_min, self.beta_max = 0.3, 0.8
        self.crossover_rate = 0.9
        self.history = []

    def quantum_walk(self, alpha=0.7):
        step = np.random.normal(0, alpha, self.dim)
        return step

    def adaptive_learning_rate(self, fitness):
        return np.exp(-fitness / np.max(fitness))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        best_global = population[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = population[indices]

                alpha = np.random.uniform(self.alpha_min, self.alpha_max)
                beta = np.random.uniform(self.beta_min, self.beta_max)
                quantum_step = self.quantum_walk(alpha)
                learning_rate = self.adaptive_learning_rate(fitness[i])

                mutant = x0 + beta * (x1 - x2) + quantum_step * learning_rate
                mutant = np.clip(mutant, lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = trial
                else:
                    new_population[i] = population[i]

            self.history.append(best_global)
            population = new_population

        return best_global