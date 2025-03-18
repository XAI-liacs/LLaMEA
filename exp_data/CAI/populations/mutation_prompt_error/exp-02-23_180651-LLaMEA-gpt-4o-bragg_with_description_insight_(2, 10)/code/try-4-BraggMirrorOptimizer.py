import numpy as np

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = float('-inf')

    def quasi_oppositional_init(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opposite_population = lb + ub - self.population
        self.population = np.vstack((self.population, opposite_population))
        self.population_size = self.population.shape[0]

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_solution = self.population[best_idx]
        return fitness

    def mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant, func.bounds.lb, func.bounds.ub)  # Enforced boundary handling

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def periodic_cost_function(self, candidate):
        periodic_penalty = np.var(np.diff(candidate.reshape(-1, 2), axis=0))
        return periodic_penalty

    def optimize(self, func):
        evaluations = 0
        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            new_population = np.empty_like(self.population)
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                
                # Enforce boundaries
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                
                # Calculate trial fitness with additional periodic cost
                trial_fitness = func(trial) - self.periodic_cost_function(trial)
                
                if trial_fitness > fitness[i]:
                    new_population[i] = trial
                else:
                    new_population[i] = self.population[i]
                evaluations += 1
                if evaluations >= self.budget:
                    break

            self.population = new_population

        return self.best_solution

    def __call__(self, func):
        # Initialize population with quasi-oppositional strategy
        self.quasi_oppositional_init(func.bounds.lb, func.bounds.ub)
        # Run optimization
        return self.optimize(func)