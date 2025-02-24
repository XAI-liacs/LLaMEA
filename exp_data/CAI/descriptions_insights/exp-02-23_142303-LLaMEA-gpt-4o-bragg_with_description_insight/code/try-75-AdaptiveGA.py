import numpy as np
from scipy.optimize import minimize

class AdaptiveGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.init_budget = budget // 2
        self.local_budget = budget - self.init_budget
        self.elitism_rate = 0.2
        self.dynamic_mutation_rate = 0.05

    def _initialize_population(self, bounds):
        lb, ub = bounds
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_parents(self, population, fitness):
        selected_indices = np.random.choice(self.population_size, size=self.population_size, p=fitness/fitness.sum())
        return population[selected_indices]

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2

    def _mutate(self, individual, bounds):
        if np.random.rand() < self.mutation_rate:
            lb, ub = bounds
            mutation_vector = np.random.uniform(lb, ub, self.dim)
            individual = np.clip(individual + self.dynamic_mutation_rate * (mutation_vector - individual), lb, ub)
        return individual

    def _adaptive_mutation(self, iteration):
        self.mutation_rate = min(0.9, self.mutation_rate + 0.01 * iteration / self.init_budget)

    def _ga_step(self, population, func, bounds):
        fitness = 1.0 / (1.0 + self._evaluate_population(population, func))
        best_indices = np.argsort(fitness)[-int(self.elitism_rate * self.population_size):]
        elites = population[best_indices]

        new_population = self._select_parents(population, fitness)
        new_population = [self._mutate(child, bounds) for child in new_population]

        children = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = new_population[i], new_population[min(i+1, self.population_size-1)]
            child1, child2 = self._crossover(parent1, parent2)
            children.append(self._mutate(child1, bounds))
            children.append(self._mutate(child2, bounds))

        new_population = np.array(children)[:self.population_size]
        return np.vstack((elites, new_population[:-len(elites)]))

    def _local_refinement(self, solution, func, bounds):
        result = minimize(func, solution, method='Nelder-Mead', options={'maxfev': self.local_budget})
        return result.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = self._initialize_population(bounds)

        for iteration in range(self.init_budget // self.population_size):
            population = self._ga_step(population, func, bounds)
            self._adaptive_mutation(iteration)

        best_solution = population[np.argmin(self._evaluate_population(population, func))]

        if self.local_budget > 0:
            best_solution = self._local_refinement(best_solution, func, bounds)

        return best_solution