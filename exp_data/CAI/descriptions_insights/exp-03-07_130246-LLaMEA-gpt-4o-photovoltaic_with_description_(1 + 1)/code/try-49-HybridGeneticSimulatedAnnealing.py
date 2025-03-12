import numpy as np
import random

class HybridGeneticSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.temperature = 1000
        self.cooling_rate = 0.95

    def initialize_population(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(self.population_size)]

    def evaluate_population(self, population, func):
        return [func(ind) for ind in population]

    def select_parents(self, population, fitness):
        total_fitness = sum(fitness)
        selection_probs = [f / total_fitness for f in fitness]
        return random.choices(population, weights=selection_probs, k=2)
    
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.dim - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2

    def mutate(self, individual, bounds):
        for i in range(self.dim):
            if random.random() < self.mutation_rate * (1 - i / self.dim):  # Enhanced mutation strategy
                individual[i] = np.random.uniform(bounds.lb[i], bounds.ub[i])
        return individual

    def acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        else:
            return np.exp((old_cost - new_cost) / temperature)

    def anneal(self, individual, fitness, func, bounds):
        current = individual
        current_cost = fitness
        for _ in range(10):  # Limit local search steps
            neighbor = self.mutate(current.copy(), bounds)
            neighbor_cost = func(neighbor)
            if self.acceptance_probability(current_cost, neighbor_cost, self.temperature) > random.random():
                current, current_cost = neighbor, neighbor_cost
        return current, current_cost

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)
        budget_used = len(population)

        while budget_used < self.budget:
            new_population_size = min(self.population_size, self.budget - budget_used)  # Dynamic population size
            new_population = []
            for _ in range(new_population_size // 2):
                parent1, parent2 = self.select_parents(population, fitness)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, bounds)
                child2 = self.mutate(child2, bounds)
                new_population.extend([child1, child2])

            new_fitness = self.evaluate_population(new_population, func)
            budget_used += len(new_population)

            for i in range(len(new_population)):
                new_population[i], new_fitness[i] = self.anneal(new_population[i], new_fitness[i], func, bounds)
                budget_used += 1

            combined = list(zip(population + new_population, fitness + new_fitness))
            combined.sort(key=lambda x: x[1])
            population, fitness = zip(*combined[:self.population_size])
            population, fitness = list(population), list(fitness)
            self.temperature *= self.cooling_rate

        best_index = np.argmin(fitness)
        return population[best_index]