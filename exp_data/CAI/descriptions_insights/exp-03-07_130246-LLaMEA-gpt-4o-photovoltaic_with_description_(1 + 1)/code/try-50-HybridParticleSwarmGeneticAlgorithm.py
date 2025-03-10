import numpy as np
import random

class HybridParticleSwarmGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.inertia_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.best_global = None
        self.best_global_fitness = float('inf')

    def initialize_population(self, bounds):
        population = [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(self.population_size)]
        velocities = [np.zeros(self.dim) for _ in range(self.population_size)]
        return population, velocities

    def evaluate_population(self, population, func):
        return [func(ind) for ind in population]

    def update_velocity(self, velocity, position, best_position, bounds):
        inertia = self.inertia_weight * velocity
        cognitive = self.cognitive_weight * random.random() * (best_position - position)
        social = self.social_weight * random.random() * (self.best_global - position)
        return inertia + cognitive + social

    def update_position(self, position, velocity, bounds):
        new_position = position + velocity
        new_position = np.clip(new_position, bounds.lb, bounds.ub)
        return new_position

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
            if random.random() < self.mutation_rate:
                individual[i] = np.random.uniform(bounds.lb[i], bounds.ub[i])
        return individual

    def local_search(self, individual, fitness, func, bounds):
        current = individual
        current_cost = fitness
        for _ in range(5):  # Limit local search steps to improve efficiency
            neighbor = self.mutate(current.copy(), bounds)
            neighbor_cost = func(neighbor)
            if neighbor_cost < current_cost:
                current, current_cost = neighbor, neighbor_cost
        return current, current_cost

    def __call__(self, func):
        bounds = func.bounds
        population, velocities = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)
        budget_used = len(population)
        best_positions = population.copy()

        while budget_used < self.budget:
            for i in range(self.population_size):
                velocities[i] = self.update_velocity(velocities[i], population[i], best_positions[i], bounds)
                population[i] = self.update_position(population[i], velocities[i], bounds)

            new_fitness = self.evaluate_population(population, func)
            budget_used += self.population_size

            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    best_positions[i] = population[i]
                    fitness[i] = new_fitness[i]
                
                if fitness[i] < self.best_global_fitness:
                    self.best_global = population[i]
                    self.best_global_fitness = fitness[i]

            for i in range(len(population)):
                population[i], fitness[i] = self.local_search(population[i], fitness[i], func, bounds)
                budget_used += 1

        best_index = np.argmin(fitness)
        return population[best_index]