import numpy as np
import random

class HybridGeneticSimulatedAnnealingParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.temperature = 1000
        self.cooling_rate = 0.95
        self.inertia_weight = 0.5  # New inertia weight parameter
        self.cognitive_coeff = 1.5  # New cognitive parameter
        self.social_coeff = 1.5  # New social parameter

    def initialize_population(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(self.population_size)]

    def initialize_velocities(self, bounds):  # New function for initializing velocities
        return [np.random.uniform(-1, 1, self.dim) for _ in range(self.population_size)]

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
            if random.random() < self.mutation_rate:
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

    def update_velocities_and_positions(self, velocities, population, personal_best, global_best, bounds):  # New function
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive_velocity = self.cognitive_coeff * r1 * (personal_best[i] - population[i])
            social_velocity = self.social_coeff * r2 * (global_best - population[i])
            velocities[i] = (self.inertia_weight * velocities[i] +
                             cognitive_velocity +
                             social_velocity)
            population[i] = np.clip(population[i] + velocities[i], bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        velocities = self.initialize_velocities(bounds)  # Initializing velocities
        fitness = self.evaluate_population(population, func)
        budget_used = len(population)

        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best_index = np.argmin(fitness)
        global_best = population[global_best_index]

        while budget_used < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
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

            # Particle Swarm Update
            self.update_velocities_and_positions(velocities, population, personal_best, global_best, bounds)

            # Update personal and global bests
            for i in range(self.population_size):
                if fitness[i] < personal_best_fitness[i]:
                    personal_best[i], personal_best_fitness[i] = population[i], fitness[i]
            global_best_index = np.argmin(fitness)
            global_best = population[global_best_index]

        best_index = np.argmin(fitness)
        return population[best_index]