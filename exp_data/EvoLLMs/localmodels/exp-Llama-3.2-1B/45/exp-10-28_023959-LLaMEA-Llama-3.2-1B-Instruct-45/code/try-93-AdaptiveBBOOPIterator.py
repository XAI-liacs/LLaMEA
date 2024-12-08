import random
import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBBOOPIterator:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, dim)
        self.population = [self.search_space] * dim
        self.fitness_values = np.zeros(dim)
        self.population_indices = list(range(dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = self.select_parents()

            # Evolve the population using mutation and selection
            self.evolve_population(parents, func)

            # Evaluate the fitness of each individual
            self.evaluate_fitness()

        # Return the best individual
        return self.get_best_individual()

    def select_parents(self):
        # Select parents using tournament selection
        tournament_size = 3
        tournament_indices = random.sample(self.population_indices, tournament_size)
        tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
        tournament_parents = []
        for i in range(tournament_size):
            parent_index = tournament_indices[i]
            parent_fitness_value = tournament_fitness_values[i]
            parent_index = random.choice(self.population_indices)
            parent_fitness_value = self.fitness_values[parent_index]
            if parent_fitness_value < parent_fitness_value:
                parent_index = parent_index
            tournament_parents.append(self.population[parent_index])
        return tournament_parents

    def evolve_population(self, parents, func):
        # Evolve the population using mutation and selection
        for parent in parents:
            # Randomly mutate the parent
            mutated_parent = parent.copy()
            if random.random() < 0.1:
                mutated_parent[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
            # Select the best parent based on fitness
            best_parent_index = np.argmax(self.fitness_values)
            self.population[self.population_indices[best_parent_index]] = mutated_parent
            self.fitness_values[best_parent_index] = func(mutated_parent)

    def evaluate_fitness(self):
        # Evaluate the fitness of each individual
        for i in range(self.dim):
            self.fitness_values[i] = func(self.population[i])

    def get_best_individual(self):
        # Return the best individual
        return self.population[np.argmax(self.fitness_values)]

    def mutation(self, individual):
        # Randomly mutate the individual
        mutated_individual = individual.copy()
        if random.random() < 0.1:
            mutated_individual[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
        return mutated_individual

    def selection(self, tournament_parents):
        # Select the best individual based on fitness
        selected_individuals = np.random.choice(self.population_indices, self.budget, replace=False)
        selected_individuals = [self.population[i] for i in selected_individuals]
        selected_individuals = [self.mutation(individual) for individual in selected_individuals]
        selected_individuals = [self.evaluate_fitness(individual) > self.fitness_values[i] for i, individual in enumerate(selected_individuals)]
        return [individuals[i] for i, individuals in enumerate(selected_individuals) if individuals[i]]

# Description: Adaptive Black Box Optimization with Metaheuristic Evolutionary Strategies
# Code: 