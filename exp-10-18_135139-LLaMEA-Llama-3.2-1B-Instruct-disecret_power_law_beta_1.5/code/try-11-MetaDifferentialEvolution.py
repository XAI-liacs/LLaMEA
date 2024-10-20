# Description: MetaDifferential Evolution with Black Box Optimization
# Code: import numpy as np
# class MetaDifferentialEvolution:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population_size = 100
#         self.mutation_rate = 0.01
#         self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))

#     def __call__(self, func):
#         while self.budget > 0:
#             # Generate a new population by perturbing the current population
#             new_population = self.generate_new_population()

#             # Evaluate the new population using the given budget
#             new_population_evaluations = np.random.randint(1, self.budget + 1)

#             # Evaluate the new population
#             new_population_evaluations = np.minimum(new_population_evaluations, self.budget)

#             # Select the fittest individuals from the new population
#             self.population = self.select_fittest(new_population, new_population_evaluations)

#             # Update the population size
#             self.population_size = min(self.population_size, len(new_population))

#             # Check if the population has been fully optimized
#             if len(self.population) == 0:
#                 break

#             # Perform mutation on the fittest individuals
#             self.population = self.mutate(self.population)

#         # Return the optimized function
#         return func

#     def generate_new_population(self):
#         new_population = self.population.copy()
#         for _ in range(self.population_size // 2):
#             # Perturb the current individual
#             new_population[random.randint(0, self.dim - 1)] += random.uniform(-5.0, 5.0)

#         return new_population

#     def select_fittest(self, new_population, new_population_evaluations):
#         # Calculate the fitness of each individual
#         fitness = np.abs(new_population_evaluations)

#         # Select the fittest individuals
#         fittest_population = new_population[fitness.argsort()[:len(fitness)]]  # Changed to fittest_population

#         return fittest_population

#     def mutate(self, population):
#         mutated_population = population.copy()
#         for _ in range(self.mutation_rate * len(population)):
#             # Select a random individual
#             individual = random.choice(mutated_population)

#             # Perform mutation
#             mutated_individual = individual.copy()
#             for i in range(self.dim):
#                 # Perturb the individual
#                 mutated_individual[i] += random.uniform(-5.0, 5.0)

#             # Clip the mutated individual to the bounds
#             mutated_individual[i] = np.clip(mutated_individual[i], -5.0, 5.0)

#         return mutated_population