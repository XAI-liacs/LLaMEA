import numpy as np
import random
import operator

class CPHHNAT:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.hypernetwork = self.initialize_hypernetwork()
        self.adaptive_hyperparameters = self.initialize_adaptive_hyperparameters()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def initialize_hypernetwork(self):
        hypernetwork = []
        for _ in range(self.population_size):
            hyperparameters = {
                'crossover_probability': np.random.uniform(0.0, 1.0),
              'mutation_probability': np.random.uniform(0.0, 1.0),
                'elitism_rate': np.random.uniform(0.0, 1.0)
            }
            hypernetwork.append(hyperparameters)
        return hypernetwork

    def initialize_adaptive_hyperparameters(self):
        adaptive_hyperparameters = {
            'crossover_probability': 0.011627906976744186,
           'mutation_probability': 0.011627906976744186,
            'elitism_rate': 0.011627906976744186
        }
        return adaptive_hyperparameters

    def evaluate(self, func):
        evaluations = []
        for individual in self.population:
            evaluation = func(individual)
            evaluations.append(evaluation)
        return evaluations

    def selection(self, evaluations):
        sorted_indices = np.argsort(evaluations)
        selected_indices = sorted_indices[:int(self.population_size * 0.2)]
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    def crossover(self, selected_individuals):
        offspring = []
        for _ in range(self.population_size - len(selected_individuals)):
            parent1, parent2 = random.sample(selected_individuals, 2)
            crossover_point = np.random.randint(1, self.dim)
            child = (parent1[:crossover_point] + parent2[crossover_point:]) if np.random.rand() < self.hypernetwork[0]['crossover_probability'] else parent1
            offspring.append(child)
        return offspring

    def mutation(self, offspring):
        mutated_offspring = []
        for individual in offspring:
            for i in range(self.dim):
                if np.random.rand() < self.hypernetwork[0]['mutation_probability']:
                    individual[i] += np.random.uniform(-1.0, 1.0)
                    individual[i] = max(-5.0, min(5.0, individual[i]))
            mutated_offspring.append(individual)
        return mutated_offspring

    def elitism(self, mutated_offspring):
        best_individuals = [min(mutated_offspring, key=operator.itemgetter(0))]
        best_individuals.extend(mutated_offspring[:int(self.population_size * 0.1)])
        return best_individuals

    def adaptive_hyperparameter_tuning(self, func):
        for _ in range(self.budget):
            evaluations = self.evaluate(func)
            selected_individuals = self.selection(evaluations)
            offspring = self.crossover(selected_individuals)
            mutated_offspring = self.mutation(offspring)
            best_individuals = self.elitism(mutated_offspring)
            self.population = best_individuals

            # Update adaptive hyperparameters based on the fitness of the best individual
            best_individual = min(self.population, key=operator.itemgetter(0))
            self.adaptive_hyperparameters['crossover_probability'] = max(0.0, min(1.0, self.adaptive_hyperparameters['crossover_probability'] + 0.001 * (best_individual[0] - self.hypernetwork[0]['crossover_probability'])))
            self.adaptive_hyperparameters['mutation_probability'] = max(0.0, min(1.0, self.adaptive_hyperparameters['mutation_probability'] + 0.001 * (best_individual[0] - self.hypernetwork[0]['mutation_probability'])))
            self.adaptive_hyperparameters['elitism_rate'] = max(0.0, min(1.0, self.adaptive_hyperparameters['elitism_rate'] + 0.001 * (best_individual[0] - self.hypernetwork[0]['elitism_rate']))

        best_individual = min(self.population, key=operator.itemgetter(0))
        return best_individual

# Example usage:
def func(x):
    return np.sum(x**2)

cpnhnAT = CPHHNAT(budget=100, dim=5)
best_individual = cpnhnAT.adaptive_hyperparameter_tuning(func)
print(best_individual)