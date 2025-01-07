import numpy as np
from collections import deque

class Quantum_Swarm_GA:
    def __init__(self, budget, dim, population_size=20, crossover_prob=0.7, mutation_prob=0.1, quantum_prob=0.3, memory_size=10):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.quantum_prob = quantum_prob
        self.memory_size = memory_size
        self.evaluations = 0
        self.tabu_list = deque(maxlen=self.memory_size)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')
        
        population = self.initialize_population(self.population_size, lb, ub)
        
        while self.evaluations < self.budget:
            new_population = []
            
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population, func)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                offspring1 = self.mutate(offspring1, lb, ub)
                offspring2 = self.mutate(offspring2, lb, ub)
                
                if np.random.rand() < self.quantum_prob:
                    offspring1 = self.quantum_perturbation(offspring1, lb, ub)
                if np.random.rand() < self.quantum_prob:
                    offspring2 = self.quantum_perturbation(offspring2, lb, ub)
                
                new_population.extend([offspring1, offspring2])

            population = new_population

            for individual in population:
                if tuple(individual) in self.tabu_list:
                    continue

                value = func(individual)
                self.evaluations += 1
                self.tabu_list.append(tuple(individual))

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = individual

                if self.evaluations >= self.budget:
                    break

        return best_global_position
    
    def initialize_population(self, population_size, lb, ub):
        return np.random.uniform(lb, ub, (population_size, self.dim))

    def select_parents(self, population, func):
        indices = np.random.choice(range(len(population)), size=2, replace=False)
        return population[indices[0]], population[indices[1]]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, self.dim)
            offspring1 = np.concatenate((parent1[:point], parent2[point:]))
            offspring2 = np.concatenate((parent2[:point], parent1[point:]))
        else:
            offspring1, offspring2 = parent1, parent2
        return offspring1, offspring2

    def mutate(self, individual, lb, ub):
        if np.random.rand() < self.mutation_prob:
            mutation_vector = np.random.normal(0, 0.1, self.dim)
            individual = np.clip(individual + mutation_vector, lb, ub)
        return individual

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)