import numpy as np

class HybridGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.fitness = None
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.quantum_mutation_prob = 0.05
        self.niche_radius = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        self.fitness = fitness
        return fitness

    def select_parents(self):
        idx = np.random.choice(self.population_size, size=2, replace=False, p=self.fitness / self.fitness.sum())
        return self.population[idx[0]], self.population[idx[1]]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim-1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, individual, lb, ub):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.randn() * 0.1
                individual[i] = np.clip(individual[i], lb[i], ub[i])

    def apply_quantum_mutation(self, individual, lb, ub):
        if np.random.rand() < self.quantum_mutation_prob:
            q_mutation = lb + (ub - lb) * np.random.rand(self.dim)
            individual = np.mean([individual, q_mutation], axis=0)
            individual = np.clip(individual, lb, ub)
        return individual

    def niche_sharing(self):
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                distance = np.linalg.norm(self.population[i] - self.population[j])
                if distance < self.niche_radius:
                    if self.fitness[i] > self.fitness[j]:
                        self.fitness[i] += self.niche_radius - distance
                    else:
                        self.fitness[j] += self.niche_radius - distance

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            self.niche_sharing()
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1, lb, ub)
                self.mutate(child2, lb, ub)
                child1 = self.apply_quantum_mutation(child1, lb, ub)
                child2 = self.apply_quantum_mutation(child2, lb, ub)
                new_population.extend([child1, child2])

            self.population = np.array(new_population)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]