import numpy as np

class QuantumSwarmDE:
    def __init__(self, budget, dim, population_size=20, mutation_factor=0.8, crossover_rate=0.9, quantum_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        scores = np.array([func(individual) for individual in population])
        self.evaluations += self.population_size
        
        while self.evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                if np.random.rand() < self.quantum_prob:
                    candidate = self.quantum_perturbation(population[i], lb, ub)
                else:
                    candidate = self.mutate_and_crossover(i, population, scores, lb, ub)
                candidate_score = func(candidate)
                self.evaluations += 1

                if candidate_score < scores[i]:
                    new_population.append(candidate)
                    scores[i] = candidate_score
                else:
                    new_population.append(population[i])

                if self.evaluations >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(scores)
        return population[best_idx]

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate_and_crossover(self, idx, population, scores, lb, ub):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = np.clip(population[a] + self.mutation_factor * (population[b] - population[c]), lb, ub)
        crossover = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(crossover, mutant, population[idx])
        return trial

    def quantum_perturbation(self, individual, lb, ub):
        q_individual = individual + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_individual, lb, ub)