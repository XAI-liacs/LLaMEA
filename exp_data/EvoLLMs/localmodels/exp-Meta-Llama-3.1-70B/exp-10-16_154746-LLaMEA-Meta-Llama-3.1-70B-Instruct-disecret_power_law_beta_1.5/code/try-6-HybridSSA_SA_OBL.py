import numpy as np

class HybridSSA_SA_OBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.T = 1000
        self.alpha = 0.99
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.leader = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.leader_fitness = np.inf

    def opposition_based_learning(self, individual):
        opposite_individual = self.lower_bound + self.upper_bound - individual
        return opposite_individual

    def levy_flight(self, individual):
        levy_flight = np.random.normal(0, 1, self.dim) / np.random.normal(0, 1, self.dim)
        levy_flight = individual + levy_flight
        return levy_flight

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.population[i])
                evaluations += 1
                if fitness < self.leader_fitness:
                    self.leader_fitness = fitness
                    self.leader = self.population[i]
                c1 = 2 * np.random.uniform(0, 1, self.dim) - 1
                c2 = 2 * np.random.uniform(0, 1, self.dim) - 1
                c3 = 2 * np.random.uniform(0, 1, self.dim) - 1
                A = 2 * c1 * self.leader - c2 * self.population[i]
                B = c3 * self.leader
                self.population[i] = A + B
                levy_flight = self.levy_flight(self.population[i])
                levy_fitness = func(levy_flight)
                evaluations += 1
                if levy_fitness < fitness:
                    self.population[i] = levy_flight
                else:
                    delta = levy_fitness - fitness
                    prob = np.exp(-delta / self.T)
                    self.T *= self.alpha
                    if np.random.uniform(0, 1) < prob:
                        self.population[i] = levy_flight
                # Apply opposition-based learning with a probability of 0.2
                if np.random.uniform(0, 1) < 0.2:
                    opposite_individual = self.opposition_based_learning(self.population[i])
                    opposite_fitness = func(opposite_individual)
                    evaluations += 1
                    if opposite_fitness < fitness:
                        self.population[i] = opposite_individual
        return self.leader