import numpy as np

class AdaptiveSymbioticOrganismsSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.interaction_probability = 0.8
        self.adaptation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                partner_index = np.random.choice(self.population_size)
                if partner_index == i:
                    continue

                if np.random.rand() < self.interaction_probability:
                    new_solution = self.mutualism_phase(population[i], population[partner_index], lb, ub)
                else:
                    new_solution = self.commensalism_phase(population[i], best_position, lb, ub)

                new_fitness = func(new_solution)
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    population[i] = new_solution

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = new_solution

                if evaluations >= self.budget:
                    break

            # Adaptive interaction probability
            self.interaction_probability = min(1.0, self.interaction_probability + self.adaptation_rate * (1 - np.mean(fitness) / (np.min(fitness) + 1e-9)))

        return best_position, fitness[best_index]

    def mutualism_phase(self, organism, partner, lb, ub):
        common_benefit = (organism + partner) / 2
        new_solution = organism + np.random.rand() * (common_benefit - organism)
        return np.clip(new_solution, lb, ub)

    def commensalism_phase(self, organism, best_organism, lb, ub):
        new_solution = organism + np.random.uniform(-1, 1, self.dim) * (best_organism - organism)
        return np.clip(new_solution, lb, ub)