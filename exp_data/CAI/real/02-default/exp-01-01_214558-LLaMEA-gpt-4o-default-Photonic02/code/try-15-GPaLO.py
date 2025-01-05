import numpy as np

class GPaLO:
    def __init__(self, budget, dim, population_size=50, alpha=0.1, beta=1.5, crossover_rate=0.8, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.beta = beta  # Levy parameter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_global_position = None
        best_global_value = float('inf')

        while self.evaluations < self.budget:
            # Evaluate and select the best solution in the population
            fitness_values = np.array([func(individual) for individual in population])
            self.evaluations += self.population_size

            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < best_global_value:
                best_global_value = fitness_values[best_idx]
                best_global_position = population[best_idx].copy()

            # Perform selection, crossover, and mutation
            selected_parents = self.tournament_selection(population, fitness_values)
            offspring = self.crossover_and_mutate(selected_parents, lb, ub)

            # Levy flight step for enhanced exploration
            levy_offspring = self.levy_flight(offspring, lb, ub)

            # Combine offspring and levy offspring, then evaluate
            population = self.select_next_generation(offspring, levy_offspring, func)
            self.evaluations += len(levy_offspring)

            if self.evaluations >= self.budget:
                break

        return best_global_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def tournament_selection(self, population, fitness_values, tournament_size=3):
        selected_parents = []
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            winner_idx = tournament_indices[np.argmin(fitness_values[tournament_indices])]
            selected_parents.append(population[winner_idx])
        return np.array(selected_parents)

    def crossover_and_mutate(self, parents, lb, ub):
        offspring = []
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            if i + 1 < self.population_size:
                parent2 = parents[i + 1]

                if np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.dim)
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                offspring.extend([self.mutate(child1, lb, ub), self.mutate(child2, lb, ub)])
                
        return np.array(offspring)

    def mutate(self, individual, lb, ub):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = (np.random.rand(self.dim) - 0.5) * (ub - lb)
            individual = np.clip(individual + mutation_vector, lb, ub)
        return individual

    def levy_flight(self, offspring, lb, ub):
        levy_step = self.alpha * (np.random.normal(size=(len(offspring), self.dim)) / 
                                  np.abs(np.random.normal(size=(len(offspring), self.dim)))**(1/self.beta))
        levy_offspring = np.clip(offspring + levy_step, lb, ub)
        return levy_offspring

    def select_next_generation(self, offspring, levy_offspring, func):
        combined_population = np.vstack((offspring, levy_offspring))
        fitness_values = np.array([func(individual) for individual in combined_population])
        best_indices = np.argsort(fitness_values)[:self.population_size]
        return combined_population[best_indices]