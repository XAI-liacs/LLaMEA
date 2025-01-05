import numpy as np

class Adaptive_Memetic_Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40  # Larger pool for diversity
        self.elite_fraction = 0.2  # Fraction of top performers
        self.local_search_rate = 0.1  # Initial rate for local search
        self.mutation_rate = 0.05
        self.crossover_rate = 0.7
        self.epsilon_decay = 0.995  # Decay factor for local search rate
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_population = population[elite_indices]
            
            # Adaptive strategy update
            self.local_search_rate *= self.epsilon_decay
            
            # Local Search on Elite
            for i in elite_population:
                if np.random.rand() < self.local_search_rate:
                    step_size = np.random.normal(0, 0.1, self.dim)
                    new_solution = np.clip(i + step_size, lb, ub)
                    new_fitness = func(new_solution)
                    evaluations += 1

                    if new_fitness < func(i):
                        i[:] = new_solution

                    if evaluations >= self.budget:
                        break
            
            # Generate offspring through crossover and mutation
            offspring = []
            while len(offspring) < self.population_size - elite_count:
                parents = np.random.choice(elite_population, size=2, replace=False)
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self.crossover(parents[0], parents[1])
                else:
                    child1, child2 = parents[0], parents[1]
                
                self.mutate(child1)
                self.mutate(child2)
                
                offspring.append(child1)
                offspring.append(child2)
                
            # Evaluate offspring
            offspring = np.array(offspring)[:self.population_size - elite_count]
            offspring_fitness = np.array([func(individual) for individual in offspring])
            evaluations += len(offspring)

            # Update population and fitness
            population = np.vstack((elite_population, offspring))
            fitness = np.hstack((fitness[elite_indices], offspring_fitness))

            if evaluations >= self.budget:
                break

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.normal(0, 0.1, self.dim)
            individual += mutation_vector