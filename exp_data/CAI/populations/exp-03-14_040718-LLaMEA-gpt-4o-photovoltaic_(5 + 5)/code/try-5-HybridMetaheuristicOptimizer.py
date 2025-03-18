import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialization
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = min(50, self.budget // 10)
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            # Selection
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            elite = population[:population_size // 5]
            
            # Crossover and Mutation (Evolutionary Strategy)
            offspring = []
            for i in range(population_size):
                parents = np.random.choice(len(elite), 2, replace=False, p=self.selection_probabilities(fitness))
                parent1, parent2 = elite[parents[0]], elite[parents[1]]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, lb, ub)
                offspring.append(child)
            
            # Evaluation
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)
            
            # Adaptive Neighborhood Search (Local Exploitation)
            for i in range(len(offspring)):
                if evaluations >= self.budget:
                    break
                if np.random.rand() < 0.3:  # Elitist selection for neighborhood search
                    neighbor = self.adaptive_neighbor_search(offspring[i], lb, ub, func)
                    offspring[i] = neighbor
                    offspring_fitness[i] = func(neighbor)
                    evaluations += 1
            
            # Selection for Next Generation
            combined_population = np.vstack((population, offspring))
            combined_fitness = np.hstack((fitness, offspring_fitness))
            sorted_indices = np.argsort(combined_fitness)
            best_indices = sorted_indices[:population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[0]

    def selection_probabilities(self, fitness):
        rank = np.argsort(np.argsort(fitness))
        max_rank = len(fitness)
        return (max_rank - rank) / max_rank

    def crossover(self, parent1, parent2):
        alpha = np.random.uniform(0, 1, self.dim)
        return alpha * parent1 + (1 - alpha) * parent2

    def mutate(self, individual, lb, ub):
        mutation_strength = 0.1 * (ub - lb)
        mutation_vector = np.random.normal(0, mutation_strength, self.dim)
        mutated = individual + mutation_vector
        return np.clip(mutated, lb, ub)

    def adaptive_neighbor_search(self, individual, lb, ub, func):
        step_size = (ub - lb) * 0.05
        for _ in range(3):  # Try multiple small steps
            neighbor = individual + np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(neighbor, lb, ub)
            if func(neighbor) < func(individual):
                individual = neighbor
        return individual