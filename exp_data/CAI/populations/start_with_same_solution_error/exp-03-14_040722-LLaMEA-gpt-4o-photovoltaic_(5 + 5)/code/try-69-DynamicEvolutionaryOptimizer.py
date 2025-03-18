import numpy as np

class DynamicEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.current_evaluations = 0

    def __call__(self, func):
        pop_size = 12 + 6 * self.dim
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += pop_size

        while self.current_evaluations < self.budget:
            if self.current_evaluations % (self.budget // 5) == 0:
                pop_size += 2

            num_parents = pop_size // 2
            parents_indices = np.argsort(fitness)[:num_parents]
            parents = population[parents_indices]

            mutation_strength = ((func.bounds.ub - func.bounds.lb) / 12.0) * (1 - np.linspace(0, 0.6, num_parents)).reshape(-1, 1)
            
            offspring = []
            for i in range(num_parents):
                parent1 = parents[i]
                parent2 = parents[np.random.choice(num_parents, p=(1/num_parents) * np.ones(num_parents))]
                crossover_point = np.random.randint(1, self.dim)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                mutated = child + np.random.normal(0, mutation_strength[i], self.dim)
                mutated = np.clip(mutated, func.bounds.lb, func.bounds.ub)
                offspring.append(mutated)

            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.current_evaluations += len(offspring)

            # Change 1: Modify elitism rate based on current best fitness
            elite_size = int(0.05 * pop_size if fitness.min() < 0.5 else 0.15 * pop_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite = population[elite_indices]

            population = np.vstack((elite, parents, offspring))
            fitness = np.hstack((fitness[elite_indices], fitness[parents_indices], offspring_fitness))
            best_indices = np.argsort(fitness)[:pop_size]
            population = population[best_indices]
            fitness = fitness[best_indices]

        best_index = np.argmin(fitness)
        return population[best_index]