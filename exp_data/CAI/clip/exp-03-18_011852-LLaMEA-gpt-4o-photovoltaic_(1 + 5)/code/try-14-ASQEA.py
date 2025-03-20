import numpy as np

class ASQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5
        self.elite_fraction = 0.1
        self.inertia_weight = 0.7  # New line
        self.cognitive_weight = 1.4  # New line
        self.social_weight = 1.4  # New line

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))  # New line
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        personal_best = np.copy(population)  # New line
        personal_best_fitness = np.copy(fitness)  # New line
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = []
            elite_size = int(self.elite_fraction * self.population_size)
            elites = population[np.argsort(fitness)[:elite_size]]
            for i in range(self.population_size - elite_size):  # Modified line
                r1, r2 = np.random.rand(), np.random.rand()  # New line
                velocity[i] = (self.inertia_weight * velocity[i] +  # New line
                               self.cognitive_weight * r1 * (personal_best[i] - population[i]) +  # New line
                               self.social_weight * r2 * (best_solution - population[i]))  # New line
                offspring = population[i] + velocity[i]  # New line
                offspring = self.adaptive_mutation(offspring, lb, ub)
                new_population.append(offspring)
            
            new_population.extend(elites)
            new_population = np.array(new_population)
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size
            
            population, fitness = self.selection(population, fitness, new_population, new_fitness)
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution, best_fitness = population[current_best_idx], fitness[current_best_idx]
            for i in range(self.population_size):  # New line
                if fitness[i] < personal_best_fitness[i]:  # New line
                    personal_best[i] = population[i]  # New line
                    personal_best_fitness[i] = fitness[i]  # New line
        
        return best_solution, best_fitness
    
    def select_parents(self, population, fitness):
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
        return population[idx1], population[idx2]
    
    def quantum_crossover(self, parent1, parent2, lb, ub):
        q1 = self.alpha * parent1 + (1 - self.alpha) * parent2
        q2 = self.alpha * parent2 + (1 - self.alpha) * parent1
        offspring = q1 if np.random.rand() < 0.5 else q2
        offspring = np.clip(offspring, lb, ub)
        return offspring
    
    def adaptive_mutation(self, individual, lb, ub):
        mutation_strength = 0.1 * (ub - lb) * np.random.rand(self.dim)
        mutated = individual + np.random.normal(0, mutation_strength, self.dim)
        return np.clip(mutated, lb, ub)

    def selection(self, population, fitness, new_population, new_fitness):
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))
        sorted_indices = np.argsort(combined_fitness)
        return combined_population[sorted_indices][:self.population_size], combined_fitness[sorted_indices][:self.population_size]