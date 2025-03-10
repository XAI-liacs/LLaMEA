import numpy as np

class HybridLayerOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.current_budget = 0
        self.layers_increment_step = 5

    def __call__(self, func):
        # Initialize population
        pop = self.initialize_population(func.bounds.lb, func.bounds.ub)
        best_solution = None
        best_score = float('-inf')
        
        while self.current_budget < self.budget:
            scores = self.evaluate_population(pop, func)

            # Update the best solution found so far
            for i, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_solution = pop[i].copy()

            # Selection
            selected_parents = self.tournament_selection(pop, scores)

            # Crossover
            offspring = self.crossover(selected_parents, func.bounds.lb, func.bounds.ub)

            # Mutation
            self.mutate(offspring, func.bounds.lb, func.bounds.ub)

            # Local Search and Incremental Layer Optimization
            self.local_search_and_layer_increment(offspring, func)

            # Next generation
            pop = offspring

        return best_solution

    def initialize_population(self, lb, ub):
        return [np.random.uniform(lb, ub, self.dim) for _ in range(self.population_size)]

    def evaluate_population(self, pop, func):
        scores = []
        for p in pop:
            if self.current_budget < self.budget:
                score = func(p)
                self.current_budget += 1
                scores.append(score)
            else:
                break
        return scores
    
    def tournament_selection(self, pop, scores):
        selected = []
        for _ in range(self.population_size):
            i, j = np.random.choice(len(pop), 2, replace=False)
            if scores[i] > scores[j]:
                selected.append(pop[i])
            else:
                selected.append(pop[j])
        return selected

    def crossover(self, parents, lb, ub):
        offspring = []
        for i in range(0, len(parents), 2):
            if np.random.rand() < self.crossover_rate:
                cross_point = np.random.randint(1, self.dim - 1)
                child1 = np.concatenate((parents[i][:cross_point], parents[i+1][cross_point:]))
                child2 = np.concatenate((parents[i+1][:cross_point], parents[i][cross_point:]))
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i], parents[i+1]])
        return np.clip(offspring, lb, ub)

    def mutate(self, offspring, lb, ub):
        for individual in offspring:
            if np.random.rand() < self.mutation_rate:
                mutation_indices = np.random.randint(0, self.dim, int(self.dim * 0.1))
                individual[mutation_indices] = np.random.uniform(lb[mutation_indices], ub[mutation_indices])
        return np.clip(offspring, lb, ub)
    
    def local_search_and_layer_increment(self, offspring, func):
        for i, ind in enumerate(offspring):
            if self.current_budget < self.budget:
                perturbed = ind + np.random.normal(0, 0.1, self.dim)
                perturbed_score = func(np.clip(perturbed, func.bounds.lb, func.bounds.ub))
                self.current_budget += 1
                if perturbed_score > func(ind):
                    offspring[i] = perturbed
                
                # Increment layer complexity if budget allows
                if self.current_budget + self.layers_increment_step <= self.budget:
                    new_dim = min(self.dim + self.layers_increment_step, len(func.bounds.ub))
                    if new_dim > len(ind):
                        extra_layers = np.random.uniform(func.bounds.lb[len(ind):new_dim], func.bounds.ub[len(ind):new_dim])
                        offspring[i] = np.concatenate((offspring[i], extra_layers))
                        self.dim = new_dim