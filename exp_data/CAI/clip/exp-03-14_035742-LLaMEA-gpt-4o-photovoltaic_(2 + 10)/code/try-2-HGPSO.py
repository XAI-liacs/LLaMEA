import numpy as np

class HGPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.inertia_weight = 0.9
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.diversity_threshold = 0.1

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = personal_best_scores.min()
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coef * r1 * (personal_best - population) +
                          self.social_coef * r2 * (global_best - population))
            population = population + velocities
            population = np.clip(population, lb, ub)
            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size
            
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best[better_mask] = population[better_mask]
            
            if personal_best_scores.min() < global_best_score:
                global_best_score = personal_best_scores.min()
                global_best = personal_best[personal_best_scores.argmin()]

            # Adaptive inertia weight
            self.inertia_weight = 0.4 + 0.5 * (1 - evaluations / self.budget)
            
            if np.std(scores) < self.diversity_threshold and evaluations < self.budget:
                mutation_idx = np.random.randint(0, self.dim, size=self.population_size)
                mutation_values = np.random.uniform(lb, ub, size=self.population_size)
                population[np.arange(self.population_size), mutation_idx] = mutation_values
                scores = np.array([func(ind) for ind in population])
                evaluations += self.population_size

            if evaluations < self.budget:
                for i in range(0, self.population_size, 2):
                    if evaluations >= self.budget: break
                    if np.random.rand() < self.crossover_rate:
                        cross_point = np.random.randint(1, self.dim - 1)
                        offspring1 = np.concatenate((population[i, :cross_point], population[i + 1, cross_point:]))
                        offspring2 = np.concatenate((population[i + 1, :cross_point], population[i, cross_point:]))
                        population[i], population[i + 1] = offspring1, offspring2
                        scores[i], scores[i + 1] = func(offspring1), func(offspring2)
                        evaluations += 2
                    
                    if evaluations < self.budget and np.random.rand() < self.mutation_rate:
                        mutation_idx = np.random.randint(0, self.dim)
                        population[i, mutation_idx] = np.random.uniform(lb[mutation_idx], ub[mutation_idx])
                        scores[i] = func(population[i])
                        evaluations += 1

        return global_best, global_best_score