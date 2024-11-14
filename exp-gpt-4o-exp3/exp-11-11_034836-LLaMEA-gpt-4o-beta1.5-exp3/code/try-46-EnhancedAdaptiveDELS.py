import numpy as np

class EnhancedAdaptiveDELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.init_population_size = 10 * dim
        self.population_size = self.init_population_size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.learning_rate = 0.05
        self.covariance_matrix = np.eye(dim)
        self.history = []

    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def _mutate(self, target_idx, population, best_individual):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.mutation_factor * (b - c) + np.random.rand(self.dim) * (best_individual - population[target_idx])
        return np.clip(mutant, self.lb, self.ub)

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _adaptive_learning(self, current_best_fitness):
        if self.history:
            recent_improvement = (self.history[-1] - current_best_fitness) / max(abs(self.history[-1]), 1e-9)
            if recent_improvement < 0.01:
                self.learning_rate = np.clip(self.learning_rate * 1.1, 0.01, 0.1)
            else:
                self.learning_rate = np.clip(self.learning_rate * 0.9, 0.01, 0.1)
        self.history.append(current_best_fitness)

    def _update_covariance(self, population, centroid):
        deviations = population - centroid
        self.covariance_matrix = np.cov(deviations, rowvar=False)
        self.covariance_matrix += np.eye(self.dim) * 1e-6  # Regularization

    def _resize_population(self, evaluations):
        if evaluations > self.budget * 0.25:
            self.population_size = max(5 * self.dim, int(self.init_population_size * 0.75))
        if evaluations > self.budget * 0.75:
            self.population_size = max(3 * self.dim, int(self.init_population_size * 0.5))

    def _hybrid_local_global_search(self, individual, best_individual, centroid):
        global_step = np.random.multivariate_normal(np.zeros(self.dim), self.covariance_matrix)
        local_step = np.random.uniform(-0.1, 0.1, self.dim)
        candidate = individual + global_step + local_step * (best_individual - centroid)
        return np.clip(candidate, self.lb, self.ub)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = self.population_size
        centroid = np.mean(population, axis=0)

        while evaluations < self.budget:
            self._adaptive_learning(best_fitness)
            for i in range(self.population_size):
                mutant = self._mutate(i, population, best_individual)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            self._update_covariance(population, centroid)
            hybrid_candidate = self._hybrid_local_global_search(best_individual, np.mean(population, axis=0), centroid)
            hybrid_fitness = func(hybrid_candidate)
            evaluations += 1

            if hybrid_fitness < best_fitness:
                best_individual = hybrid_candidate
                best_fitness = hybrid_fitness

            self._resize_population(evaluations)

        return best_individual, best_fitness