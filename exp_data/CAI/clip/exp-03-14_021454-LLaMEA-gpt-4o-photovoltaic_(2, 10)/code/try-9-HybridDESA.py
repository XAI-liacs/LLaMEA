import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, dim * 5)
        self.F_lower = 0.5
        self.F_upper = 0.9
        self.CR = 0.9
        self.temp_start = 1.0
        self.temp_end = 0.01
        self.dynamic_population_reduction = True

    def _adaptive_differential_evolution(self, func, population, scores):
        new_population = []
        for i, target in enumerate(population):
            F = np.random.uniform(self.F_lower, self.F_upper)
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, target)
            trial_score = func(trial)
            if trial_score < scores[i]:
                new_population.append(trial)
                scores[i] = trial_score
            else:
                new_population.append(target)
        return np.array(new_population), scores

    def _simulated_annealing(self, func, best_solution, best_score, temp):
        candidate = best_solution + np.random.uniform(-0.5, 0.5, self.dim)
        candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
        candidate_score = func(candidate)
        if candidate_score < best_score or np.random.rand() < np.exp(-(candidate_score - best_score) / temp):
            return candidate, candidate_score
        return best_solution, best_score

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        best_idx = np.argmin(scores)
        best_solution = population[best_idx]
        best_score = scores[best_idx]

        num_evaluations = self.population_size
        temp_decay = (self.temp_start - self.temp_end) / self.budget

        while num_evaluations < self.budget:
            population, scores = self._adaptive_differential_evolution(func, population, scores)
            best_idx = np.argmin(scores)
            if scores[best_idx] < best_score:
                best_solution = population[best_idx]
                best_score = scores[best_idx]

            temp = self.temp_start - num_evaluations * temp_decay
            best_solution, best_score = self._simulated_annealing(func, best_solution, best_score, temp)
            num_evaluations += 1

            if self.dynamic_population_reduction and num_evaluations > self.budget * 0.5:
                self.population_size = max(4, self.population_size // 2)

        return best_solution, best_score