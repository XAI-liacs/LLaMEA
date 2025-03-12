import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = self.dim * 10  # Population size for DE
        self.f = 0.8  # DE mutation factor
        self.cr = 0.9  # DE crossover probability
        self.local_search_iters = 5  # Number of local search iterations
        self.bounds = None

    def de_step(self, population, scores):
        new_population = np.copy(population)
        for i in range(self.pop_size):
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), self.bounds.lb, self.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            new_individual = np.where(cross_points, mutant, population[i])
            new_population[i] = new_individual

        return new_population

    def local_search(self, individual):
        best = individual
        best_score = self.evaluate(best)
        initial_iters = self.local_search_iters

        for _ in range(initial_iters):
            candidate = best + np.random.normal(0, 0.01, self.dim)
            candidate = np.clip(candidate, self.bounds.lb, self.bounds.ub)
            candidate_score = self.evaluate(candidate)
            if candidate_score > best_score:
                best, best_score = candidate, candidate_score
            else:
                self.local_search_iters += 1  # Adaptive adjustment

        return best

    def evaluate(self, individual):
        return self.func(individual)

    def __call__(self, func):
        self.func = func
        self.bounds = func.bounds
        population = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.pop_size, self.dim))
        scores = np.array([self.evaluate(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            self.f = 0.9 - 0.5 * (evaluations / self.budget)  # Adaptive DE mutation factor
            new_population = self.de_step(population, scores)
            new_scores = np.array([self.evaluate(ind) for ind in new_population])
            evaluations += self.pop_size

            for i in range(self.pop_size):
                if new_scores[i] > scores[i]:
                    population[i], scores[i] = new_population[i], new_scores[i]

            best_idx = np.argmax(scores)
            best_individual = population[best_idx]

            if evaluations + self.local_search_iters <= self.budget:
                refined_best = self.local_search(best_individual)
                refined_score = self.evaluate(refined_best)
                evaluations += self.local_search_iters

                if refined_score > scores[best_idx]:
                    population[best_idx] = refined_best
                    scores[best_idx] = refined_score

        return population[np.argmax(scores)]