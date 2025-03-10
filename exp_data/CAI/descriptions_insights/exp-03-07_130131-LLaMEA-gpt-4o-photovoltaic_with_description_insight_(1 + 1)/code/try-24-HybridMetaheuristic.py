import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = self.dim * 10
        self.f_min = 0.5
        self.f_max = 0.9
        self.cr = 0.9
        self.local_search_iters = 5
        self.bounds = None
        self.conv_rate_thresh = 0.01

    def de_step(self, population, scores):
        new_population = np.copy(population)
        diversity = np.std(population, axis=0).mean()
        cr_adaptive = max(0.1, self.cr * diversity)  # Change 1
        for i in range(self.pop_size):
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            f = self.f_min + (self.f_max - self.f_min) * (scores[i] / np.max(scores))
            mutant = np.clip(a + f * (b - c), self.bounds.lb, self.bounds.ub)
            cross_points = np.random.rand(self.dim) < cr_adaptive  # Change 2
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            new_individual = np.where(cross_points, mutant, population[i])
            new_population[i] = new_individual

        return new_population

    def local_search(self, individual):
        best = individual
        best_score = self.evaluate(best)
        temperature = 1.0  # Simulated annealing initial temperature

        for _ in range(self.local_search_iters):
            candidate = best + np.random.normal(0, 0.01, self.dim)
            candidate = np.clip(candidate, self.bounds.lb, self.bounds.ub)
            candidate_score = self.evaluate(candidate)
            delta_score = candidate_score - best_score
            if delta_score > 0 or np.exp(delta_score / temperature) > np.random.rand():
                best, best_score = candidate, candidate_score
            temperature *= 0.9  # Cooling schedule for simulated annealing

        return best

    def evaluate(self, individual):
        return self.func(individual)

    def crowding_distance(self, population, scores):
        distances = np.zeros(self.pop_size)
        index = np.argsort(scores)
        for i in range(1, self.pop_size - 1):
            distances[index[i]] = scores[index[i + 1]] - scores[index[i - 1]]
        return distances

    def __call__(self, func):
        self.func = func
        self.bounds = func.bounds
        population = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.pop_size, self.dim))
        scores = np.array([self.evaluate(ind) for ind in population])
        evaluations = self.pop_size

        last_best_score = -np.inf
        while evaluations < self.budget:
            new_population = self.de_step(population, scores)
            new_scores = np.array([self.evaluate(ind) for ind in new_population])
            evaluations += self.pop_size

            distances = self.crowding_distance(new_population, new_scores)
            for i in range(self.pop_size):
                if new_scores[i] > scores[i] or distances[i] > distances[np.argmax(scores)]:
                    population[i], scores[i] = new_population[i], new_scores[i]

            best_idx = np.argmax(scores)
            best_individual = population[best_idx]

            if evaluations + self.local_search_iters <= self.budget:
                current_best_score = scores[best_idx]
                conv_rate = (current_best_score - last_best_score) / abs(last_best_score + 1e-9)
                if conv_rate < self.conv_rate_thresh:
                    refined_best = self.local_search(best_individual)
                    refined_score = self.evaluate(refined_best)
                    evaluations += self.local_search_iters

                    if refined_score > scores[best_idx]:
                        population[best_idx] = refined_best
                        scores[best_idx] = refined_score

                last_best_score = current_best_score

        return population[np.argmax(scores)]