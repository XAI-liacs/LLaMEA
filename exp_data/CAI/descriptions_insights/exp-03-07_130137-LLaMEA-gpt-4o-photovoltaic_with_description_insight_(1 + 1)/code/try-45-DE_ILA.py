import numpy as np

class DE_ILA:
    def __init__(self, budget, dim, pop_size=50, F=0.8, CR=0.9, initial_layers=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.initial_layers = initial_layers  # Start with a smaller number of layers
        self.evaluations = 0

    def _initialize_population(self, bounds, layers):
        pop = (np.random.rand(self.pop_size, layers) ** 2) * (bounds.ub[:layers] - bounds.lb[:layers]) + bounds.lb[:layers]
        return pop

    def _mutate(self, target_idx, pop):
        indices = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.F * (1 - self.evaluations / self.budget)  # Adaptive F
        mutant = pop[a] + adaptive_F * (pop[b] - pop[c]) + 0.1 * (np.random.rand(len(pop[a])) - 0.5)
        return np.clip(mutant, 0, 1)

    def _crossover(self, target, mutant):
        adaptive_CR = self.CR * (1 - np.std(target))  # Adaptive CR
        cross_points = np.random.rand(len(target)) < adaptive_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, len(target))] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        bounds = func.bounds
        current_layers = self.initial_layers
        best_solution = None
        best_score = float("inf")

        while self.evaluations < self.budget:
            if best_solution is None:
                pop = self._initialize_population(bounds, current_layers)
            else:
                new_pop = np.random.rand(self.pop_size, current_layers) * (bounds.ub[:current_layers] - bounds.lb[:current_layers]) + bounds.lb[:current_layers]
                new_pop[:, :current_layers-len(best_solution)] = best_solution[:current_layers-len(best_solution)]
                pop = new_pop

            scores = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                solution = pop[i]
                score = func(solution)
                self.evaluations += 1
                scores[i] = score
                if score < best_score:
                    best_score = score
                    best_solution = solution

            next_pop = np.copy(pop)
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                target = pop[i]
                mutant = self._mutate(i, pop)
                trial = self._crossover(target, mutant)
                trial_real = trial * (bounds.ub[:current_layers] - bounds.lb[:current_layers]) + bounds.lb[:current_layers]
                trial_score = func(trial_real)
                self.evaluations += 1
                if trial_score < scores[i]:
                    next_pop[i] = trial
                    scores[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        best_solution = trial_real

            pop = next_pop

            if current_layers < self.dim and self.evaluations < self.budget:
                current_layers += 1

        return best_solution