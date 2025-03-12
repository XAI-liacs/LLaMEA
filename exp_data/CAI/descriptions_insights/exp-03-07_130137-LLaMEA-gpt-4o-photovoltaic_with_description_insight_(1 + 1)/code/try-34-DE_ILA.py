import numpy as np

class DE_ILA:
    def __init__(self, budget, dim, pop_size=50, F=0.8, CR=0.9, initial_layers=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.initial_layers = initial_layers
        self.evaluations = 0
        self.adaptive_F = F  # Adaptive differential weight
        self.adaptive_CR = CR  # Adaptive crossover probability

    def _initialize_population(self, bounds, layers):
        pop = (np.random.rand(self.pop_size, layers) ** 2) * (bounds.ub[:layers] - bounds.lb[:layers]) + bounds.lb[:layers]
        return pop

    def _mutate(self, target_idx, pop):
        indices = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = pop[a] + self.adaptive_F * (pop[b] - pop[c])
        return np.clip(mutant, 0, 1)

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(len(target)) < self.adaptive_CR
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
            
            # Adaptive strategy
            successful_trials = scores < np.min(scores) + np.std(scores)
            if np.any(successful_trials):
                self.adaptive_F = np.mean(scores[successful_trials]) / best_score
                self.adaptive_CR = 0.5 * (1.0 + self.adaptive_CR)

            pop = next_pop
            
            # Elitism
            pop[np.argmax(scores)] = best_solution

            if current_layers < self.dim and self.evaluations < self.budget:
                current_layers += 1

        return best_solution