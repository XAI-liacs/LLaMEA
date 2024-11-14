import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.crossover_rate = 0.85  # Slightly reduced crossover rate
        self.mutation_factor = 0.9  # Increased mutation factor for stronger exploration
        self.population_scaling = 0.4 + (0.4 * np.random.rand())  # Wider range for dynamic scaling
        self.population_size = int(self.budget / (7 * dim))  # Adjusted population size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.greedy_step_probability = 0.3  # Added greedy step probability
        self.max_local_search_iterations = 4  # Further increased local search iterations
        self.local_search_probability = 0.5  # Increased local search probability

    def _evaluate_population(self, func):
        return np.array([func(ind) for ind in self.population])

    def _mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_mutation_factor = self.mutation_factor * (1 + np.random.uniform(-0.15, 0.15))  # More adaptive mutation
        mutant_vector = self.population[a] + self.population_scaling * adaptive_mutation_factor * (
            self.population[b] - self.population[c])
        mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
        return mutant_vector
    
    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def _stochastic_local_search(self, solution, func, max_iters):
        best_solution = solution.copy()
        best_score = func(best_solution)
        for _ in range(max_iters):
            perturbation = np.random.normal(0.0, 0.1, self.dim)
            candidate_solution = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_score = func(candidate_solution)
            if candidate_score < best_score:
                best_solution = candidate_solution
                best_score = candidate_score
        return best_solution

    def _greedy_step(self, func, current_solution, current_score):
        step_size = 0.1
        greedy_solution = current_solution + step_size * np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        greedy_solution = np.clip(greedy_solution, self.lower_bound, self.upper_bound)
        greedy_score = func(greedy_solution)
        return (greedy_solution, greedy_score) if greedy_score < current_score else (current_solution, current_score)

    def __call__(self, func):
        population_scores = self._evaluate_population(func)
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self._mutate(i)
                trial = self._crossover(target, mutant)
                trial_score = func(trial)
                evaluations += 1

                if trial_score < population_scores[i]:
                    self.population[i] = trial
                    population_scores[i] = trial_score

                    if np.random.rand() < self.local_search_probability:
                        improved_solution = self._stochastic_local_search(trial, func, self.max_local_search_iterations)
                        improved_score = func(improved_solution)
                        evaluations += 1

                        if improved_score < trial_score:
                            self.population[i] = improved_solution
                            population_scores[i] = improved_score

                if np.random.rand() < self.greedy_step_probability:
                    greedy_solution, greedy_score = self._greedy_step(func, self.population[i], population_scores[i])
                    if greedy_score < population_scores[i]:
                        self.population[i] = greedy_solution
                        population_scores[i] = greedy_score

        best_index = np.argmin(population_scores)
        return self.population[best_index]