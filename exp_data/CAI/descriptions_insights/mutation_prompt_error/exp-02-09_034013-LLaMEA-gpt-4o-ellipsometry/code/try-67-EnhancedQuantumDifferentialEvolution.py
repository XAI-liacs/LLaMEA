import numpy as np

class EnhancedQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.q_prob = 0.1  # Probability to use quantum-inspired move

    def quantum_move(self, individual, best, lb, ub):
        factor = np.random.uniform(0.5, 1.5)
        new_position = best + factor * (individual - best)
        return np.clip(new_position, lb, ub)

    def adaptive_mutation(self):
        return 0.5 + np.random.rand() * 0.3  # Adaptive mutation factor

    def levy_flight(self, individual, lb, ub):
        step = np.random.normal(0, 0.1, self.dim) * (individual - lb)
        new_position = individual + step
        return np.clip(new_position, lb, ub)

    def quantum_crossover(self, individual, best):
        return np.where(np.random.rand(self.dim) < self.q_prob, best, individual)

    def rank_based_selection(self, scores, population, num_select):
        ranked_indices = np.argsort(scores)
        selected_indices = ranked_indices[:num_select]
        return population[selected_indices]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            num_select = max(1, int(self.population_size * 0.3))
            selected_population = self.rank_based_selection(scores, population, num_select)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation with adaptive rate
                a, b, c = np.random.choice(selected_population.shape[0], 3, replace=False)
                self.F = self.adaptive_mutation()
                mutant = selected_population[a] + self.F * (selected_population[b] - selected_population[c])
                mutant = np.clip(mutant, lb, ub)

                # Quantum-inspired crossover
                trial = self.quantum_crossover(mutant, global_best_position)

                # Lévy flight exploration
                if np.random.rand() < 0.2:
                    trial = self.levy_flight(trial, lb, ub)

                # Selection
                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

                # Quantum-inspired move with a small probability
                if np.random.rand() < self.q_prob:
                    population[i] = self.quantum_move(population[i], global_best_position, lb, ub)

                # Update global best
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = population[i].copy()

        return global_best_position, global_best_score