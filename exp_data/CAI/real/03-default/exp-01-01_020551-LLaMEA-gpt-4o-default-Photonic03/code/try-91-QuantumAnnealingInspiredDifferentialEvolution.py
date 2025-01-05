import numpy as np

class QuantumAnnealingInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim * 5)
        self.initial_cr = 0.9  # Crossover probability
        self.initial_f = 0.8   # Differential weight
        self.adaptive_rate = 0.01  # Adaptive rate for tuning parameters
        self.temperature = 1.0  # Initial temperature for simulated annealing
        self.alpha = 0.95  # Cooling rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(scores)
        global_best_position = population[global_best_index].copy()
        evaluations = self.population_size
        cr = self.initial_cr
        f = self.initial_f

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + f * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < cr
                trial = np.where(crossover, mutant, population[i])
                
                trial_score = func(trial)
                evaluations += 1
                
                if trial_score < scores[i] or np.exp((scores[i] - trial_score) / self.temperature) > np.random.rand():
                    population[i] = trial
                    scores[i] = trial_score
                    if trial_score < scores[global_best_index]:
                        global_best_index = i
                        global_best_position = trial.copy()

            self.temperature *= self.alpha  # Cool down
            cr = self.initial_cr - self.adaptive_rate * (evaluations / self.budget)
            f = self.initial_f + self.adaptive_rate * (evaluations / self.budget)

        return global_best_position, scores[global_best_index]