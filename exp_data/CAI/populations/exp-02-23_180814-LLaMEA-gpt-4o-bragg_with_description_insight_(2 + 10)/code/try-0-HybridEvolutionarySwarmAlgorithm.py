import numpy as np

class HybridEvolutionarySwarmAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w = 0.5  # Inertia weight

    def initialize_population(self, lb, ub):
        return lb + np.random.rand(self.population_size, self.dim) * (ub - lb)

    def mutate(self, target_idx, pop):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), 0, 1)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, target)
        return trial

    def optimize(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.initialize_population(lb, ub)
        velocities = np.random.rand(self.population_size, self.dim) * (ub - lb)
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                mutant = self.mutate(i, pop)
                trial = self.crossover(pop[i], mutant)
                trial_score = func(trial)

                if trial_score < personal_best_scores[i]:
                    personal_best[i] = trial
                    personal_best_scores[i] = trial_score

                    if trial_score < global_best_score:
                        global_best = trial
                        global_best_score = trial_score

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]))
                pop[i] = np.clip(pop[i] + velocities[i], lb, ub)

            # Encourage periodicity by modifying fitness function
            if func.name == 'bragg':
                pop = self.apply_periodicity_constraint(pop)

        return global_best

    def apply_periodicity_constraint(self, pop):
        # Enforce periodicity in the solution
        for i in range(self.population_size):
            pop[i] = np.tile(np.mean(pop[i].reshape(-1, 2), axis=0), self.dim // 2)
        return pop

    def __call__(self, func):
        best_solution = self.optimize(func)
        return best_solution

# Example usage:
# optimizer = HybridEvolutionarySwarmAlgorithm(budget=1000, dim=20)
# best_solution = optimizer(func)