import numpy as np

class HybridPSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 1.49  # cognitive coefficient
        self.c2 = 1.49  # social coefficient
        self.w = 0.72   # inertia weight
        self.F = 0.8    # DE mutation factor
        self.CR = 0.9   # DE crossover rate
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        np.random.seed(0)
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - pop[i]) +
                                 self.c2 * r2 * (global_best_position - pop[i]))
                pop[i] = np.clip(pop[i] + velocities[i], self.bounds[0], self.bounds[1])
            
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                F_adapt = self.F + 0.2 * np.random.rand()  # Adaptive F
                mutant = np.clip(a + F_adapt * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                score_trial = func(trial)
                evaluations += 1
                if score_trial < personal_best_scores[i]:
                    personal_best_scores[i] = score_trial
                    personal_best_positions[i] = trial
                    if score_trial < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = trial
                if evaluations >= self.budget:
                    break
        
        return global_best_position, personal_best_scores[global_best_index]