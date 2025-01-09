import numpy as np

class PSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w = 0.729
        self.f = 0.5
        self.cr = 0.9
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize the swarm
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        V = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        p_best = np.copy(X)
        p_best_score = np.array([func(x) for x in X])
        g_best = p_best[np.argmin(p_best_score)]
        g_best_score = np.min(p_best_score)
        
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                # Update velocities and positions (PSO step)
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                V[i] = self.w * V[i] + self.c1 * r1 * (p_best[i] - X[i]) + self.c2 * r2 * (g_best - X[i])
                X[i] += V[i]
                X[i] = np.clip(X[i], self.lb, self.ub)

                # Evaluate the function
                score = func(X[i])
                evals += 1

                # Update personal bests
                if score < p_best_score[i]:
                    p_best[i] = X[i]
                    p_best_score[i] = score

            # Update global best
            if evals < self.budget:
                current_g_best = p_best[np.argmin(p_best_score)]
                current_g_best_score = np.min(p_best_score)
                if current_g_best_score < g_best_score:
                    g_best = current_g_best
                    g_best_score = current_g_best_score

            # Adaptive Differential Evolution step
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = X[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, X[i])
                
                # Evaluate the trial vector
                trial_score = func(trial)
                evals += 1

                # Selection
                if trial_score < p_best_score[i]:
                    X[i] = trial
                    p_best[i] = trial
                    p_best_score[i] = trial_score

            # Update global best after the DE step
            if evals < self.budget:
                current_g_best = p_best[np.argmin(p_best_score)]
                current_g_best_score = np.min(p_best_score)
                if current_g_best_score < g_best_score:
                    g_best = current_g_best
                    g_best_score = current_g_best_score

        return g_best