import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.c1 = 1.3  # Cognitive component (changed)
        self.c2 = 1.5  # Social component
        self.inertia = 0.8  # Inertia weight
        self.f = 0.6  # Differential Evolution scaling factor (changed)
        self.cr = 0.75  # Crossover probability
        self.evaluations = 0

    def __call__(self, func):
        # Initialize particle positions and velocities
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        positions = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.pop_size, np.inf)
        
        # Evaluate initial population
        global_best_position = None
        global_best_score = np.inf

        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            score = func(positions[i])
            self.evaluations += 1
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]
            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[i]

        # Main loop
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # PSO Update
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * (1 + 0.7 * np.sin(np.pi * self.evaluations / self.budget)) * r2 * (global_best_position - positions[i]))  # Nonlinear adaptation
                velocities[i] = np.clip(velocities[i], -0.5 * (ub - lb), 0.5 * (ub - lb))  # Adaptive velocity clamping
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                # DE Mutation and Crossover
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = positions[indices]
                mutant = np.clip(a + self.f * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, positions[i])

                # Evaluate new solution
                if self.evaluations < self.budget:
                    score = func(trial)
                    self.evaluations += 1

                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = trial
                    
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = trial

            # Adaptive mechanism
            self.inertia = max(0.2, 0.8 - 0.6 * (self.evaluations / self.budget))  # Nonlinear inertia decrease

        return global_best_position