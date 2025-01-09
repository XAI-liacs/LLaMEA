import numpy as np

class PSO_DE:
    def __init__(self, budget=10000, dim=10, swarm_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.F = F  # Scaling factor for differential evolution
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        np.random.seed()
        # Initialize swarm positions and velocities
        X = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        V = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
        P = X.copy()  # Personal best positions
        P_fitness = np.array([func(x) for x in X])
        G = X[np.argmin(P_fitness)]  # Global best position
        G_fitness = np.min(P_fitness)
        
        evaluations = self.swarm_size
        w = 0.9  # Inertia weight

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Differential Evolution mutation
                indices = np.random.choice(self.swarm_size, 3, replace=False)
                X_r1, X_r2, X_r3 = X[indices]
                mutant = X_r1 + self.F * (X_r2 - X_r3)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Crossover
                mask = np.random.rand(self.dim) < self.CR
                trial = np.where(mask, mutant, X[i])
                
                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < P_fitness[i]:
                    P[i] = trial
                    P_fitness[i] = trial_fitness
                    
                    if trial_fitness < G_fitness:
                        G = trial
                        G_fitness = trial_fitness
            
            # Update particle velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                w = 0.9 - 0.5 * (evaluations / self.budget)  # Adaptive inertia weight
                V[i] = w * V[i] + 2.05 * r1 * (P[i] - X[i]) + 2.05 * r2 * (G - X[i])
                X[i] = np.clip(X[i] + V[i], self.lb, self.ub)

                # Evaluate new position
                f = func(X[i])
                evaluations += 1

                # Update personal best
                if f < P_fitness[i]:
                    P[i] = X[i]
                    P_fitness[i] = f

                    # Update global best
                    if f < G_fitness:
                        G = X[i]
                        G_fitness = f

            # Elitism: Retain the best solution found
            if G_fitness < self.f_opt:
                self.f_opt = G_fitness
                self.x_opt = G

        return self.f_opt, self.x_opt