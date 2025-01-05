import numpy as np

class HybridDE_SA_Adaptive:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.8, CR=0.9, T_init=1000, alpha=0.99):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_F = F
        self.CR = CR
        self.T = T_init
        self.alpha = alpha
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        evals = self.pop_size
        
        while evals < self.budget:
            F = self.initial_F * (1 - evals / self.budget)  # Adaptive F
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Differential Evolution mutation
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                evals += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

            # Simulated Annealing-inspired perturbation
            for i in range(self.pop_size):
                perturbed = pop[i] + np.random.normal(0, self.T, self.dim)
                perturbed = np.clip(perturbed, func.bounds.lb, func.bounds.ub)
                perturbed_fitness = func(perturbed)
                evals += 1

                # Acceptance probability now considers a logarithmic temperature decrease
                if (perturbed_fitness < fitness[i] or 
                    np.exp(-(perturbed_fitness - fitness[i]) / (self.T * np.log1p(evals/self.pop_size))) > np.random.rand()):
                    pop[i] = perturbed
                    fitness[i] = perturbed_fitness
                    if perturbed_fitness < self.f_opt:
                        self.f_opt = perturbed_fitness
                        self.x_opt = perturbed

            # Update temperature
            self.T *= self.alpha
        
        return self.f_opt, self.x_opt