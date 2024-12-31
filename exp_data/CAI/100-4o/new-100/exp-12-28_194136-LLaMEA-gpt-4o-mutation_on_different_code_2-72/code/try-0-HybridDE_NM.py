import numpy as np

class HybridDE_NM:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight

    def nelder_mead(self, simplex, func):
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        for _ in range(self.dim):
            simplex.sort(key=lambda vertex: vertex[1])
            centroid = np.mean([x for x, _ in simplex[:-1]], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1][0])
            xr_val = func(xr)
            if simplex[0][1] <= xr_val < simplex[-2][1]:
                simplex[-1] = (xr, xr_val)
            elif xr_val < simplex[0][1]:
                xe = centroid + gamma * (xr - centroid)
                xe_val = func(xe)
                simplex[-1] = (xe, xe_val) if xe_val < xr_val else (xr, xr_val)
            else:
                xc = centroid + rho * (simplex[-1][0] - centroid)
                xc_val = func(xc)
                if xc_val < simplex[-1][1]:
                    simplex[-1] = (xc, xc_val)
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = (simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0]), 
                                      func(simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])))
        return simplex[0]

    def __call__(self, func):
        # Initialize a population
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]
        population_fitness = [func(ind) for ind in population]
        evals = self.population_size
        
        while evals < self.budget:
            new_population = []
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Mutation
                mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
                
                # Crossover
                trial = np.array([mutant[j] if np.random.rand() < self.CR else population[i][j] for j in range(self.dim)])
                
                # Selection
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < population_fitness[i]:
                    new_population.append(trial)
                    population_fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])
                
                # Update global best
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial
            
            # Local search phase with Nelder-Mead
            if evals < self.budget:
                simplex = [(ind, fit) for ind, fit in zip(new_population[:self.dim+1], population_fitness[:self.dim+1])]
                best_vertex = self.nelder_mead(simplex, func)
                if best_vertex[1] < self.f_opt:
                    self.f_opt = best_vertex[1]
                    self.x_opt = best_vertex[0]
                    evals += len(simplex)
            population = new_population
        
        return self.f_opt, self.x_opt