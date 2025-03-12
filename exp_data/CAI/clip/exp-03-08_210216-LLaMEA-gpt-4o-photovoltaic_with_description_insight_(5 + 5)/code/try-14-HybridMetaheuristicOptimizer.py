import numpy as np

try:
    from cma import CMAEvolutionStrategy
except ModuleNotFoundError:
    CMAEvolutionStrategy = None

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.de_mutation_factor = 0.8
        self.de_crossover_prob = 0.9
        self.current_evaluations = 0

    def differential_evolution(self, func, pop, bounds):
        new_pop = np.copy(pop)
        for i in range(len(pop)):
            if self.current_evaluations >= self.budget:
                break
            # Randomly select three other individuals
            indices = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[indices]
            # Create a mutant vector
            mutant = a + self.de_mutation_factor * (b - c)
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            # Crossover
            crossover_mask = np.random.rand(self.dim) < self.de_crossover_prob
            offspring = np.where(crossover_mask, mutant, pop[i])
            # Selection
            if self.current_evaluations < self.budget:
                if func(offspring) < func(pop[i]):
                    new_pop[i] = offspring
                self.current_evaluations += 1
        return new_pop

    def cma_es_refinement(self, func, x0, bounds):
        if self.current_evaluations >= self.budget or CMAEvolutionStrategy is None:
            return x0
        es = CMAEvolutionStrategy(x0, 0.5, {'bounds': [bounds.lb, bounds.ub], 'popsize': self.population_size})
        while not es.stop() and self.current_evaluations < self.budget:
            solutions = es.ask()
            fitnesses = [func(sol) for sol in solutions]
            self.current_evaluations += len(solutions)
            es.tell(solutions, fitnesses)
            es.disp()
        return es.result[0]

    def adaptive_layer_refinement(self, func, bounds):
        layers = 10
        best_solution = np.random.uniform(bounds.lb, bounds.ub, self.dim)
        while layers <= self.dim and self.current_evaluations < self.budget:
            # DE phase
            pop = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, layers))
            pop = self.differential_evolution(func, pop, bounds)
            best_solution = pop[np.argmin([func(ind) for ind in pop])]
            
            # CMA-ES phase
            best_solution = self.cma_es_refinement(func, best_solution, bounds)
            
            # Increase complexity by adding layers
            layers = min(layers + 5, self.dim)
        
        return best_solution

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.adaptive_layer_refinement(func, bounds)
        return best_solution