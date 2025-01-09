import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.temperature = 1.0  # Initial temperature for Simulated Annealing
        self.cooling_rate = 0.95  # Cooling rate for temperature

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Differential Evolution
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                x = population[i]
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, x)
                
                # Evaluate new trial individual
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fitness

            # Simulated Annealing refinement
            best_idx = np.argmin(fitnesses)
            best_solution = population[best_idx]
            best_fitness = fitnesses[best_idx]

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                candidate = population[i]
                # Perturb solution
                perturbation = np.random.normal(0, 1, self.dim) * self.temperature
                new_candidate = np.clip(candidate + perturbation, self.lower_bound, self.upper_bound)
                
                # Evaluate new candidate
                new_fitness = func(new_candidate)
                evaluations += 1

                # Accept new candidate with Metropolis criterion
                if new_fitness < fitnesses[i] or np.random.rand() < np.exp((fitnesses[i] - new_fitness) / self.temperature):
                    population[i] = new_candidate
                    fitnesses[i] = new_fitness

            # Reduce temperature
            self.temperature *= self.cooling_rate

        return best_solution, best_fitness