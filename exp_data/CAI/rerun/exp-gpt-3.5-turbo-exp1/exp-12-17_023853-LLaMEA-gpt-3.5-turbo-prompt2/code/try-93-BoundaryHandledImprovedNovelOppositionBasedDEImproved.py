class BoundaryHandledImprovedNovelOppositionBasedDEImproved(ImprovedNovelOppositionBasedDEImproved):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def clip_to_bounds(solution):
            return np.clip(solution, -5.0, 5.0)

        def generate_population():
            return np.random.uniform(-5.0, 5.0, size=(self.NP, self.dim))

        population = generate_population()
        best_solution = clip_to_bounds(population[np.argmin([func(ind) for ind in population])])
        
        for _ in range(self.budget):
            trial_population = []
            for i in range(self.NP):
                idxs = [idx for idx in range(self.NP) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                strategy = np.random.choice([0, 1, 2, 3, 4, 5, 6])  # Updated mutation strategies including the novel opposition-based mutation

                F = self.F * np.exp(-_ / self.budget) * (1 + np.random.normal(0, 1) * 0.1)  # Dynamic adjustment of F with budget and adaptive mutation scale
                CR = self.CR * np.exp(-_ / self.budget)  # Dynamic adjustment of CR with budget
                
                if strategy == 0:
                    mutant = clip_to_bounds(population[a] + F * (population[b] - population[c]))
                elif strategy == 1:
                    mutant = clip_to_bounds(population[a] + F * (population[b] - population[c]) + F * (population[a] - best_solution))
                elif strategy == 2:
                    mutant = clip_to_bounds(best_solution + F * (population[b] - population[c]))
                elif strategy == 3:
                    mutant = clip_to_bounds(best_solution + F * (population[a] - best_solution) + F * (population[b] - population[c]))
                elif strategy == 4:
                    mutant = clip_to_bounds(population[a] + F * (population[b] - population[c]) + F * (population[a] - best_solution) + F * (best_solution - population[c]))
                elif strategy == 5:
                    opposite = 2 * best_solution - population[i]
                    mutant = clip_to_bounds(best_solution + F * (opposite - population[a]))
                else:  # New mutation strategy for opposition-based learning with adaptive control parameters
                    opposite = 2 * best_solution - population[i]
                    mutant = clip_to_bounds(best_solution + F * (opposite - population[a]) + F * (population[b] - best_solution) + F * (best_solution - population[c]))

                j_rand = np.random.randint(self.dim)
                trial_ind = np.array([mutant[j] if np.random.rand() < CR or j == j_rand else population[i, j] for j in range(self.dim)])

                if func(trial_ind) < func(population[i]):
                    trial_population.append(trial_ind)
                else:
                    trial_population.append(population[i])

            population = np.array(trial_population)
            best_solution = clip_to_bounds(population[np.argmin([func(ind) for ind in population])])

        return best_solution