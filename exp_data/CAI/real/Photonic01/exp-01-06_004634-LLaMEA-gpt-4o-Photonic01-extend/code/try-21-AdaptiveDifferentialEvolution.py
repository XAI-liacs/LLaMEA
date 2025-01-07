class AdaptiveDifferentialEvolution:
    # ... (previous code remains unchanged) ...

    def select(self, target_idx, trial, func):
        target = self.population[target_idx]
        trial_fitness = func(trial)
        target_fitness = func(target)
        self.evaluation_count += 2
        if trial_fitness < target_fitness:
            self.mutation_factor = min(1.0, self.mutation_factor + 0.05)  # Slightly increased increment
            return trial
        else:
            self.mutation_factor = max(0.4, self.mutation_factor - 0.05)  # Slightly increased decrement
            return target

    # ... (subsequent code remains unchanged) ...