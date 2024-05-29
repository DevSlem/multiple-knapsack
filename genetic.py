import random
import time
import argparse
from util import make_directory, load_knapsack_problem
import json

class GeneticAlgorithm:
    def __init__(self, num_items, num_knapsacks, capacities, values, weights, population_size=200, crossover_rate=0.7, mutation_rate=0.05, selection_pressure=0.5):
        self.num_items = num_items
        self.num_knapsacks = num_knapsacks
        self.capacities = capacities
        self.values = values
        self.weights = weights
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.population = [self.random_solution() for _ in range(population_size)]
        self.fitness_cache = {}

    def random_solution(self):
        return [random.randint(-1, self.num_knapsacks - 1) for _ in range(self.num_items)]

    def calculate_fitness(self, solution):
        if tuple(solution) in self.fitness_cache:
            return self.fitness_cache[tuple(solution)]
        
        total_value = 0
        knapsack_weights = [0] * self.num_knapsacks
        for item_index, knapsack_index in enumerate(solution):
            if knapsack_index == -1:
                continue
            if knapsack_weights[knapsack_index] + self.weights[item_index] <= self.capacities[knapsack_index]:
                knapsack_weights[knapsack_index] += self.weights[item_index]
                total_value += self.values[item_index]

        self.fitness_cache[tuple(solution)] = total_value
        return total_value

    def tournament_selection(self):
        selected = []
        for _ in range(2):
            tournament = random.sample(self.population, k=int(self.population_size * self.selection_pressure))
            selected.append(max(tournament, key=self.calculate_fitness))
        return selected

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.num_items - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        else:
            return parent1[:], parent2[:]

    def mutate(self, solution):
        for i in range(self.num_items):
            if random.random() < self.mutation_rate:
                solution[i] = random.randint(-1, self.num_knapsacks - 1)
        return solution

    def generate_new_population(self):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self.tournament_selection()
            child1, child2 = self.crossover(parent1, parent2)
            new_population.extend([self.mutate(child1), self.mutate(child2)])
        self.population = new_population[:self.population_size]

    def run(self, generations=1000):
        for generation in range(generations):
            self.generate_new_population()
            best_solution = max(self.population, key=self.calculate_fitness)
            print(f"Generation {generation}: Best Fitness = {self.calculate_fitness(best_solution)}")
        return self.calculate_fitness(best_solution), best_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str)
    args = parser.parse_args()
    for idx in range(1, 11):
        problem_name = args.problem_name + str(idx)

        knapsack_df, item_df = load_knapsack_problem(problem_name)
        capacities = knapsack_df['capacity'].values
        values = item_df['value'].values
        weights = item_df['weight'].values

        start = time.time()
        ga = GeneticAlgorithm(len(values), len(capacities), capacities, values, weights)
        fit, gen = ga.run()
        train_time = time.time() - start
        
        result_dict = dict()
        result_dict["method"] = "GA"
        result_dict["n_knapsacks"] = len(knapsack_df)
        result_dict["n_items"] = len(item_df)
        result_dict["time"] = train_time
        result_dict["solution"] = {
            "total_value": int(fit)
        }

        directory = f"results/{problem_name}/ga"
        make_directory(directory)
        with open(f"{directory}/result.json", 'w') as f:
            json.dump(result_dict, f, indent=4)
