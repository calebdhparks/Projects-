from population import Population


def main():
    pop_size = [200,200,100,10000]
    target = "To be or not to be."
    mutation_rate = [.01,.02,.01,.01]
    runs = 20

    # you don't need to call this function when the ones right bellow are fully implemented
    # pop.print_population_status()

    avg_fitness=[]
    avg_gen=[]

    # Uncomment these lines bellow when you implement all the functions
    for x in range(len(pop_size)):
        fitness=0
        gen=0
        for _ in range(runs):
            pop = Population(target, pop_size[x], mutation_rate[x])
            while not pop.finished:
                pop.natural_selection()
                pop.generate_new_population()
                pop.evaluate()
                pop.print_population_status()
            hist=pop.getGen()
            fitness+=hist[1]
            gen+=hist[0]
        avg_gen.append(gen/runs)
        avg_fitness.append(fitness/runs)
    print("Average Fitness per population size",avg_fitness,"\nAverage generation per population size",avg_gen)

# [0.9469605263157883, 0.9438289473684197, 0.9477894736842092, 0.900229999999896] [480.6, 296.3, 1012.5, 72.35]

if __name__ == "__main__":
    main()
