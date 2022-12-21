import graphviz
import matplotlib.pyplot as plt
import numpy as np
import warnings


def plot_stats(statistics, ylog=False, view=False, filename="avg_fitness.svg"):
    """Plots the population's average and best fitness."""
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)"
        )
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, "b-", label="average")
    # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, "g-.", label="+1 sd")
    plt.plot(generation, best_fitness, "r-", label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale("symlog")

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_species(statistics, view=False, filename="speciation.svg"):
    """Visualizes speciation throughout evolution."""
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)"
        )
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(
    config,
    genome,
    view=False,
    filename=None,
    node_names=None,
    show_disabled=True,
    prune_unused=False,
    node_colors=None,
    fmt="svg",
):
    """Receives a genome and draws a neural network with arbitrary topology."""

    # TEMPORARY UNTIL NEW VERSION
    import copy
    import neat
    from neat.graphs import required_for_output

    def get_pruned_copy(genome, genome_config):
        used_node_genes, used_connection_genes = get_pruned_genes(
            genome.nodes,
            genome.connections,
            genome_config.input_keys,
            genome_config.output_keys,
        )
        new_genome = neat.DefaultGenome(None)
        new_genome.nodes = used_node_genes
        new_genome.connections = used_connection_genes
        return new_genome

    def get_pruned_genes(node_genes, connection_genes, input_keys, output_keys):
        used_nodes = required_for_output(input_keys, output_keys, connection_genes)
        used_pins = used_nodes.union(input_keys)

        # Copy used nodes into a new genome.
        used_node_genes = {}
        for n in used_nodes:
            used_node_genes[n] = copy.deepcopy(node_genes[n])

        # Copy enabled and used connections into the new genome.
        used_connection_genes = {}
        for key, cg in connection_genes.items():
            in_node_id, out_node_id = key
            if cg.enabled and in_node_id in used_pins and out_node_id in used_pins:
                used_connection_genes[key] = copy.deepcopy(cg)

        return used_node_genes, used_connection_genes

    # END

    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (graphviz)"
        )
        return

    # If requested, use a copy of the genome which omits all components that
    # won't affect the output.
    if prune_unused:
        genome = get_pruned_copy(genome, config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {"shape": "circle", "fontsize": "20", "height": "1.0", "width": "0.5"}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {
            "style": "filled",
            "shape": "rectangle",
            "fillcolor": node_colors.get(k, "lightgreen"),
        }
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {"style": "filled", "fillcolor": node_colors.get(k, "lightblue")}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {"style": "filled", "fillcolor": node_colors.get(n, "pink")}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = "solid" if cg.enabled else "dotted"
            color = "green" if cg.weight > 0 else "red"
            width = str(0.5 + abs(cg.weight / 5.0))
            dot.edge(
                a, b, _attributes={"style": style, "color": color, "penwidth": width}
            )

    dot.render(filename, view=view)

    return dot
