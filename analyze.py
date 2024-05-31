import sys
import getopt
from itertools import starmap
import matplotlib.pyplot as plt
from ingress import Ingress, Portal, Link, Field, Tree
from typing import Callable
from render import plot_IITC_elements, plot_portals
        
# TODO: opportunity to speed things up with concurency
def generate_trees(axis0: list[Field], axis1: list[Callable]) -> list[list]:
    result = []
    for field in axis0:
        result.append(list(starmap(Tree, ((root, split) for root, split in zip([Field(*field.portals, field.level) for _ in range(len(axis1))], axis1)))))

    # making sure each tree's root is unique
    ids = [tuple(map(lambda t: id(t.get_root), row) for row in result)]
    flat_ids = Ingress.flatten_iterable_of_tuples(ids)
    assert len(flat_ids) == len(set(flat_ids)), f"ids of tree roots are NOT unique, all trees: {len(flat_ids)} unique ids: {len(set(flat_ids))}"

    return result

def plot_input_analysis(x_labels, arrays, subplot_titles):
        plt.figure(figsize=(16, 9))
        # Determine the number of subplots needed
        num_arrays = len(arrays)
        num_cols = 2  # Adjust the number of columns as needed
        num_rows = (num_arrays + num_cols - 1) // num_cols

        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 9))

        # Flatten the axes array if there are multiple rows
        if num_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Plot each array as a heatmap
        for i, array in enumerate(arrays):
            ax = axes[i]
            cax = ax.imshow(array, cmap='viridis', aspect='auto')
            fig.colorbar(cax, ax=ax)
            ax.set_title(subplot_titles[i])

            # Set custom x-axis labels at the top
            ax.xaxis.set_ticks_position('top')
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

def plot_plan_heatmap(portals, links, links_that_can_flip, frm_portal_ends):
    plt.figure(figsize=(16, 9))
    sct = plt.scatter(list(map(Portal.get_lng, portals)), list(map(Portal.get_lat, portals)), c=list(map(Portal.get_value, portals)), cmap='viridis')
    plt.colorbar(sct)
    for l in links:
        plt.plot(list(map(Portal.get_lng, l.get_portals())), list(map(Portal.get_lat, l.get_portals())), color="#00ff00", zorder=0)
    for l in links_that_can_flip:
        plt.plot(list(map(Portal.get_lng, l.get_portals())), list(map(Portal.get_lat, l.get_portals())), color="#0000ff", zorder=0)
    for l in frm_portal_ends:
        plt.plot(list(map(Portal.get_lng, l.get_portals())), list(map(Portal.get_lat, l.get_portals())), color="#ff0000", zorder=0)
        
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def assistance():
    print("Syntax: ")

def main(opts: list[tuple[str, str]], args):
    input_path = None
    plan_path = None

    for o, a in opts:
        if o == "-h":
            assistance()
            sys.exit(2)
        elif o == "-p":
            if a.lower() == "all":
                for file in Ingress.portal_group_map.values():
                    Ingress.add_from_bkmrk_file(file)
            else:
                for portal_group in a.split(","):
                    Ingress.add_from_bkmrk_file(Ingress.portal_group_map[portal_group.strip()])
        elif o == "--input":
            input_path = a
        elif o == "--plan":
            plan_path = a

    assert input_path or plan_path, "make sure to use --input or --plan options to provide json files for analysis"
    assert Ingress.used_portals, "no portals selected to split with, make sure you are using -p"

    if input_path:
        input = Ingress.read_json(input_path, assistance)
        base_fields = Ingress.parse_input(input)[0]
        split_methods = [Field.spiderweb, Field.hybrid(3), Field.hybrid(4), Field.hybrid(5), Field.hybrid(6), Field.hybrid(7), Field.hybrid(8), Field.homogen]
        # needs to mirror split_methods
        x_labels = ["spiderweb", "hybrid3", "hybrid4", "hybrid5", "hybrid6", "hybrid7", "hybrid8", "homogen"]

        trees = generate_trees(base_fields, split_methods)
        arrays = [
            [list(map(Tree.get_MU, row)) for row in trees],
            [list(map(Tree.get_mean_level, row)) for row in trees],
            [list(map(Tree.get_standard_deviation, row)) for row in trees],
        ]
        # needs to mirror arrays
        subplot_titles = ["MU", "μ level", "σ"]
        plot_input_analysis(x_labels, arrays, subplot_titles)

    if plan_path:
        plan = Ingress.read_json(plan_path, assistance)
        simulation = Ingress.simulate_plan(plan)
        all_steps = map(lambda r: simulation[r]["steps"], simulation)
        simulation_objects = Ingress.flatten_iterable_of_tuples(Ingress.flatten_iterable_of_tuples(all_steps))

        portals = list(filter(lambda o: isinstance(o, Portal), simulation_objects))
        links: list[Link] = list(filter(lambda o: isinstance(o, Link), simulation_objects))
        fields = list(filter(lambda o: isinstance(o, Field), simulation_objects))
        links_that_can_flip: list[Link] = list(filter(lambda l: not any(map(l.is_within_field, fields)) or l.get_length() < 2000, links))
        frm_portal_indicators = [Link(l.get_frm(), l.get_to().find_middle(l.get_frm()).find_middle(l.get_frm()).find_middle(l.get_frm())) for l in links]

        plot_plan_heatmap(portals, links, links_that_can_flip, frm_portal_indicators)

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:", ["input=", "plan="])
    main(opts, args)