import sys
import getopt
import json
import numpy as np
from itertools import starmap
import matplotlib.pyplot as plt
from main import Ingress, Field, Tree
from typing import Callable
        
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

def main(opts: list[tuple[str, str]], args):
    for o, a in opts:
        if o == "-h":
            help()
            sys.exit(2)
        elif o == "-p":
            if a.lower() == "all":
                for file in Ingress.portal_group_map.values():
                    Ingress.add_from_bkmrk_file(file)
            else:
                for portal_group in a.split(","):
                    Ingress.add_from_bkmrk_file(Ingress.portal_group_map[portal_group.strip()])
    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"
    print(f"{len(Ingress.used_portals)} portals in Ingress.used_portals")
    try:
        with open('./input.json', 'r') as f:
            input: list[dict] = json.load(f)
    except FileNotFoundError:
        help(first_time = True)
        return
    except json.decoder.JSONDecodeError:
        print("input.json is empty, make sure to copy/paste whatever IITC gives you into input.json (and save it)")
        return

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
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:c:s:l", ["noplan", "nosim", "nolegend"])
    main(opts, args)