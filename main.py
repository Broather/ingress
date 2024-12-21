from itertools import starmap
import json
import getopt, sys
from typing import Callable, Optional
from ingress import Ingress, Field, Tree

def assistance(first_time = False):
    if first_time:
        print("Looks like it's your first time launching main.py")

        open("./input.json", "w").close()
        print("input.json created")

        print("""Install IITC for your browser at http://iitc.me and go to http://www.intel.ingress.com
        For more details on which IITC extentions to install refer to the README.md file\n\n""")

    print("Syntax: python main.py [-hol] [-c <rainbow|grayscale>] [-s <spiderweb|hybrid5|homogen>]")
    print("""
    Options:
        h: calls this help function
        a: adds all defined portal groups to Ingress.used_portals
        c: selects the color map to use, (default: rainbow for all layers)
        s: selects the split method to use, (default: hybrid6)
        l: display only the leaf fields, aka the top most layer of each section
        --input: specify path to json file containing IITC output
        --splitprofile: specify split methods for each base field
        --noplan: skip making a plan
        --nosim: skip simulating created plan
        --nolegend: skip including a legend in output.json
    """)

def main(options: list[tuple[str, str]]):
    # defaults
    make_plan = True
    make_simulation = True
    make_legend = True
    split_method = Field.hybrid(6)
    onlyleaves = False
    color_map = Ingress.color_maps["rainbow"]
    input_path = "./input.json"
    split_profile = None
    Ingress.load_portals("./portals")

    # option parsing
    for o, a in options:
        if o == "-h":
            assistance()
            sys.exit(2)
        elif o == "-c":
            if (c_map := Ingress.color_maps.get(a)) is None:
                print(f"ERROR: color map {a} not recognised, your options are {Ingress.color_maps.keys()}")
                assistance()
                sys.exit(2)
            else:
                color_map = c_map
        elif o == "-s":
            split_method = Ingress.parse_split_method(a.strip())
        elif o == "-l":
            onlyleaves = True
        elif o == "--noplan":
            make_plan = False
        elif o == "--nosim":
            make_simulation = False
        elif o == "--nolegend":
            make_legend = False
        elif o == "--input":
            input_path = a
        elif o == "--splitprofile":
            split_profile = list(map(Ingress.parse_split_method, map(str.strip, a.split(","))))
        else:
            assert False, f"ERROR: unhandled option: {o}"

    assert Ingress.used_portals, "no portals selected to split with, make sure you are using -p"
    print(f"{len(Ingress.used_portals)} portals in Ingress.used_portals")

    input = Ingress.read_json(input_path, assistance)
    if input is None: return

    # base_fields get split, routes get applied to them to make a plan, other just gets appended to output
    base_fields, routes, other = Ingress.parse_input(input)
    # create a uniform split_profile if not defined
    if split_profile is None:
        split_profile = [split_method] * len(base_fields)

    assert len(split_profile) == len(base_fields), f"--splitprofile option must recieve {len(base_fields)} methods, recieved {len(split_profile)}"
    all_trees: list[Tree] = list(starmap(Tree, list(zip(base_fields, split_profile))))
    all_fields = []
    for tree in all_trees:
        if onlyleaves:
            fields = list(filter(Field.is_leaf, tree.get_fields()))
        else:
            fields = tree.get_fields()


        all_fields.extend(fields)


    output = Ingress.render(all_fields, color_map)

    if make_legend:
        legend = Ingress.create_legend(all_fields, onlyleaves)
        output.extend(Ingress.render(legend, color_map))

    Ingress.output_to_json(output + other, "./output.json")
    Ingress.copy_to_clipboard(output + other)

    if make_plan:
        plan = Ingress.create_plan(routes, all_trees)
        Ingress.output_to_json(plan, "./plan.json")
        if make_simulation:
            Ingress.simulate_plan(plan)

if __name__ == "__main__":
    opts, _ = getopt.getopt(sys.argv[1:], "hc:s:l", ["noplan", "nosim", "nolegend", "splitprofile="])
    main(opts)
