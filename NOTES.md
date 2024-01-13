-> Ingress.parse_input(input.json) -> (start, route, base, other) DONE
-> tree = Tree(base) DONE (optimise to maximise the MU)
-> Ingress.render(tree: Tree, color_map: dict, offset: bool, onlyleaves: bool) DONE
-> Ingress.plan(tree, start, route)
-> simulate_plan(plan)

## TODO:
- [x] object oriented base to build more features on top of (Tree, Field, Portal, Ingress)
- [x] Ingress.plan takes the tree, starting point and route to make the plan
- [ ] implement herringbones
- [ ] optimise Tree generating to prefer MU (NOTE: revisit Field.score)
    - [ ] if multiple split_portals have the same score make em' both and compare the total area (not that simple)
- [ ] Ingress.snapshot(steps, portal_name) returns an IITC output of a snapshot of the links and fields created at Steps[portal_name]
- [ ] use argparse instead of getopt
- [ ] copy over rekey.py code to a rekey subcommand in main.py
- [x] have Ingress.plan sort other_portals with primary parameter lowers_feld_level and secondary Portal.distance (desc)
- [x] make Ingress.render(tree: Tree, color_map: dict, offset: bool, onlyleaves: bool) -> list[dict] for IITC 
    - [x] color_map: maps Field.level to a color (ingress, rainbow, white)
    - [x] offset: adds an offset * Field.level to lattitude to separate layers 
    - [x] onlyleaves: renders only the top layer of each section (great with ingress color_map)
        - [x] render only tree leaves

- [x] make multi-Trees for multi-polygons (NOTE: make user-proof)
- [x] pass render options through getopt
- [x] puts contents of output.json to clipboard as well
- [x] main.py creates input.json, output.json and plan.json if doesn't exists
- [ ] add tutorial to README.md about adding portals and using the -p option