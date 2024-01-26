-> Ingress.parse_input(input.json) -> (start, route, base, other) DONE
-> tree = Tree(base) DONE (optimise to maximise the MU)
-> Ingress.render(tree: Tree, color_map: dict, offset: bool, onlyleaves: bool) DONE
-> Ingress.plan(tree, start, route)
-> simulate_plan(plan)

                                              -> output.json -> IITC intel ingress
IITC intel Ingress -> input.json -> main.py <  
                                              -> plan.json -> snapshot.py -> snaphot.json -> IITC intel ingress

                                                    -> output.json -> (render.py) -> output.png
portal data -> (some drawing interface) -> input.json -> main.py <  
                                                    -> plan.json -> (render.py) -> plan.gif

## TODO:
- [ ] optimise Tree generating to prefer MU (NOTE: revisit Field.score) (ANOTHER NOTE: it would be choise as if it would make herringbones just because that would get you the most MU) (PS: and make sure to make it as a separate node so I can see number go up)
    - [ ] if multiple split_portals have the same score make em' both and compare the total MU (not that simple)

- [x] rework Ingress.plan to "apply" a route to any number of trees
- [ ] have render.py animate a plan with multiple routes
- [ ] Ingress.plan should also give a list of portals to remote key view for charging all the fields 
- [ ] Tree estimates how much XM per day it takes to upkeep if all portals have L8,L7,L6,L6,L5,L5,L4,L4 resonators
- [ ] merge snapshot into render and plot list[Link|Field|Portal] directly instead of map(render, list[Link|Field|Portal]) -> list[IITC_elements]

- [ ] introduce some OOP to make things more flexible
    - [ ] snapshot.py have step 1 be the finished product
    - [ ] render.py when rendering plan have a couple of frames of the first frame (the finished product)

- [ ] gain independence from intel.ingress.com by making my own drawing UI thing
    - [ ] click and drag make multiple points that snap to portals
    - [ ] can split routes with knife tool
- [ ] get portal data while abiding tos
    - [ ] implement herringbones

- [ ] feed it street data to get accurate route length
    - [ ] let it figure out the route given street data and portals and starting position (and end position? like home)

- [ ] use argparse instead of getopt?
    - [ ] import over rekey.py functions to a rekey subcommand in main.py
    - [ ] import over snapshot.py functions to a snapshot subcommand in main.py

- [ ] add tutorial to README.md about adding portals and using the -p option

## DONE:
- [x] object oriented base to build more features on top of (Tree, Field, Portal, Ingress)
- [x] Ingress.plan takes the tree, starting point and route to make the plan
- [x] snapshot.py -p PV path/to/plan.json step_number returns an IITC output of a snapshot of the links and fields created until a certain step
- [x] render.py path/to/output.json creates output.png from a .json file with IITC giberish init`
- [x] render.py path/to/plan.json creates output.gif with each step of the plan
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