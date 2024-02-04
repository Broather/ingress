-> Ingress.parse_input(input.json) -> (start, route, base, other) DONE
-> tree = Tree(base) DONE (optimise to maximise all the MU, maximise all the MU, even with all this weather)
-> Ingress.render(tree: Tree, color_map: dict) DONE
-> Ingress.create_plan(tree, start, route) DONE
-> render(plan) DONE

                                              -> output.json -> IITC intel ingress
IITC intel Ingress -> input.json -> main.py <  
                                              -> plan.json -> snapshot.py -> snaphot.json -> IITC intel ingress

                                                    -> output.json -> (render.py) -> output.png
portal data -> (some drawing interface) -> input.json -> main.py <  
                                                    -> plan.json -> (render.py) -> plan.gif

## TODO:
- [ ] optimise Tree generating to prefer MU (NOTE: revisit Field.score) (ANOTHER NOTE: it would be choise as if it would make herringbones just because that would get you the most MU) (PS: and make sure to make it as a separate mode so I can see number go up)
    - [x] spider-web mode: choose split portal to be the closest one to any of min(map(portal.distance, Field.portals))
    - [x] homogenious mode: choose split portal to evenly distribute the rest of the portals. If multiple portals have a pretty good distribution make all of them and compare how homogenious they are min(my_amplitude(map(Field.count_portals, Field.split(portal))))
    - [x] hybrid mode: a mix between spider and homogen if there's a split portal within a threshold of Field.portals then take that, otherwise one that makes a homogen
    - [ ] absolute max MU mode: make every combination of splitting portals and compare the MU between them (will have to see execution time)

- [x] have some kind of grayscale colormap mapping the range of levels to 0-255 of value of a HSV color
- [x] have some kind of hue colormap mapping the range of levels to 0-255 of bhue of a HSV color
- [ ] add a colored legend of IITC elements to the side of all fields
    - [ ] Tree.get_bounding_box() -> (tl: Portal, br: Portal)
    - [ ] Ingress.merge_iterable_to_set(map(Tree.get_bounding_box(), all_trees))
- [x] rework Ingress.plan to "apply" a route to any number of trees
- [ ] have render.py animate a plan with multiple routes
- [ ] have render.py color visited portals green
- [ ] Ingress.plan should also give a list of portals to remote key view for charging every portal (600m radius) 
- [ ] Tree estimates how much XM per day it takes to upkeep if all portals have L8,L7,L6,L6,L5,L5,L4,L4 resonators
- [ ] merge snapshot into render and plot list[Link|Field|Portal] directly instead of map(render, list[Link|Field|Portal]) -> list[IITC_elements]

- [ ] gain independence from intel.ingress.com by making my own drawing UI thing
    - [ ] get portal data while abiding tos
    - [ ] click and drag make multiple points that snap to portals
    - [ ] can split routes with knife tool
    - [ ] implement drawing herringbones

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