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

## KOSMOS:
- An algorithm that "finds" 2D shapes in a group of portals 
- Ingress is a game about making triangles, but what if I make squares? Perfect, equal side squares
- Like im in a mafia movie and that's my calling card when I pull off a heist (operation)
- [ ] make an interactive plan-animation (planimation) viewer with a timeline that shows where one route ends and another starts and ability to zoom in with the scroll-wheel
## TODO:
- [ ] optimise Tree generating to prefer MU
    - [x] spider-web mode: choose split portal to be the closest one to any of min(map(portal.distance, Field.portals))
    - [x] homogenious mode: choose split portal to evenly distribute the rest of the portals
        - [x] deal with the edge case when there are multiple split_portals with the best distribution possible (calculate the variance of a tree and take the one with the lowest variance)
    - [x] hybrid mode: a mix between spider and homogen if there's a split portal within a threshold of Field.portals then take that, otherwise one that makes a homogen
    - [ ] absolute max MU mode: make every combination of splitting portals and compare the MU between them (will have to see execution time)

- [ ] Ingress.bounding_box doesn't scale the view when going from one to another even if it tries to maintain a 1:1 proportion between lat and lng
- [ ] add percentual padding to Ingress.bounding_box and use it in create_legend
- [ ] Ingress.create_plan should also give a list of the least amount of portals to remote key view from (600m radius) 
- [ ] Tree estimates how much XM per day it takes to upkeep if all portals have L8,L7,L6,L6,L5,L5,L4,L4 resonators
- [x] implement __hash__ and __eq__ for Link just in case
- [ ] implement __hash__ and __eq__ for Field just in case
- [ ] render.py simulate_plan should warn when active portal is within a field
- [ ] render.py simulate_plan alternate option that divides each link creation in a separate step (render all green and additively plot)

- [ ] gain independence from intel.ingress.com by making my own drawing UI thing
    - [ ] get portal data while abiding tos (probs hardcoded .json city files like I've been doing)
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
- [x] have some kind of grayscale colormap mapping the range of levels to 0-255 of value of a HSV color
- [x] have some kind of hue colormap mapping the range of levels to 0-255 of bhue of a HSV color
- [x] add a colored legend of IITC elements to the side of all fields
    - [x] Ingress.bounding_box(fields: list[Field]) -> (tl: Portal, br: Portal)
- [x] redo render.py plot_IITC_elements to use Ingress.bounding_box
- [x] have render.py animate a plan with multiple routes
- [x] have render.py color visited portals green
- [x] rework Ingress.plan to "apply" a route to any number of trees