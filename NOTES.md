## TODO:
- [x] object oriented base to build more features on top of (Tree, Field, Portal, Ingress)

-> Ingress.parse_input(input.json) -> (start, route, base, other) DONE
-> tree = Tree(base) DONE (optimise to maximise the MU)
-> Ingress.render(tree: Tree, color_map: dict, offset: bool, top: bool) DONE
-> plan = Plan(tree, start, route)
-> simulate_plan(plan)

- [ ] planner takes the tree, starting point and route to make the plan
- [ ] plan simulator goes through the steps and gives an approximate time to complete with the keys and mods I currently have in my inventory
- [ ] optimise Tree generating to prefer AP or MU, or balanced

- [x] make Ingress.render(tree: Tree, color_map: dict, offset: bool, top: bool) -> list[dict] for IITC 
    - [x] color_map: maps Field.level to a color (ingress, rainbow, white)
    - [x] offset: adds an offset * Field.level to lattitude to separate layers 
    - [ ] top: renders only the top layer of each section (great with ingress color_map) #NOTE might be difficult to implement because need to split triangles if a portion of it is covered in a higher layer
        - [ ] render only tree leaves
        - [ ] implement Field splitting where partial overlap (or just leave it if it looks good)

- [ ] puts contents of output.json to clipboard as well
- [ ] main.py creates input.json and output.json if doesn't exists
- [ ] add tutorial about adding portals and using the -p option