## TODO:
- [x] object oriented base to build more features on top of (Tree, Field, Portal, Ingress)

input.json
-> parse into (start, route, base, other) DONE
-> tree = Tree(base) DONE
-> tree.output("output.json") DONE?
-> plan = Plan(start, route, tree)
-> simulate_plan(plan)

- [ ] planner takes the tree, starting point and route to make the plan
- [ ] plan simulator goes through the steps and gives an approximate time to complete with the keys and mods I currently have in my inventory
- [ ] infinitywar.exe

- [ ] make Ingress.render(tree: Tree, color_map: dict, offset: bool, top: bool) -> list[dict] for IITC 
    - [x] color_map: maps Field.level to a color
    - [x] offset: adds an offset * Field.level to lattitude to separate layers 
    - [ ] top: renders only the top layer of each section (great with ingress color_map) #NOTE might be difficult to implement because need to split triangles if a portion of it is covered in a higher layer
        - [ ] render only tree leafs
        - [ ] implement Field splitting where partial overlap (or just leave it if it looks good)

- [x] colormap the layers for better understanding of how thick a trinagle is at a glance (ingress lvl, rainbow, gray)
- [ ] option to color bottom layers in black so only top layers show (or not render them at all)
- [ ] separate layers with a lat offset based on level (like a sandwich commercial)

- [ ] puts contents of output.json to clipboard aswell
- [ ] main.py creates input.json and output.json if doesn't exists
- [ ] add tutorial about adding portals and using the -p option 