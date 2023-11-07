## TODO:
- [x] object oriented base to build more features on top of (Tree, Field, Portal, Ingress)

input.json
-> parse into (start, route, base, other) DONE
-> tree = Tree(base) DONE
-> tree.output("output.json") DONE
-> plan = Plan(start, route, tree)
-> simulate_plan(plan)

- [ ] planner takes the tree, starting point and route to make the plan
- [ ] plan simulator goes through the steps and gives an approximate time to complete with the keys and mods I currently have in my inventory
- [ ] infinitywar.exe

- [ ] colormap the layers for better understanding of how thick a trinagle is at a glance (ingress lvl, rainbow, gray)
- [ ] option to color bottom layers in black so only top layers show (or not render them at all)
- [ ] separate layers with a lat offset based on level (like a sandwich commercial)