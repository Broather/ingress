import getopt
import sys
import json
import itertools
import math
from main import Ingress, Portal

color_palette = {
    "blue": "#0022ff",
    "green": "#00ff08" 
}
class Link:
    instance_count = 0
    def __init__(self, frm, to, color) -> None:
        self.a: Portal = Ingress.find_portal(frm)
        self.b: Portal = Ingress.find_portal(to)
        self.portals = {self.a, self.b}
        self.color = color
        Link.instance_count += 1
    
    def __repr__(self) -> str:
        return f"{self.portals}"

    def is_touching(self, other: object)  -> bool:
        if isinstance(other, Link):
            return self.a in other.portals or self.b in other.portals
        else:
            return False

    def intersection(self, other: object) -> set:
        if isinstance(other, Link) and self.is_touching(other):
            return self.portals & other.portals 

    def is_loop(self, one, other) -> bool:
        if isinstance(one, Link) and isinstance(other, Link):
            return self.is_touching(one) and self.is_touching(other) and one.is_touching(other) and one.intersection(other).isdisjoint(self.portals)
        else:
            return False
            
class Field:
    instance_count = 0
    instance_total_MU = 0

    def __init__(self, l1: Link, l2: Link, l3: Link, color) -> None:
        self.portals = l1.portals | l2.portals | l3.portals
        self.color = color
        Field.instance_count += 1
        Field.instance_total_MU += math.ceil(self.get_MU())

    def __repr__(self) -> str:
        return f"{self.portals}"
    
    def get_MU(self) -> float:
        sides = list(itertools.starmap(Portal.distance, itertools.combinations(self.portals, 2)))
        assert len(sides) == 3, f"field {self} has {len(sides)}, should have 3"
        
        # Semi-perimeter of the triangle
        s = sum(sides) / 2

        # Heron's formula for area of a triangle
        area = math.sqrt(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]))

        MU_COEFICIENT = 4.25 * 10**-5
        return math.ceil(MU_COEFICIENT*area)

def my_append(link: Link, output: list, only_links: bool) -> list:
    """
    checks to see if field or fields can be made from link and the current state of things (output) 
    """
    # check how many links in output are touching link
    is_touching_map = list(map(link.is_touching, output))
    output.append(link)
    if not only_links:
        # check if any combination of link and 2 links that are touching create a closed loop
        touching_links = [l for l, is_touching in zip(output, is_touching_map) if is_touching]
        for one, other_one in itertools.combinations(touching_links, 2):
            if link.is_loop(one, other_one):
                field = Field(link, one, other_one, link.color)
                # print(f"The links {link} {one} {other_one} create a field: {field}, appending...")
                output.append(field)

    return output
    
def render(object: Link|Field|Portal) -> dict:
    if isinstance(object, Link):
        data = {
            "type": "polyline",
            "latLngs": [{"lat": portal.lat, "lng": portal.lng} for portal in object.portals],
            "color": object.color
        }
    elif isinstance(object, Field):
        data = {
            "type": "polygon",
            "latLngs": [{"lat": portal.lat, "lng": portal.lng} for portal in object.portals],
            "color": object.color
        }
    # TODO: all portals are colored blue, not elegant 
    elif isinstance(object, Portal):
        data = {
            "type": "marker",
            "latLng": {"lat": object.lat, "lng": object.lng},
            "color": color_palette["blue"]
        }
    else:
        raise Exception(f"ERROR attempted to render unrecognised object of type: {type(object)}")
    return data

def render_plan(steps: dict, step_to_stop_at: int, only_links: bool) -> list[dict]:
    relevant_steps = list(itertools.islice(steps, step_to_stop_at))
    print(f"Total steps: {len(steps)}, going through {step_to_stop_at} of them")

    output = []
    for key in relevant_steps[:-1]:
        for other in steps[key]["links"]:
            # output.append(Link(key, other))
            output = my_append(Link(key, other, color_palette["green"]), output, only_links)

    # last step is colored blue
    last_step = relevant_steps[-1]

    # append the portal the player is interacting with
    if last_step_portal := Ingress.find_portal(last_step): 
        output.append(last_step_portal)
        
    for other in steps[last_step]["links"]:
        # output.append(Link(key, other))
        output = my_append(Link(last_step, other, color_palette["blue"]), output, only_links)
    
    output = list(map(render, output))
    return output
    
def help():
    print("Syntax: python snapshot.py [-hl] [-p comma_separated_list[<PV|VP|..>]] path/to/plan.json step_number")
    print("""simulates going through the plan's steps and creates snapshot.json as the progress when stopped at step_number 
    
    Arguments:
        path/to/plan.json:str path to the plan we want to take a snapshot of
        step_number:int numbered step to take the snapshot at
    Options:
        h: calls this help function
        l: renders only links
        p: defines which portal groups to use in making fields (only way I could think of to get portal data here)
    """)

def main(opts: list[tuple[str, str]], args):
    assert len(args) == 2, f"ERROR: snapsot only accepts two positional arguments, {len(args)} were given"
    path = args[0]
    step_to_stop_at = int(args[1])

    # defaults
    only_links = False
    for o, a in opts:
        if o == "-h":
            help()
        elif o == "-l":
            only_links = True
        elif o == "-p":
            for portal_group in a.split(","):
                with open(Ingress.portal_group_map[portal_group.strip()], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])
    
    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"
    try:
        with open(path, "r", encoding="utf-8") as f:
            input: list[dict] = json.load(f)
    except FileNotFoundError:
        print(f"{path} not found")
        return
        
    if steps := input[0].get("Steps") == None:
        assert False, f"ERROR: unrecognised input file {path}"

    output = render_plan(steps, step_to_stop_at, only_links)

    Ingress.output_to_json(output, "./snapshot.json")
    Ingress.copy_to_clipboard(output)

    AP_FOR_CREATING_A_LINK = 313
    AP_FOR_CREATING_A_FIELD = 1250

    print(f"Links made: {Link.instance_count}")
    print(f"Fields made: {Field.instance_count}")
    print(f"AP from links and fields: {AP_FOR_CREATING_A_LINK*Link.instance_count + AP_FOR_CREATING_A_FIELD*Field.instance_count}")
    print(f"estimated MU from fields: {Field.instance_total_MU}")

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hlp:", [])
    main(opts, args)