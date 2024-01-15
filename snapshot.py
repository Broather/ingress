import getopt
import sys
import json
import itertools
from main import Ingress, Portal

class Link:
    def __init__(self, frm, to, color) -> None:
        self.a: Portal = Ingress.find_portal(frm)
        self.b: Portal = Ingress.find_portal(to)
        self.portals = {self.a, self.b}
        self.color = color
    
    def __repr__(self) -> str:
        return f"{self.portals}"

    def is_touching(self, other: object)  -> bool:
        if isinstance(other, Link):
            return self.a in other.portals or self.b in other.portals
        else:
            return False

    def intersection(self, other: object) -> Portal|None:
        if isinstance(other, Link) and self.is_touching(other):
            return self.portals & other.portals 

    def is_loop(self, one, other) -> bool:
        if isinstance(one, Link) and isinstance(other, Link):
            return self.is_touching(one) and self.is_touching(other) and one.is_touching(other) and one.intersection(other) not in self.portals
        else:
            return False
            
class Field:
    def __init__(self, l1: Link, l2: Link, l3: Link, color) -> None:
        self.portals = l1.portals | l2.portals | l3.portals
        self.color = color
    def __repr__(self) -> str:
        return f"{self.portals}"

def help():
    print("Syntax: python snapshot.py [-h] [-p comma_separated_list[<PV|VP|..>]] path/to/plan.json step_number")
    print("""simulates going through the plan's steps and creates snapshot.json as the progress when stopped at step_number 
    
    Arguments:
        path/to/plan.json:str path to the plan we want to take a snapshot of
        step_number:int numbered step to take the snapshot at
    Options:
        h: calls this help function
        p: defines which portal groups to use in making fields (only way I could think of to get portal data here)
    """)

def main(opts: list[tuple[str, str]], args):
    assert len(args) == 2, f"ERROR: snapsot only accepts two positional arguments, {len(args)} were given"
    path = args[0]
    step_to_stop_at = int(args[1])

    for o, a in opts:
        if o == "-h":
            help()
        elif o == "-p":
            for portal_group in a.split(","):
                with open(Ingress.portal_group_map[portal_group.strip()], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])
    
    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"
    try:
        with open(path, "r", encoding="utf-8") as f:
            input = json.load(f)
        steps: dict = input[0]["Steps"]
    except FileNotFoundError:
        print(f"{path} no found")
        return

    relevant_steps = list(itertools.islice(steps, step_to_stop_at))
    print(f"going through these steps: {relevant_steps}")

    blue = "#0022ff"
    green = "#00ff08"
    output = []
    for key in relevant_steps[:-1]:
        for other in steps[key]["links"]:
            # output.append(Link(key, other))
            output = my_append(Link(key, other, green), output)

    # last step is colored blue
    last_step = relevant_steps[-1]
    for other in steps[last_step]["links"]:
        # output.append(Link(key, other))
        output = my_append(Link(last_step, other, blue), output)
    
    output = list(map(render, output))
    Ingress.output_to_json(output, "./snapshot.json")

def my_append(link: Link, output: list) -> list:
    """
    checks to see if field or fields can be made from link and the current state of things (output) 
    """
    # check how many links in output are touching link
    is_touching_map = list(map(link.is_touching, output))
    amount_of_links_touching = sum(is_touching_map)
    print(f"There are {amount_of_links_touching} links touching {link}")
    if amount_of_links_touching < 2:
        output.append(link)
    else:
        # check if any combination of link and 2 links that are touching create a closed loop
        touching_links = [l for l, is_touching in zip(output, is_touching_map) if is_touching]
        for one, other_one in itertools.combinations(touching_links, 2):
            if link.is_loop(one, other_one):
                field = Field(link, one, other_one, link.color)
                print(f"The links {link} {one} {other_one} create a field: {field}, appending...")
                output.append(field)

    return output

    
    return output
def render(object: Link|Field) -> dict:
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
    else:
        raise Exception(f"ERROR attempted to render unrecognised object of type: {type(object)}")
    return data

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:", [])
    main(opts, args)