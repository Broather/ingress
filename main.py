from typing import Iterable
import json
import itertools
import getopt, sys
import pyperclip
import os
import math

def my_argmin(lst: list) -> int:
    return list.index(lst, min(lst))

class Portal:
    def __init__(self, label: str, lat: float, lng: float) -> None:
        self.label = label.replace(' ', '_')
        self.lat = lat
        self.lng = lng

    def __hash__(self) -> int:
        return hash((self.lat, self.lng))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Portal):
            return self.lat == other.lat and self.lng == other.lng
        return False

    def __repr__(self) -> str:
        return f"{self.label}"

    def get_latlng(self) -> dict:
        return {"lat": self.lat, "lng": self.lng}

    def get_lat(self) -> float:
        return self.lat

    def get_lng(self) -> float:
        return self.lng

    def get_label(self) -> str:
        return self.label
        
    def transform(self, other_portal: object, distance: float) -> None:
        if isinstance(other_portal, Portal):
            """moves portal along the vector [other_portal, self] by distance (made by GPT-3.5)"""
            
            vector_lng = self.lng - other_portal.lng
            vector_lat = self.lat - other_portal.lat
            magnitude = math.sqrt(vector_lng**2 + vector_lat**2)

            assert magnitude != 0, "ERROR: Portal.transform is attempting to transform with a portal that has the same coordinates as self"
            # Normalize the vector (convert it to a unit vector)
            vector_lng /= magnitude
            vector_lat /= magnitude

            # Update the coordinates based on the normalized vector and distance
            self.lng += vector_lng * distance
            self.lat += vector_lat * distance
        else:
            assert False, f"ERROR: Portal.transform recieved obeject of type {type(other_portal)}. Expected type Portal"

    def distance(self, other: object) -> float:
        """
        Haversine distance between 2 Portals

        arguments:
            self Portal: one portal
            other Portal: the other portal

        returns the distance in meters between 2 Portals
        """
        if isinstance(other, Portal):
            # Convert latitude and longitude from degrees to radians
            lat1, lng1, lat2, lng2 = map(math.radians, [self.lat, self.lng, other.lat, other.lng])

            # Haversine formula
            dlat = lat2 - lat1
            dlng = lng2 - lng1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            RADIUS_OF_EARTH = 6371000 # meters
            distance = RADIUS_OF_EARTH * c

            return distance
        return -1

    @staticmethod
    def from_latLng(latLng: dict):
        """
        creates a Portal from geographic coordinates dictionary
        
        arguments:
            latLng dict: like {lat: 69.69, lng: 420.420}

        returns Portal
        """
        return Portal(Ingress.get_label(latLng), *Ingress.parse_latLng(latLng))

    @staticmethod
    def from_IITC_marker_polyline(marker: dict, polyline: dict):
        """
        returns list[Portal]
        """
        start_portal = Portal(Ingress.get_label(marker["latLng"]), *Ingress.parse_latLng(marker["latLng"]))
        latLngs = polyline["latLngs"]
        route_portals = [Portal(Ingress.get_label(latLng), *Ingress.parse_latLng(latLng)) for latLng in latLngs]

        if route_portals[0] == start_portal:
            return route_portals
        elif route_portals[-1] == start_portal:
            return route_portals[::-1]
        else:
            assert False, "ERROR: a polyline's beginning and end does not match a marker's coordinates"

    @staticmethod
    def from_IITC_polygon(IITC_polygon: dict):
        """
        returns list[Portal]
        """
        if IITC_polygon["type"] != "polygon":
            print(f"WARNING: from_IITC_polygon is attempting to parse type of {IITC_polygon['type']}")
        latLngs = IITC_polygon["latLngs"]
        return [Portal(Ingress.get_label(latLng), *Ingress.parse_latLng(latLng)) for latLng in latLngs]
        
class Field:
    def __init__(self, p1: Portal, p2: Portal, p3: Portal, level: int, is_herringbone = False) -> None:
        self.portals: list[Portal] = [p1,p2,p3]
        self.level: int = level
        self.split_portal = None
        self.children: list[Field] = []
        self.is_herringbone = is_herringbone
    
    def __repr__(self) -> str:
        return f"{self.portals} {len(self.get_portals())}"

    def get_links(self) -> list[frozenset]:
        return list(map(frozenset, itertools.combinations(self.portals, 2)))

    def get_MU(self) -> int:
        MU_COEFICIENT = 4.25 * 10**-5
        return math.ceil(MU_COEFICIENT*self.get_area())

    def get_area(self) -> float:
        sides = list(itertools.starmap(Portal.distance, itertools.combinations(self.portals, 2)))

        # Semi-perimeter of the triangle
        s = sum(sides) / 2

        # Heron's formula for area of a triangle
        area = math.sqrt(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]))

        return area

    def get_level(self) -> int:
        return self.level

    def get_portals(self) -> list[Portal]:
        """returns all portals that are inside of field (not including the 3 that make up the field)"""
        return list(filter(self.is_in, Ingress.used_portals))

    def get_portals_inclusive(self) -> list[Portal]:
        """returns all portals that are inside of field (including the 3 that make up the field)"""
        return self.get_portals() + self.portals
    
    def is_leaf(self):
        """returns True if field does not have children"""
        return len(self.children) == 0

    def is_in(self, portal: Portal) -> bool:
        """
        Check if a Portal is inside the Field on Earth's surface. (made by GPT-3.5)

        arguments:
            self Field: contains data about portals it contains
            portal Portal: Latitude and longitude of the point (in degrees).

        returns bool: True if the point is inside the triangle, otherwise False.
        """

        def sign(p1: Portal, p2: Portal, p3: Portal):
            return (p1.lat - p3.lat) * (p2.lng - p3.lng) - (p2.lat - p3.lat) * (p1.lng - p3.lng)

        def barycentric_coordinates(p: Portal, a: Portal, b: Portal, c: Portal):
            denom = (b.lng - c.lng) * (a.lat - c.lat) + (c.lat - b.lat) * (a.lng - c.lng)
            weight_a = ((b.lng - c.lng) * (p.lat - c.lat) + (c.lat - b.lat) * (p.lng - c.lng)) / denom
            weight_b = ((c.lng - a.lng) * (p.lat - c.lat) + (a.lat - c.lat) * (p.lng - c.lng)) / denom
            weight_c = 1 - weight_a - weight_b
            return weight_a, weight_b, weight_c

        a, b, c = self.portals

        assert a.lat != b.lat and a.lng != b.lng, "ERROR: Field has 2 Portals in the same location: (a and b)"
        assert a.lat != c.lat and a.lng != c.lng, "ERROR: Field has 2 Portals in the same location: (a and c)"
        assert b.lat != c.lat and b.lng != c.lng, "ERROR: Field has 2 Portals in the same location: (b and c)"

        d1 = sign(portal, a, b)
        d2 = sign(portal, b, c)
        d3 = sign(portal, c, a)

        if d1 * d2 * d3 == 0.0:
            return False

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        if (not has_neg and not has_pos):
            return True

        w1, w2, w3 = barycentric_coordinates(portal, a, b, c)
        return (w1 >= 0) and (w2 >= 0) and (w3 >= 0)

    def score(self, center_portal: Portal) -> int:
        """
        splits the Field on portal and returns a score based on the distribution of portals on each of the 3 new fields
        """
        fields = [Field(*outer_portals, center_portal, self.level + 1) for outer_portals in itertools.combinations(self.portals, 2)]
        portal_counts = [len(field.get_portals()) for field in fields]
        return max(portal_counts) - min(portal_counts)

    def split(self):
        """
        returns list[Field]
        """
        potencial_center_portals = list(filter(self.is_in, Ingress.used_portals))

        scores = []
        for portal in potencial_center_portals:
            scores.append(self.score(portal))

        self.split_portal = potencial_center_portals[my_argmin(scores)]
        return [Field(*outer_portals, self.split_portal, self.level + 1) for outer_portals in itertools.combinations(self.portals, 2)]

    def grow(self) -> None:
        if len(self.get_portals()) > 0:
            self.children = self.split()
            for child in self.children:
                child.grow()

class Tree:
    all_instances = []

    def __init__(self, root: Field) -> None:
        self.root = root
        self.root.grow()
        Tree.all_instances.append(self)
        print(self)
    
    def __repr__(self) -> str:
        return f"{self.root} {self.get_level_range()} {self.get_MU()} MU {len(self.get_links())} links {len(self.get_fields())} fields"

    def get_fields(self, node: Field = None) -> list[Field]:
        """returns a list of all lower standing nodes (inclusive). Goes from root if node not given (made by GTP-3.5)"""

        if node is None:
            node = self.root

        fields = []
        queue = [node]

        while queue:
            current_node = queue.pop(0)
            fields.append(current_node)

            for child in current_node.children:
                queue.append(child)

        return fields
    
    def get_level_range(self) -> tuple[int, int]:
        """returns a tree's range of levels or thicc-ness"""
        leaves = list(filter(Field.is_leaf, self.get_fields()))
        leaf_levels = list(map(Field.get_level, leaves))

        return (min(leaf_levels), max(leaf_levels))

    def get_MU(self):
        return sum(map(Field.get_MU, self.get_fields()))

    def display(self, field: Field = None):
        if field is None:
            field = self.root

        print(f"{'    ' * field.level}{field.portals}")

        for child in field.children:
            self.display(child)
    
    def get_links(self) -> set[frozenset]:
        links = set()
        for field in self.get_fields():
            links.update(field.get_links())

        return links

    def get_fields_portal_is_a_part_of(self, portal: Portal, field: Field = None, snowball: list[Field] = None) -> list[Field]:
        return list(filter(lambda f: portal in f.portals, self.get_fields()))
        
    def get_lowest_level_fields_level_portal_is_a_part_of(self, portal: Portal) -> int:
        fields = self.get_fields_portal_is_a_part_of(portal)
        level = min(list(map(Field.get_level, fields))) 
        return level

class Ingress:
    portal_group_map = {
        "PV": "./portals/pavilosta.json",
        "AR": "./portals/akmens-rags.json",
        "ZP": "./portals/ziemupe.json",
        "CR": "./portals/cirava.json",
        "GD": "./portals/gudenieki.json",
        "JK": "./portals/jurkalne.json",
        "VP": "./portals/ventspils.json",
        }
        
    used_portals: list[Portal] = []

    color_maps = {"rainbow" : {
            "0": "#ff0000",
            "1": "#ffff00",
            "2": "#00ff00",
            "3": "#00ffff",
            "4": "#0000ff",
            "5": "#ff00ff",
            "default": "#bbbbbb"},
        "ingress": {
            "0": "#f0ff20",
            "1": "#ffb01c",
            "2": "#ef8733",
            "3": "#ff642c",
            "4": "#c80425",
            "5": "#ff0e82",
            "6": "#b300ff",
            "7": "#5100ff",
            "default": "#bbbbbb"},
        "gray": { 
            "default": "#bbbbbb"}}
    
    @staticmethod
    def merge_iterable_to_set(iterable: Iterable):
        union_set = set()
        for it in iterable:
            union_set.update(it)
        return union_set

    @staticmethod
    def find_portal(label: str):
        """
        finds a Portal from used_portals with label

        return Portal|None
        """
        for p in Ingress.used_portals:
            if p.get_label() == label:
                return p
        return None

    @staticmethod
    def output_to_json(object: object, json_file_path:str):
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(object, f, indent=2, ensure_ascii=False)
        print(f"{os.path.basename(json_file_path)} created successfully")   

    @staticmethod
    def copy_to_clipboard(object):
        pyperclip.copy(json.dumps(object))
        print("output copied to clipboard successfully")

    @staticmethod
    def get_label(latLng: dict):
        for portal in Ingress.used_portals:
            if portal.get_latlng() == latLng:
                return portal.get_label()
            
        assert False, f"portal with {latLng} not found in used_portals"
    
    @staticmethod
    def parse_lat_comma_lng(latlng: str):
        return [float(coordinate) for coordinate in latlng.split(",")]

    @staticmethod
    def parse_latLng(latLng: dict) -> list[float]:
        return [float(latLng["lat"]), float(latLng["lng"])]
    
    @staticmethod
    def parse_input(IITC_elements: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """
        parses the contents of input.json (which should be full of IITC copy/paste)

        Args:
        input (list[dict])

        Returns:
        tuple: (groups, other)
        where:
            groups: (portal_order: list[Portal], base_fields: list[Field])
            other: any drawn elements that are not white
        """
        white = "#ffffff"
                
        markers = [e for e in IITC_elements if e["type"] == "marker" and e["color"] == white]
        polylines = [e for e in IITC_elements if e["type"] == "polyline" and e["color"] == white]
        assert len(markers) == len(polylines), f"ERROR: amount of markers and polylines should match, markers: {len(markers)}, polylines: {len(polylines)}"

        polygons = [e for e in IITC_elements if e["type"] == "polygon" and e["color"] == white]

        base_fields = []
        routes = []

        for polygon in polygons:
            base_fields.append(Field(*Portal.from_IITC_polygon(polygon), 0))

        for marker, polyline in zip(markers, polylines):
            routes.append(Portal.from_IITC_marker_polyline(marker, polyline))

        other = [e for e in IITC_elements if e["color"] != white]

        return (base_fields, routes, other)

    @staticmethod
    def render(fields: list[Field], color_map: dict, output: list = None) -> list[dict]:
        if output == None: output = []

        for field in fields:
            data = {
                "type": "polygon",
                "latLngs": [{"lat": portal.lat, "lng": portal.lng} for portal in field.portals],
                "color": color_map.get(str(field.level), color_map["default"])
            }
            output.append(data)

        return output

    @staticmethod
    def plan(routes: list[tuple[Portal]], trees: list[Tree]) -> dict:
        AVERAGE_WALKING_SPEED = 84 # meters/minute
        COOLDOWN_BETWEEN_HACKS = 5 # minutes

        # warn if there are any portals missed by routes
        fields = []
        for tree_fields in map(Tree.get_fields, trees):
            fields.extend(tree_fields)

        all_portals = Ingress.merge_iterable_to_set(map(Field.get_portals_inclusive, fields))
        route_portals = Ingress.merge_iterable_to_set(routes)

        if (missed_portals_count := len(all_portals) - len(route_portals)) > 0:
            print(f"WARNING: routes missed {missed_portals_count} portal/-s, plan/-s might not be accurate. Missing portals {all_portals.difference(route_portals)}")

        links = Ingress.merge_iterable_to_set(map(Tree.get_links, trees))

        # making a plan for each route
        visited_portals = set()
        created_links = set()
        plan = []
        for route in routes:
            steps = {}
            SBUL_count = 0
            
            for step in route:
                visited_portals.add(step)
                connected_links = list(filter(set([step]).issubset, links))
                outbound_links = list(filter(visited_portals.issuperset, links.difference(created_links)))
                created_links.update(outbound_links)

                if len(outbound_links) > 8: SBUL_count += 1

                # TODO: I hope I have the motivation to rewrite this masterpiece
                link_order = sorted(outbound_links, key = lambda l: (min(map(Field.get_level, filter(lambda f: l in f.get_links(), fields))), -tuple(l)[0].distance(tuple(l)[1])))
                
                steps[step.get_label()] = {
                "keys": len(connected_links)-len(outbound_links),
                # TODO: this one's a bit rought aswell
                "links": list(map(lambda l: tuple(l.difference(set([step])))[0].get_label(), link_order))
                }

            route_length = round(sum(itertools.starmap(Portal.distance, itertools.pairwise(route))), 2)
            total_keys_required = len(links)

            plan.append({
                "Title": f"{route[0]}...{route[-1]}",
                "Mods_required": {"SBUL": SBUL_count},
                "Route_length_(meters)": route_length,
                "Total_keys_required": total_keys_required,
                "Estimated_time_to_complete_(minutes)": round(route_length / AVERAGE_WALKING_SPEED + total_keys_required * COOLDOWN_BETWEEN_HACKS, 2),
                "Steps": steps
                })
        
        return plan

    @classmethod
    def add_from_bkmrk(cls, bkmrk: dict) -> None:
        for id in bkmrk:
            portal = Portal(bkmrk[id]["label"], *Ingress.parse_lat_comma_lng(bkmrk[id]["latlng"]))
            cls.used_portals.append(portal)
    
def help(first_time = False):
    if first_time:
        print("Looks like it's your first time launching main.py")

        open("./input.json", "w").close()
        print("input.json created")

        print("""
        install IITC for your browser at http://iitc.me and go to http://www.intel.ingress.com.
        more details on which IITC extentions to install hopefully in the README.md file\n\n""")
        
    print("Syntax: python main.py [-hol] [-p comma_separated_list[<PV|...>]] [-c <rainbow|ingress|gray>]")
    print("""
    Options:
        h: calls this help function
        l: display only the leaf fields, aka the top most layer of each section
        p: defines which portal groups to use in making fields (only way I could think of to get portal data here)
        c: selects the color map to use, default is rainbow for all layers
    """)

def main(opts: list[tuple[str, str]], args):
    # defaults
    color_map = Ingress.color_maps["rainbow"]
    onlyleaves = False
    no_plan = False
    
    # option parsing
    for o, a in opts:
        if o == "-h":
            help()
            sys.exit(2)
        elif o == "-p":
            for portal_group in a.split(","):
                with open(Ingress.portal_group_map[portal_group.strip()], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])
        elif o == "-c":
            if color_map := Ingress.color_maps.get(a) == None:
                print(f"ERROR: color map {a} not recognised")
                help()
                sys.exit(2)
        elif o == "-l":
            onlyleaves = True
        elif o == "--noplan":
            no_plan = True
        else:
            assert False, f"ERROR: unparsed option: {o}"
    
    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"
    print(f"{len(Ingress.used_portals)} portals in Ingress.used_portals")
    try:
        with open('./input.json', 'r') as f:
            input: list[dict] = json.load(f)
    except FileNotFoundError:
        help(first_time = True)
        return
    except json.decoder.JSONDecodeError:
        print("input.json empty, make sure to copy/paste whatever IITC gives you into input.json")
        return

    # base_fields get split, routes get applied to them to make a plan, other just gets appended to output
    base_fields, routes, other = Ingress.parse_input(input)

    output = []
    for base_field in base_fields:
        tree = Tree(base_field)
        fields = tree.get_fields()

        if onlyleaves:
            fields = list(filter(Field.is_leaf, fields))
        
        output.extend(Ingress.render(fields, color_map))


    Ingress.output_to_json(output + other, "./output.json")
    Ingress.copy_to_clipboard(output + other)

    if not no_plan: 
        plan = Ingress.plan(routes, Tree.all_instances)
        Ingress.output_to_json(plan, "./plan.json")
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:c:l", ["noplan"])
    main(opts, args)