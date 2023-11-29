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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Portal):
            return self.lat == other.lat and self.lng == other.lng
        return False

    def __repr__(self) -> str:
        return f"{self.label}"

    def get_latlng(self) -> dict:
        return {"lat": self.lat, "lng": self.lng}

    def get_label(self) -> str:
        return self.label
    
    def distance(self, other) -> float:
        """
        haversine distance between 2 Portals
        arguments: 
            self Portal: one portal
            other Portal: the other portal

        returns the distance in meters between 2 Portals
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [self.lat, self.lng, other.lat, other.lng])

        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        RADIUS_OF_EARTH = 6371000  # Earth radius in meters
        # Calculate the distance
        distance = RADIUS_OF_EARTH * c

        return distance

    @staticmethod
    def from_latLng(latLng: dict):
        """
        creates a Portal from geographic coordinates like {lat: 69.6969, lng:420.420}

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
        return str(self.portals)

    def get_level(self) -> int:
        return self.level

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

        assert a.lat != b.lat and a.lng != b.lng, "ERROR: Field has 2 Portls in the same location: (a and b)"
        assert a.lat != c.lat and a.lng != c.lng, "ERROR: Field has 2 Portls in the same location: (a and c)"
        assert b.lat != c.lat and b.lng != c.lng, "ERROR: Field has 2 Portls in the same location: (b and c)"

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

    def count_portals(self) -> int:
        return sum(list(map(self.is_in, Ingress.used_portals)))
    
    def get_inside_portals(self) -> list[Portal]:
        return [portal for portal in Ingress.used_portals if self.is_in(portal)]

    def score(self, center_portal: Portal) -> int:
        """
        splits the Field on portal and returns a score based on the distribution of portals on each of the 3 new fields
        """
        fields = [Field(*outer_portals, center_portal, self.level + 1) for outer_portals in itertools.combinations(self.portals, 2)]
        portal_counts = [field.count_portals() for field in fields]
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
        if self.count_portals() > 0:
            self.children = self.split()
            for child in self.children:
                child.grow()

class Tree:
    def __init__(self, root: Field) -> None:
        self.root = root
        self.root.grow()

    def display(self, field: Field = None):
        if field is None:
            field = self.root

        print(f"{'    ' * field.level}{field.portals}")

        for child in field.children:
            self.display(child)
    
    def get_links(self, field: Field = None, snowball: list[tuple] = None) -> list[tuple]:
        if snowball == None: snowball = []

        if field == None:
            field = self.root
            snowball.extend(itertools.combinations(field.portals, 2))
        
        # if fiels is NOT a leaf, aka, has children
        if len(field.children) > 0:
            assert field.split_portal != None, f"ERROR: field: {field} has childern but not assigned split_portal"
            snowball.extend([(portal, field.split_portal) for portal in field.portals])
            
            for child in field.children:
                snowball = self.get_links(child, snowball)
        
        # (root_outer_links + (root.portals and root.split_portal) + (child.portals and child.split_portal))
        return snowball

    def get_fields_portal_is_a_part_of(self, portal: Portal, field: Field = None, snowball: list[Field] = None) -> list[Field]:
        if snowball == None: snowball = []
        if field == None: field = self.root

        if portal in field.portals:
            snowball.append(field)
        
        for child in field.children:
            snowball = self.get_fields_portal_is_a_part_of(portal, child, snowball)

        return snowball
        
    def get_lowest_level_fields_level_portal_is_a_part_of(self, portal: Portal) -> int:
        fields = self.get_fields_portal_is_a_part_of(portal)
        level = min(list(map(Field.get_level, fields))) 
        return level

class Ingress:
    portal_group_map = {
        "PV": "./portals/pavilosta.json",
        "VP": "./portals/ventspils.json"}
        
    used_portals: list[Portal] = []

    color_maps = {
        "rainbow" :
            {"0": "#ff0000",
            "1": "#ffff00",
            "2": "#00ff00",
            "3": "#00ffff",
            "4": "#0000ff",
            "5": "#ff00ff",
            "default": "#bbbbbb"},
        "ingress": 
            {"0": "#f0ff20",
            "1": "#ffb01c",
            "2": "#ef8733",
            "3": "#ff642c",
            "4": "#c80425",
            "5": "#ff0e82",
            "6": "#b300ff",
            "7": "#5100ff",
            "default": "#bbbbbb"},
        "gray": 
            {"default": "#bbbbbb"}}
    
    @staticmethod
    def output_to_json(object: object, json_file_path:str):
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(object, f, indent=2, ensure_ascii=False)
        print(f"{os.path.basename(json_file_path)} created successfully")   

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
        assert len(markers) == len(polylines), f"ERROR: amount of markers and polylines should match! markers: {len(markers)}, polylines: {len(polylines)}"

        polygons = [e for e in IITC_elements if e["type"] == "polygon" and e["color"] == white]
        assert len(polygons) == len(markers), f"ERROR: amount of polygons and (marker, polyline) pairs should match: polygons: {len(polygons)}, marker-polyline: {len(markers)} "

        groups = []
        for marker, polyline, polygon in zip(markers, polylines, polygons):
            portal_order = Portal.from_IITC_marker_polyline(marker, polyline)
            if len(polygon['latLngs']) == 3:
                base_field = Field(*Portal.from_IITC_polygon(polygon), 0)
            else:
                assert len(polygon['latLngs']) == 4, f"ERROR: unable to parse polygon with {len(polygon['latLngs'])} points"
                herringbone_portals = Portal.from_IITC_polygon(polygon)
                herringbone_base_portals = [portal for portal in herringbone_portals if portal not in portal_order]
                base_field = Field(herringbone_portals[0], *herringbone_base_portals, 0, is_herringbone = True)
            groups.append((portal_order, base_field))

        other = [e for e in IITC_elements if e["color"] != white]

        return (groups, other)

    @staticmethod
    def render(field: Field, color_map: dict, offset: bool, onlyleaves: bool, output: list = None) -> list[dict]:
        if output == None: output = []
        if offset and onlyleaves and field.level == 0: print("WARNING: having offset and onlyleaves enabled at the same time makes 0 sense")

        is_leaf = (len(field.children) == 0)
        if not onlyleaves or (onlyleaves and is_leaf):
            data = {
                "type": "polygon",
                "latLngs": [{"lat": portal.lat + 0.0001*offset*field.level, "lng": portal.lng} for portal in field.portals],
                "color": color_map.get(str(field.level), color_map["default"])
            }
            output.append(data)

        for child in field.children:
            output = Ingress.render(child, color_map, offset, onlyleaves, output)
        
        return output

    @staticmethod
    def plan(tree: Tree, portal_order: list[Portal]) -> dict:
        AVERAGE_WALKING_SPEED = 84 # meters/minute
        COOLDOWN_BETWEEN_HACKS = 5 # minutes

        root = tree.root
        all_root_portals = root.get_inside_portals() + root.portals
        # assure that portal_order contains root.portals
        if not all([portal in portal_order for portal in root.portals]):
            print(f"WARNING: route does not go to all field portals for field: {root.portals}")
        # assure that every portal in root is also present in portal_order
        if len(all_root_portals) != len(portal_order):
            print(f"WARNING: route missed {len(all_root_portals) - len(portal_order)} portals in field: {root.portals}")
        
        links = tree.get_links()
        steps = {}
        visited_portals = []
        SBUL_count = 0
        
        for portal in portal_order:
            other_portals: list[Portal] = []
            # TODO: not technically "available"
            available_links = [link for link in links if portal in link]
            for link in available_links:
                other_portals.extend(p for p in link if p != portal and p in visited_portals)
                # sort by lowest field lvl (asc), but if levels are the same sort them by portal distance (desc)
                other_portals.sort(key = lambda p: (tree.get_lowest_level_fields_level_portal_is_a_part_of(p), -portal.distance(p)))

            if len(other_portals) > 8: SBUL_count += 1

            steps[portal.get_label()] = {
                "keys": len(available_links)-len(other_portals),
                "links": [p.get_label() for p in other_portals]
                }
            visited_portals.append(portal)
        
        route_length = round(sum(itertools.starmap(Portal.distance, itertools.pairwise(portal_order))), 2)
        total_keys_required = len(tree.get_links())

        return {
            "Title": str(tree.root.portals),
            "Mods_required": {"SBUL": SBUL_count},
            "Route_length_(meters)": route_length,
            "Total_keys_required": total_keys_required,
            "Estimated_time_to_complete_(minutes)": round(route_length / AVERAGE_WALKING_SPEED + total_keys_required * COOLDOWN_BETWEEN_HACKS, 2),
            "Steps": steps
            }

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
        o: adds an offset to layers so it's easier to tell them apart
        l: display only the leaf fields, aka the top most layer of each section
        p: defines which portal groups to use in making fields (only way I could think of to get portal data here)
        c: selects the color map to use, default is rainbow for all layers
    """)

def main(opts: list[tuple[str, str]], args):
    # defaults part
    color_map = Ingress.color_maps["rainbow"]
    offset = False
    onlyleaves = False
    no_plan = False
    
    # option parsing part
    for o, a in opts:
        if o == "-h":
            help()
            sys.exit(2)
        elif o == "-p":
            for portal_group in a.split(","):
                with open(Ingress.portal_group_map[portal_group.strip()], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])
        elif o == "-c":
            if a in Ingress.color_maps.keys():
                color_map = Ingress.color_maps[a]
            else: print(f"WARNING: color map {a} not recognised")
        elif o == "-o":
            offset = True
        elif o == "-l":
            onlyleaves = True
        elif o == "--noplan":
            no_plan = True
        else:
            assert False, f"ERROR: unsupported option: {o}"
    
    try:
        with open('./input.json', 'r') as f:
            input: list[dict] = json.load(f)
    except FileNotFoundError:
        help(first_time = True)
        return
    except json.decoder.JSONDecodeError:
        print("input.json empty, make sure to copy/paste whatever IITC gives you into input.json")
        return

    assert len(Ingress.used_portals) > 0, f"no portals selected to split with, make sure you are using -p"
    groups, other = Ingress.parse_input(input)

    output = []
    plan = []
    for group in groups:
        portal_order, base_field = group
        tree = Tree(base_field)
        
        output.extend(Ingress.render(tree.root, color_map, offset, onlyleaves))
        plan.append(Ingress.plan(tree, portal_order))
        
    Ingress.output_to_json(output + other, "./output.json")
    
    pyperclip.copy(json.dumps(output + other))
    print("output copied to clipboard successfully")

    if not no_plan: Ingress.output_to_json(plan, "./plan.json")
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:c:ol", ["noplan"])
    main(opts, args)