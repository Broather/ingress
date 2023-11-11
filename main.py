import json
import itertools
import getopt, sys
import pyperclip

def my_argmin(lst: list) -> int:
    return list.index(lst, min(lst))

class Portal:
    def __init__(self, label: str, lat: float, lng: float) -> None:
        self.label = label
        self.lat = lat
        self.lng = lng

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Portal):
            return self.lat == other.lat and self.lng == other.lng
        else:
            return False

    def __repr__(self) -> str:
        return f"{self.label.replace(' ', '_')}"

    def get_latlng(self) -> dict:
        return {"lat": self.lat, "lng": self.lng}
    def get_label(self) -> str:
        return self.label
    
    @staticmethod
    def from_latLng(latLng: dict):
        """
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
    def __init__(self, p1: Portal, p2: Portal, p3: Portal, level: int) -> None:
        self.children: list[Field] = []
        self.portals: list[Portal] = [p1,p2,p3]
        self.level: int = level
    
    def __repr__(self) -> str:
        return str({
            "children": self.children,
            "portals": self.portals,
            "level": self.level,
            })

    def is_in(self, portal: Portal) -> bool:
        """
        Check if a Portal is inside the Field on Earth's surface. (made by GPT3)

        Args:
        self Field: contains data about portals it contains
        portal Portal: Latitude and longitude of the point (in degrees).

        Returns:
        bool: True if the point is inside the triangle, False otherwise.
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

    def count_portals(self):
        return sum(list(map(self.is_in, Ingress.used_portals)))

    def score(self, center_portal: Portal) -> int:
        """
        splits the Field on portal and returns a score based on the distribution of portals on each of the 3 new fields
        """
        fields = [Field(*outer_portals, center_portal, self.level + 1) for outer_portals in itertools.combinations(self.portals, 2)]
        portal_counts = [field.count_portals() for field in fields]
        return max(portal_counts) - min(portal_counts)

    def split(self):
        potencial_center_portals = list(filter(self.is_in, Ingress.used_portals))

        scores = []
        for portal in potencial_center_portals:
            scores.append(self.score(portal))

        center_portal = potencial_center_portals[my_argmin(scores)]
        return [Field(*outer_portals, center_portal, self.level + 1) for outer_portals in itertools.combinations(self.portals, 2)]

    def grow(self):
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
    
    def change_color(self, input: list[dict], _from: str, to: str) -> list[dict]:
        if input["color"] == _from:
            input["color"] = to
        
        return input

class Ingress:
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
        tuple: (portal_order, base_field, other)
        where:
            portal_order: list[Portal]
            base_fields: list[Field]
            other: any drawn elements that are not white
        """
        white = "#ffffff"
        # Markers share a portal with a polygon and polyline
        # Polylines share it's beginning and end portals with a polygon and every other portala is in the polygon
        # Polygons don't overlap
                
        markers = [e for e in IITC_elements if e["type"] == "marker" and e["color"] == white]
        polylines = [e for e in IITC_elements if e["type"] == "polyline" and e["color"] == white]
        assert len(markers) == len(polylines), f"ERROR: amount of markers and polylines should match! markers: {len(markers)}, polylines: {len(polylines)}"

        polygons = [e for e in IITC_elements if e["type"] == "polygon" and e["color"] == white]
        assert len(polygons) == len(markers), f"ERROR: amount of polygons and (marker, polyline) pairs should match: polygons: {len(polygons)}, marker-polyline: {len(markers)} "

        groups = []
        for marker, polyline, polygon in zip(markers, polylines, polygons):
            portal_order = Portal.from_IITC_marker_polyline(marker, polyline)
            base_field = Field(*Portal.from_IITC_polygon(polygon), 0)
            groups.append((portal_order, base_field))

        other = [e for e in IITC_elements if e["color"] != white]

        return (groups, other)

    @staticmethod
    def render(field: Field, color_map: dict, offset: bool, onlyleaves: bool, output: list = []) -> list[dict]:
        if offset and onlyleaves and field.level == 0: print("WARNING: having offset and onlyleaves enabled at the same time makes 0 sense")

        is_leaf = len(field.children) == 0
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
    def plan(tree: Tree, start: dict, route: dict) -> list[dict]: 
        output = []
        # TODO: make a list of links that need to be made
        links = Ingress.get_links(tree)
        # TODO: with start and route make the 
        for portal in route:
            output.append({
                portal.label: "amount of keys needed",
                "links": [link for link in links if link.contains(portal)]
            })
        return output
    
    @classmethod
    def add_from_bkmrk(cls, bkmrk: dict) -> None:
        for id in bkmrk:
            portal = Portal(bkmrk[id]["label"], *Ingress.parse_lat_comma_lng(bkmrk[id]["latlng"]))
            cls.used_portals.append(portal)
    
def help():
    print("Syntax: python main.py [-h] [-p comma_separated_list[<PV|...>]]")

def add_split(input: list, split: list[dict]) -> list:
        for triangle in split:
            input.append(triangle)
        return input

def score_distibution(lst: list) -> int:
    return max(lst) - min(lst)

def my_argmin(lst: list) -> int:
    return list.index(lst, min(lst))

def split(center_portal: Portal, field: dict) -> list[dict]:
    center_portal_latlng = center_portal.get_latlng()
    field_latlng = field['latLngs']

    assert len(field_latlng) == 3, f"field must have exactly 3 portals  , this one has {len(field_latlng)}"
    
    latlng_combinations = list(itertools.combinations(field_latlng, 2))
    return [{
        "type": "polygon",
        "latLngs": [
            center_portal_latlng,
            *latlng_combinations[0]
        ],
        "color": "#bbbbbb"
    },{
        "type": "polygon",
        "latLngs": [
            center_portal_latlng,
            *latlng_combinations[1]
        ],
        "color": "#bbbbbb"
    },{
        "type": "polygon",
        "latLngs": [
            center_portal_latlng,
            *latlng_combinations[2]
        ],
        "color": "#bbbbbb"
    }]

portal_group_map = {
    "PV": "./portals/pavilosta.json"
}

def main(opts: list[tuple[str, str]], args):
    # defaults part
    color_map = Ingress.color_maps["gray"]
    offset = False
    onlyleaves = False
    ignore_plan = False
    # option parsing part
    
    for o, a in opts:
        if o == "-h":
            help()
            sys.exit(2)
        elif o == "-p":
            for portal_group in a.split(","):
                with open(portal_group_map[portal_group.strip()], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])
        elif o == "-c":
            if a in Ingress.color_maps.keys():
                color_map = Ingress.color_maps[a]
            else: print(f"WARNING: color map {a} not recognised")
        elif o == "-o":
            offset = True
        elif o == "-l":
            onlyleaves = True
        else:
            assert False, f"unsupported option: {o}"
    
    with open('./input.json', 'r') as f:
        input: list[dict] = json.load(f)

    groups, other = Ingress.parse_input(input)
    
    assert len(Ingress.used_portals) > 0, f"no portals selected to split with, make sure you are using -p"

    output = []
    for group in groups:
        portal_order, base_field = group
        tree = Tree(base_field)
        output += Ingress.render(tree.root, color_map, offset, onlyleaves)
    with open("./output.json", "w") as f:
        json.dump(output + other, f, indent=2)
    print("output.json created successfully")   
    
    pyperclip.copy(json.dumps(output + other))
    print("output coped to clipboard successfully")
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:c:ol", [])
    main(opts, args)