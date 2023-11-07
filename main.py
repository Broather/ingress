import json
import itertools
import getopt, sys

def my_argmin(lst: list) -> int:
    return list.index(lst, min(lst))

def parse_latlng(latlng: str):
    return [float(coordinate) for coordinate in latlng.split(",")]

def get_label(latLng: dict):
    for portal in Ingress.used_portals:
        if portal.get_latlng() == latLng:
            return portal.get_label()
        
    raise Exception(f"portal with {latLng} not found in used_portals")

class Portal:
    @staticmethod
    def from_IITC_polygon(IITC_polygon: dict):
        if IITC_polygon["type"] != "polygon":
            print(f"WARNING: from_IITC_polygon is attempting to parse type of {IITC_polygon['type']}")
        latLngs = IITC_polygon["latLngs"]
        return [Portal(get_label(latLng), float(latLng["lat"]), float(latLng["lng"])) for latLng in latLngs]

    def __init__(self, label: str, lat: float, lng: float) -> None:
        self.label = label
        self.lat = lat
        self.lng = lng

    def __repr__(self) -> str:
        return f"{self.label.replace(' ', '_')}"

    def get_latlng(self) -> dict:
        return {"lat": self.lat, "lng": self.lng}
    def get_label(self) -> str:
        return self.label
        
class Field:
    def __init__(self, p1: Portal, p2: Portal, p3: Portal, level: int, color_map: dict, offset: bool = False) -> None:
        self.children: list[Field] = []
        self.portals: list[Portal] = [p1,p2,p3]
        self.level: int = level
        self.color_map = color_map
        self.color: str = color_map.get(str(level), color_map["default"])
        self.offset = offset
    
    def __repr__(self) -> str:
        return str({
            "children": self.children,
            "portals": self.portals,
            "level": self.level,
            "color": self.color
            })

    def is_in(self, portal: Portal) -> bool:
        """
        Check if a Portal is inside a Field on Earth's surface. (made by GPT3)

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

        assert a.lat != b.lat and a.lng != b.lng, "trangle points have same value: (a and b)"
        assert a.lat != c.lat and a.lng != c.lng, "trangle points have same value: (a and c)"
        assert b.lat != c.lat and b.lng != c.lng, "trangle points have same value: (b and c)"

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
        return sum(list(map(self.is_in, Ingress.used_portals))) # [is_in(*(portal_1, polygon)), is_in(*(portal_2, polygon)), ...]

    def score(self, center_portal: Portal) -> int:
        """
        splits the Field on portal and returns a score based on the distribution of portals on each of the 3 new fields
        """
        fields = [Field(*outer_portals, center_portal, self.level + 1, self.color_map, self.offset) for outer_portals in itertools.combinations(self.portals, 2)]
        portal_counts = [field.count_portals() for field in fields]
        return max(portal_counts) - min(portal_counts)

    def split(self):
        potencial_center_portals = list(filter(self.is_in, Ingress.used_portals))

        scores = []
        for portal in potencial_center_portals:
            scores.append(self.score(portal))

        center_portal = potencial_center_portals[my_argmin(scores)]
        return [Field(*outer_portals, center_portal, self.level + 1, self.color_map, self.offset) for outer_portals in itertools.combinations(self.portals, 2)]

    def recursive_split(self):
        if self.count_portals() > 0:
            self.children = self.split()
            for child in self.children:
                child.recursive_split()

    def recursive_output(self, output: list = []):
        data = {
            "type": "polygon",
            "latLngs": [{"lat": portal.lat + self.level*0.0001*self.offset,"lng": portal.lng} for portal in self.portals],
            "color": self.color
        }
        output.append(data)
        for child in self.children:
            output = child.recursive_output(output)
        
        return output

            
class Tree:
    color_maps = {
                "rainbow" :
                    {"0": "#ff0000",
                    "1": "#ff7300",
                    "2": "#fff200",
                    "3": "#44ff00",
                    "4": "#00d0ff",
                    "5": "#1900ff",
                    "6": "#ff00f2",
                    "default": "#ffffff"},
                "ingress": 
                    {"0": "#f0ff20",
                    "1": "#ffb01c",
                    "2": "#ef8733",
                    "3": "#ff642c",
                    "4": "#c80425",
                    "5": "#ff0e82",
                    "6": "#b300ff",
                    "7": "#5100ff",
                    "default": "#ffffff"},
                "white": 
                    {"default": "#ffffff"}}
    def __init__(self, root_t: dict, color_map: str = "white", offset: bool = False) -> None:
        self.root = Field(*Portal.from_IITC_polygon(root_t), 0, Tree.color_maps[color_map], offset)
        self.root.recursive_split()

    def display(self, node=None):
        if node is None:
            node = self.root

        print(f"{'    ' * node.level}{node.portals}")

        for child in node.children:
            self.display(child)
    
    def output(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.root.recursive_output(), f, indent=2)

    def change_color(self, input: list[dict], _from: str, to: str) -> list[dict]:
        if input["color"] == _from:
            input["color"] = to
        
        return input

class Ingress:
    used_portals: list[Portal] = []

    @staticmethod
    def add_from_bkmrk(bkmrk: dict) -> None:
        for id in bkmrk:
            portal = Portal(bkmrk[id]["label"], *parse_latlng(bkmrk[id]["latlng"]))
            Ingress.used_portals.append(portal)
    
def help():
    print("Syntax: python main.py [-h] [-p comma_separated_list[<PV|...>]]")

def parse_input(input: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    parses the contents of input.json (which chould be full of IITC draw objects)

    Args:
    input (list[dict])

    Returns:
    tuple: (start, route, base_t, other)
    where:
        start: white markers
        route: white polylines
        base_t: white polygons
        other: any drawn elements that are not white
    """
    white = "#ffffff"

    start = [e for e in input if e["type"] == "marker" and e["color"] == white]
    route = [e for e in input if e["type"] == "polyline" and e["color"] == white]
    base_t = [e for e in input if e["type"] == "polygon" and e["color"] == white]
    other = [e for e in input if e["color"] != white]

    return (start, route, base_t, other)

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
    
    # option parsing part
    
    for o, a in opts:
        if o == "-h":
            help()
            sys.exit(2)
        elif o == "-p":
            for portal_group in a.split(","):
                with open(portal_group_map[portal_group.strip()], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])
        else:
            assert False, f"unsupported option: {o}"
    
    with open('./input.json', 'r') as f:
        input: list[dict] = json.load(f)

    start, route, base_t, other = parse_input(input)

    assert len(Ingress.used_portals) > 0, f"no portals selected to split with, make sure you are using -p"
    assert len(start) == 1, f"must have only one starting point, for now, {len(start)} detected"
    assert len(route) == 1, f"must have only one route, for now, {len(route)} detected"
    assert len(base_t) == 1, f"must have only one base triangle, for now, {len(base_t)} detected"

    tree = Tree(base_t[0], "ingress", True)
    tree.output("./output.json")
    print("output.json created successfully")   
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:", [])
    main(opts, args)