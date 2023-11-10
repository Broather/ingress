import json
import itertools
import getopt, sys

def my_argmin(lst: list) -> int:
    return list.index(lst, min(lst))

class Portal:
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
        print(f"marker: {marker}")
        print(f"polyline: {polyline}")
        start_portal = Portal(Ingress.get_label(marker["latLng"]), *Ingress.parse_latLng(marker["latLng"]))
        
        # return [Portal()]

    @staticmethod
    def from_IITC_polygon(IITC_polygon: dict):
        """
        returns list[Portal]
        """
        if IITC_polygon["type"] != "polygon":
            print(f"WARNING: from_IITC_polygon is attempting to parse type of {IITC_polygon['type']}")
        latLngs = IITC_polygon["latLngs"]
        return [Portal(Ingress.get_label(latLng), *Ingress.parse_latLng(latLng)) for latLng in latLngs]

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
    def __init__(self, p1: Portal, p2: Portal, p3: Portal, level: int) -> None:
        self.children: list[Field] = []
        self.portals: list[Portal] = [p1,p2,p3]
        self.level: int = level
    
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
    def __init__(self, root_t: dict) -> None:
        self.root = Field(*Portal.from_IITC_polygon(root_t), 0)
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
    def parse_input(input: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """
        parses the contents of input.json (which should be full of IITC copy/paste)

        Args:
        input (list[dict])

        Returns:
        tuple: (portal_order, base_field, other)
        where:
            portal_order: list[Portal]
            base_field: Field
            other: any drawn elements that are not white
        """
        white = "#ffffff"

        markers = [e for e in input if e["type"] == "marker" and e["color"] == white]
        polylines = [e for e in input if e["type"] == "polyline" and e["color"] == white]
        assert len(markers) == len(polylines), f"ERROR: amount of markers and polylines should match markers: {len(markers)}, polylines: {len(polylines)}"
        for marker, polyline in zip(markers, polylines):
            # TODO: somewhere above figure out groups and parse as a group
            portal_order = Portal.from_IITC_marker_polyline(marker, polyline)

        polygons = [e for e in input if e["type"] == "polygon" and e["color"] == white]
        for polygon in polygons:
            # TODO: somewhere above figure out groups and parse as a group
            base_field = Field(*Portal.from_IITC_polygon(polygon), 0)

        other = [e for e in input if e["color"] != white]

        return (portal_order, base_field, other)

    @staticmethod
    def add_from_bkmrk(bkmrk: dict) -> None:
        for id in bkmrk:
            portal = Portal(bkmrk[id]["label"], *Ingress.parse_lat_comma_lng(bkmrk[id]["latlng"]))
            Ingress.used_portals.append(portal)
            
    @staticmethod
    def render(field: Field, color_map: dict, offset: bool, top: bool, output: list = []) -> list[dict]:
        data = {
            "type": "polygon",
            "latLngs": [{"lat": portal.lat + 0.0001*offset*field.level, "lng": portal.lng} for portal in field.portals],
            "color": color_map.get(str(field.level), color_map["default"])
        }
        output.append(data)
        for child in field.children:
            output = Ingress.render(child, color_map, offset, top, output)
        
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

    route, base_t, other = Ingress.parse_input(input)
    

    assert len(Ingress.used_portals) > 0, f"no portals selected to split with, make sure you are using -p"
    assert len(route) == 1, f"must have only one route, for now, {len(route)} detected"
    assert len(base_t) == 1, f"must have only one base triangle, for now, {len(base_t)} detected"

    tree = Tree(base_t[0])
    with open("./output.json", "w") as f:
        json.dump(Ingress.render(tree.root, Ingress.color_maps["rainbow"], True, True), f, indent=2)
    
    # tree.output("./output.json")
    print("output.json created successfully")   
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:", [])
    main(opts, args)