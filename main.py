from typing import Iterable, Callable
import json
import math
import numpy as np
import operator
import itertools
import getopt, sys
import pyperclip
import os
import math
from colorsys import hsv_to_rgb

def my_translate(value, from_min, from_max, to_min, to_max):
    # Figure out how 'wide' each range is
    leftSpan = from_max - from_min
    rightSpan = to_max - to_min

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - from_min) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return to_min + (valueScaled * rightSpan)

def my_argmin(lst: list) -> int:
    return lst.index(min(lst))

class Portal:
    def __init__(self, label: str, lat: float, lng: float, score: int = -1) -> None:
        self.label = label.replace(' ', '_')
        self.lat = lat
        self.lng = lng
        self.score = score

    def __hash__(self) -> int:
        return hash((self.lat, self.lng))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Portal):
            return self.lat == other.lat and self.lng == other.lng
        return False

    def __repr__(self) -> str:
        return f"{self.label}"

    def get_label(self) -> str:
        return self.label

    def get_lat(self) -> float:
        return self.lat

    def get_lng(self) -> float:
        return self.lng

    def get_latlng(self) -> dict:
        return {"lat": self.lat, "lng": self.lng}
    
    def get_score(self) -> int:
        return self.score
        
    def create_link(self, other: object):
        "returns Link|None"
        if isinstance(other, Portal):
            return Link(self, other)
        return None
    
    def is_contained_by_link(self, other: object) -> bool:
        if isinstance(other, Link):
            return self in other.portals
        return False

    def is_contained_by_field(self, other: object) -> bool:
        if isinstance(other, Field):
            return other.is_in(self)
        return False
    
    def get_link_other_portal(self, other: object):
        """returns Portal"""
        if isinstance(other, Link) and self.is_contained_by_link(other):
            if other.portals.index(self) == 0:
                return other.portals[1]
            elif other.portals.index(self) == 1:
                return other.portals[0]

    def find_middle(self, other_portal: object):
        """returns a portal which is in the middle of self and other_portal (note to self, only ever divide by 2 and nothing else)"""
        if not isinstance(other_portal, Portal):
            assert False, f"ERROR: Portal.transform recieved object of type {type(other_portal)}. Expected type Portal"

        assert self != other_portal, f"ERROR: Portal.transform recieved portals with same hash. self: {self}, other_portal: {other_portal}"

        # Create a Portal object based on the normalized vector and distance
        return Portal("anon", lat = (self.lat + other_portal.lat) / 2, lng = (self.lng + other_portal.lng) / 2)
            

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
    def from_IITC_marker_polyline(marker: dict, polyline: dict) -> tuple:
        """
        returns tuple[Portal]
        """
        start_portal = Ingress.find_portal_from_latLng(marker["latLng"])
        route_portals = tuple(Ingress.find_portal_from_latLng(latLng) for latLng in polyline["latLngs"])

        if route_portals[0] == start_portal:
            return route_portals
        elif route_portals[-1] == start_portal:
            return route_portals[::-1]
        else:
            assert False, "ERROR: a polyline's beginning and end does not match a marker's coordinates"

class Link:
    def __init__(self, frm: Portal, to: Portal, level: int = 0) -> None:
        self.portals = (frm, to)
        self.level = level
    
    def __repr__(self) -> str:
        return f"link{self.portals}"
    
    def __hash__(self) -> int:
        return hash(self.portals)

    def __eq__(self, other: object) -> bool:
        """returns True if other is a Link instance and conatains the same portals as self"""
        if isinstance(other, Link):
            return self.__hash__() == hash(other.portals) or self.__hash__() == hash(other.portals[::-1])
        return False

    def get_portals(self):
        return self.portals
    
    def get_level(self):
        return self.level

    def get_fields(self, other_links: Iterable):
        """returns all fields that would be created when adding self to the web of links (others)
        
        returns tuple[Field]"""
        if len(other_links) < 2: return tuple()

        output = []
        if all(map(lambda o: isinstance(o, Link), other_links)):
            for second_link, third_link in itertools.combinations(other_links, 2):
                if self.is_loop(second_link, third_link):
                   output.append(Field.from_links(self, second_link, third_link)) 
            return tuple(output)
        return tuple()

    def get_length(self) -> float:
        p1, p2 = self.portals
        return p1.distance(p2)

    def is_contained_by_field(self, other: object) -> bool:
        if isinstance(other, Field):
            return self in other.get_links()
        return False

    def is_touching(self, other: object)  -> bool:
        if isinstance(other, Link):
            return bool(set(self.portals).intersection(set(other.portals)))
        return False

    def intersection(self, other: object) -> set:
        if isinstance(other, Link) and self.is_touching(other):
            return set(self.portals).intersection(set(other.portals)) 
        return set()

    def is_loop(self, one, other) -> bool:
        if isinstance(one, Link) and isinstance(other, Link):
            return self.is_touching(one) and self.is_touching(other) and one.is_touching(other) and one.intersection(other).isdisjoint(self.portals)
        else:
            return False

class Field:
    def __init__(self, p1: Portal, p2: Portal, p3: Portal, level: int) -> None:
        self.portals: tuple[Portal] = (p1,p2,p3)
        self.level: int = level
        self.split_portal = None
        self.children: list[Field] = []
    
    def __repr__(self) -> str:
        return f"Field{self.portals}"

    @staticmethod
    def from_links(l1: Link, l2: Link, l3: Link):
        """returns Field"""
        portals = tuple(set(l1.portals).union(set(l2.portals), set(l3.portals)))
        assert len(portals) == 3, f"ERROR: Field.from_links recieved links that don't share exactly 3 portals, they share {len(portals)}"
        
        return Field(*portals, min(map(Link.get_level, (l1, l2, l3))))

    @staticmethod
    def from_route(route: tuple[Portal], level: int):
        """returns Field"""
        field = Field(*route[:3], level)
        field.portals = route

        return field

    @staticmethod
    def from_IITC_polygon(IITC_polygon: dict):
        """returns Field"""
        if IITC_polygon["type"] != "polygon":
            print(f"WARNING: from_IITC_polygon is attempting to parse type of {IITC_polygon['type']}")
        latLngs = IITC_polygon["latLngs"]
        portals = tuple(map(Ingress.find_portal_from_latLng, latLngs))
        assert len(portals) == 3, "ERROR: Field.from_IITC_polygon recieved a polygon made with NOT 3 portals"

        return Field(*portals, 0)

    def get_zelda_fields(self, subdivisions: int = 1) -> tuple:
        """returns tuple[Field] that look like the triforce thingy from Zelda"""
        fields = []
        for portal in self.portals:
            other_portal, another_portal = tuple(set(self.portals).difference([portal]))
            one_between_portal = portal.find_middle(other_portal)
            anoher_between_portal = portal.find_middle(another_portal)
            for _ in range(subdivisions-1):
                one_between_portal = portal.find_middle(one_between_portal)
                anoher_between_portal = portal.find_middle(anoher_between_portal)
            fields.append(Field(portal, one_between_portal, anoher_between_portal, self.get_level() + 1))
        
        return tuple(fields)
    
    def get_links(self) -> tuple[Link]:
        return tuple(itertools.starmap(Link, ((p1,p2,self.get_level()) for p1, p2 in itertools.combinations(self.portals, 2))))

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
    
    def get_portals(self) -> tuple[Portal]:
        """returns all portals that are inside of field (not including the 3 that make up the field)"""
        return tuple(filter(self.is_in, Ingress.used_portals))

    def get_portals_inclusive(self) -> tuple[Portal]:
        """returns all portals that are inside of field (including the 3 that make up the field)"""
        return self.get_portals() + self.portals
    
    def is_leaf(self):
        """returns True if field does not have children"""
        return len(self.children) == 0

    def has_portals(self):
        """returns True if field has portals within"""
        return len(self.get_portals()) > 0

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

        assert a.lat != b.lat and a.lng != b.lng, f"ERROR: Field has 2 Portals in the same location: ({a} and {b})"
        assert a.lat != c.lat and a.lng != c.lng, f"ERROR: Field has 2 Portals in the same location: ({a} and {c})"
        assert b.lat != c.lat and b.lng != c.lng, f"ERROR: Field has 2 Portals in the same location: ({b} and {c})"

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

    def homogen(self) -> Portal:
        """
        returns the portal that splits the field in the most uniform way
        """
        potential_split_portals = self.get_portals()
        for portal in potential_split_portals:
            fields = self.split(lambda _: portal)
            portal_counts = tuple(map(tuple.__len__, map(Field.get_portals, fields)))
            count_range = max(portal_counts) - min(portal_counts)
            portal.score = count_range
            
        best_score = min(map(Portal.get_score, potential_split_portals))
        best_portals = tuple(filter(lambda p: p.get_score() == best_score, potential_split_portals))
        if len(best_portals) > 1:
            for portal in best_portals:
                # TODO: does this masterpiece need changin'? Let me know in the comments below
                portal.score = Tree(self, lambda field: portal if field.get_level() == self.get_level() else field.homogen(), announce_self=False).get_variance()
            if min(map(Portal.get_score, best_portals)) != max(map(Portal.get_score, best_portals)):
                print("My efforts are not in vain!")
                print(f"When splitting {self} found multiple best split portals: {best_portals}")
                print(f"Picking {min(best_portals, key = Portal.get_score)} because it has a score of {min(best_portals, key = Portal.get_score).get_score()} compared to {list(map(Portal.get_score, best_portals))}")
            return min(best_portals, key = Portal.get_score)
        else:
            return min(potential_split_portals, key = Portal.get_score)

    def spiderweb(self) -> Portal:
        """return the portal that's nearest to any of self.portals"""
        potential_split_portals = self.get_portals()
        for portal in potential_split_portals:
            portal.score = min(map(portal.distance, self.portals)) 
        
        return min(potential_split_portals, key = Portal.get_score)
    
    @staticmethod
    def hybrid(subdivisions) -> Callable:
        def foo(self: Field) -> Portal:
            zelda_fields = self.get_zelda_fields(subdivisions = subdivisions)
            if any(map(Field.get_portals, zelda_fields)):
                return self.spiderweb()
            else:
                return self.homogen()
        return foo
    
    def split(self, split_method: Callable) -> tuple:
        """returns tuple[Field]"""
        split_portal = split_method(self)
        return tuple(Field(*link.portals, split_portal, self.get_level() + 1) for link in self.get_links())

class Tree:
    def __init__(self, root: Field, split_method: Callable, announce_self = True) -> None:
        self.root = root
        self.split_method = split_method

        while leaves_with_portals_in_them := tuple(filter(Field.has_portals, filter(Field.is_leaf, self.get_fields()))):
            for leaf in leaves_with_portals_in_them:
                leaf.children = leaf.split(self.split_method)

        if announce_self: print(self)
    
    def __repr__(self) -> str:
        return f"{self.root} mean lvl: {self.get_mean_level():.2f} stndrd deviation: {self.get_standard_deviation():.2f} links: {len(self.get_links())} fields: {len(self.get_fields())}"

    def get_fields(self, node: Field = None) -> tuple[Field]:
        """returns a tuple of all lower standing nodes (inclusive). Goes from root if node not given (made by GTP-3.5)"""

        if node is None:
            node = self.root

        fields = []
        queue = [node]

        while queue:
            current_node = queue.pop(0)
            fields.append(current_node)

            for child in current_node.children:
                queue.append(child)

        return tuple(fields)
    
    def get_mean_level(self) -> float:
        leaves = list(filter(Field.is_leaf, self.get_fields()))
        leaf_areas = np.array(list(map(Field.get_area, leaves)))
        normalized_leaf_areas = leaf_areas / self.root.get_area()

        assert 1-sum(normalized_leaf_areas) < .01, "the sum of normalized leaf areas is off by at least 1%"

        leaf_levels = np.array(list(map(Field.get_level, leaves)))
        # .1 of the entire area is 1 layer
        # .23 of the entire area is 2 layers
        # .52 of the entire area is 3 layers
        # ...
        # sum of multiplications
        return sum(normalized_leaf_areas * leaf_levels)
    
    def get_variance(self) -> float:
        leaves = tuple(filter(Field.is_leaf, self.get_fields()))
        leaf_areas = np.array(list(map(Field.get_area, leaves)))
        normalized_leaf_areas = leaf_areas / self.root.get_area()

        assert 1-sum(normalized_leaf_areas) < .01, "the sum of normalized leaf areas is off by at least 1%"

        leaf_levels = np.array(list(map(Field.get_level, leaves)))
        
        return sum(normalized_leaf_areas * (leaf_levels - self.get_mean_level())**2)
    
    def get_standard_deviation(self) -> float:
        return math.sqrt(self.get_variance())

    def get_level_range(self) -> tuple[int, int]:
        """returns a tree's range of levels or it's thicc-ness"""
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
    
    def get_links(self) -> tuple[Link]:
        all_links = map(Field.get_links, self.get_fields())
        return set(Ingress.flatten_iterable_of_tuples(all_links))

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

    color_maps = {"rainbow" : lambda variable: hsv_to_rgb(variable, 1, 1),
                "grayscale": lambda variable: hsv_to_rgb(1, 0, variable)}
    
    split_methods = {"hybrid": Field.hybrid(6),
                     "spiderweb": Field.spiderweb,
                     "homogen": Field.homogen}

    @staticmethod
    def simulate_step(active_portal: Portal, portals_to_link_to: tuple[Portal], previous_steps: list[tuple[Portal|Link|Field]]) -> tuple[Portal|Link|Field]:
        if len(portals_to_link_to) == 0: return (active_portal, )
        links = tuple(map(active_portal.create_link, portals_to_link_to))

        output = []
        for link in links:
            touching_links = tuple(filter(link.is_touching, Ingress.flatten_iterable_of_tuples(previous_steps + output)))
            fields = link.get_fields(touching_links)
            output.append((link,) + fields)

        return Ingress.flatten_iterable_of_tuples(output)

    @staticmethod
    def simulate_plan(plan: dict, step_per_link: bool = False) -> tuple[(tuple, tuple)]:
        """returns route: {bounding_box: tuple, steps: [[<Portal|Link|Field>]]}"""

        if step_per_link:
            all_steps = []
        else:
            all_steps = []
        output = {}
        routes: dict = plan["routes"]
        for route_title in routes:
            route_steps = []
            steps: dict = routes[route_title]["steps"]
            for active_portal in map(Ingress.find_portal, steps):
                if any(map(active_portal.is_contained_by_field, Ingress.flatten_iterable_of_tuples(all_steps+route_steps))): 
                    print(f"WARNING: in route {route_title} {list(routes.keys()).index(route_title)} portal {active_portal} {list(steps.keys()).index(active_portal.get_label())} is within a field when being linked from")
                portals_to_link_to = tuple(map(Ingress.find_portal, steps[active_portal.get_label()]["links"]))
                if len(portals_to_link_to) == 0: 
                    route_steps.append((active_portal,))
                    continue

                route_steps.append(Ingress.simulate_step(active_portal, portals_to_link_to, all_steps+route_steps))

            bounding_box = Ingress.bounding_box(Ingress.flatten_iterable_of_tuples(route_steps), grow_to_square=True)
            all_steps.extend(route_steps)
            output.append((bounding_box, route_steps))
        
        return tuple(output)
        
    @staticmethod
    def draw_level(display_level: int, bounding_box: tuple[Portal, Portal], field_level: int = -1) -> tuple[Field]:
        if field_level < 0: field_level = display_level-1
        
        digit_bounding_boxes = tuple([bounding_box])
        digits = [char for char in str(display_level)]
        # NOTE: does not support any number of digits, but when the time comes I'll implement split_bounding_box_vertically(splits = 2)
        if len(digits) > 1:
            digit_bounding_boxes = Ingress.split_bounding_box_vertically(digit_bounding_boxes[0])
                
        output = []
        for digit, bounding_box in zip(digits, digit_bounding_boxes):
            tl, br = bounding_box
            lat_unit = (br.lat - tl.lat)/5
            lng_unit = (br.lng - tl.lng)/5
            portal_grid = dict([((lat_offset, lng_offset), Portal(str((lat_offset, lng_offset)), tl.lat + lat_offset*lat_unit, tl.lng + lng_offset*lng_unit)) for lng_offset in range(5) for lat_offset in range(5)])
            if digit == "0":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((1,0),(0,1),(1,3),(4,2),(3,1),(0,2),(1,4),(3,4),(4,3),(3,0)))),
                    field_level))
            elif digit == "1":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((1,0),(1,0),(0,1),(0,3),(3,3),(3,4),(4,4),(4,0),(3,0),(3,1),(1,1)))),
                    field_level))
            elif digit == "2":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((1,0),(0,1),(0,3),(1,4),(2,4),(3,2),(3,4),(4,4),(4,0),(3,0),(2,2),(1,2),(2,1)))),
                    field_level))
            elif digit == "3":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((0,0),(0,3),(1,4),(2,3),(3,4),(4,3),(4,1),(2,0),(4,2),(3,3),(2,2),(2,1),(1,0),(2,2),(1,3),(0,3),(1,1)))),
                    field_level))
            elif digit == "4":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((2,0),(0,2),(0,4),(1,4),(2,3),(0,3),(2,1),(2,4),(3,4),(3,3),(4,3),(4,2),(3,2),(3,0)))),
                    field_level))
            elif digit == "5":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((0,0),(0,4),(1,4),(1,1),(2,1),(1,2),(2,4),(3,4),(4,3),(4,0),(3,0),(3,3),(2,3),(3,2),(3,1),(2,0)))),
                    field_level))
            elif digit == "6":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((1,0),(0,1),(0,3),(1,4),(1,3),(0,2),(1,1),(2,4),(3,4),(3,1),(2,1),(3,4),(4,3),(4,1),(3,0)))),
                    field_level))
            elif digit == "7":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((0,0),(0,4),(1,4),(2,2),(2,3),(3,3),(3,2),(4,2),(4,1),(3,1),(3,0),(2,0),(2,1),(1,3),(1,1)))),
                    field_level))
            elif digit == "8":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((1,0),(0,1),(1,2),(2,1),(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(4,3),(4,2),(3,1),(2,2),(2,3),(3,3),(4,2),(4,1),(3,0),(2,1)))),
                    field_level))
            elif digit == "9":
                output.append(Field.from_route(
                    tuple(map(portal_grid.get, ((1,0),(0,1),(0,3),(1,4),(3,4),(4,3),(4,1),(3,0),(2,0),(4,2),(3,3),(1,3),(1,1),(0,2),(1,3),(2,1)))),
                    field_level))
        return tuple(output)
    
    @staticmethod
    def create_legend(fields: list[Field], onlyleaves: bool) -> list[Field]:
        tl, br = Ingress.bounding_box(fields)
        tr, bl = (Portal("top right", tl.lat, br.lng), Portal("bottom left", br.lat, tl.lng))

        min_field_level = min(map(Field.get_level, fields))
        max_field_level = max(map(Field.get_level, fields))

        # TODO: declare start and end differently based on where to put legend and what order
        # delta_longitude = abs(tl.lng - br.lng)
        delta_latitude = abs(br.lat - tl.lat)
        segment_length = delta_latitude/(max_field_level+1-min_field_level)
        PADDING = delta_latitude*.05
        output = []
        for level in range(min_field_level, max_field_level+1):
            offset = segment_length * (level - min_field_level)
            item_tl = Portal(f"legend item's top left", bl.lat+offset+segment_length, bl.lng-(segment_length*1.87)-PADDING)
            item_br = Portal(f"legend item's bottom right", bl.lat+offset, bl.lng-PADDING)
            output.append(Ingress.draw_level(level+1, (item_tl, item_br)))

            if not onlyleaves:
                for lvl in range(level):
                    output.append(Ingress.draw_level(level+1, (item_tl, item_br), field_level = lvl))
                    
        return Ingress.flatten_iterable_of_tuples(output)
    
    @staticmethod
    def bounding_box(objects: list[Portal|Link|Field], grow_to_square: bool = False, padding: bool = False) -> tuple[Portal, Portal]:
        # TODO: add some debuging squares or something to see how it's growing to square in different situations
        portals = tuple(filter(lambda o: isinstance(o, Portal), objects))
        links = tuple(filter(lambda o: isinstance(o, Link), objects))
        fields = tuple(filter(lambda o: isinstance(o, Field), objects))

        all_portals = portals + Ingress.flatten_iterable_of_tuples(map(Link.get_portals, links)) + Ingress.flatten_iterable_of_tuples(map(Field.get_portals_inclusive, fields))
        
        all_latitudes = list(map(Portal.get_lat, all_portals))
        all_longitudes = list(map(Portal.get_lng, all_portals))
        tl = Portal("top left", lat = max(all_latitudes), lng = min(all_longitudes))
        br = Portal("bottom right", lat = min(all_latitudes), lng = max(all_longitudes))

        if grow_to_square:
            delta_latitude = abs(br.lat - tl.lat)
            delta_longitude = abs(tl.lng - br.lng)
            lat_against_lng = delta_latitude/delta_longitude
            if lat_against_lng > 1/1.88:
                # lng too smol
                difference = delta_latitude * 1.88 - delta_longitude
                # print(f"bounding box isn't square, latitude bigger by {difference} correcting")
                tl.lng -= difference/2
                br.lng += difference/2
            elif lat_against_lng < 1/1.88:
                # lat too smol
                difference = delta_longitude/1.88 - delta_latitude
                # print(f"bounding box isn't square, longitude bigger by {difference} correcting")
                tl.lat += difference/2
                br.lat -= difference/2
            else:
                # deltas identical, box is a square, no action reqired
                pass

        if padding:
            # add 5% padding on all sides
            delta_latitude = abs(br.lat - tl.lat)
            delta_longitude = abs(tl.lng - br.lng)

            br.lat += delta_latitude * .05
            tl.lat -= delta_latitude * .05

            br.lng -= delta_longitude * .05
            tl.lng += delta_longitude * .05
            
        return (tl, br)
    
    @staticmethod
    def split_bounding_box_vertically(bounding_box: tuple[Portal, Portal]):
        tl, br = bounding_box
        tr, bl = (Portal("top right", tl.lat, br.lng), Portal("bottom left", br.lat, tl.lng))
        right_bb_tl = tl.find_middle(tr)
        left_bb_br = bl.find_middle(br)

        return ((tl, left_bb_br), (right_bb_tl, br))
    @staticmethod
    def flatten_iterable_of_tuples(iterable: Iterable) -> tuple:
        return tuple(element for tuple in iterable for element in tuple)

    @staticmethod
    def flatten_iterable_of_sets(iterable: Iterable) -> set:
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
    def find_portal_from_latLng(latLng: dict) -> Portal:
        for portal in Ingress.used_portals:
            if portal.get_latlng() == latLng:
                return portal
            
        assert False, f"portal with {latLng} not found in used_portals"
    
    @staticmethod
    def parse_lat_comma_lng(latlng: str):
        return [float(coordinate) for coordinate in latlng.split(",")]

    @staticmethod
    def parse_latLng(latLng: dict) -> list[float]:
        return [float(latLng["lat"]), float(latLng["lng"])]
    
    @staticmethod
    def parse_input(IITC_elements: list[dict]) -> tuple[list[Field], list[list[Portal]], list[dict]]:
        """
        parses the contents of input.json (which should be full of IITC copy/paste)

        Args:
        input: list[dict]

        Returns (base_fields, routes, other)
        """
        white = "#ffffff"
                
        markers = [e for e in IITC_elements if e["type"] == "marker" and e["color"] == white]
        polylines = [e for e in IITC_elements if e["type"] == "polyline" and e["color"] == white]
        assert len(markers) == len(polylines), f"ERROR: amount of markers and polylines should match, markers: {len(markers)}, polylines: {len(polylines)}"
        # TODO: assert that every marker is at the start of end of any polyline

        polygons = [e for e in IITC_elements if e["type"] == "polygon" and e["color"] == white]

        base_fields = []
        routes = []

        for polygon in polygons:
            base_fields.append(Field.from_IITC_polygon(polygon))

        # TODO: still relies on correct definition order
        for marker, polyline in zip(markers, polylines):
            routes.append(Portal.from_IITC_marker_polyline(marker, polyline))

        other = [e for e in IITC_elements if e["color"] != white]

        return (base_fields, routes, other)

    @staticmethod
    def render(objects: tuple[Portal|Link|Field], color_map_function: Callable, output: list = None) -> list[dict]:
        if output == None: output = []
        if len(objects) == 0: return []
        
        portals = tuple(filter(lambda o: isinstance(o, Portal), objects))
        links = tuple(filter(lambda o: isinstance(o, Link), objects))
        fields = tuple(filter(lambda o: isinstance(o, Field), objects))

        # TODO: using Portal.score is not elegant
        max_level = max(Ingress.flatten_iterable_of_tuples((map(Portal.get_score, portals), map(Link.get_level, links), map(Field.get_level, fields))))
        # TODO: or maybe for object in objects and isinstance the type?
        # TODO: or maybe Link and Field inherits from parent with get_level, get_IITC_type and get_portals 

        for portal in portals:
            output.append({
                "type": "marker",
                "latLng": {"lat": portal.lat, "lng": portal.lng},
                "color": "#{:02x}{:02x}{:02x}".format(*[int(my_translate(v, 0,1, 0,255)) for v in color_map_function(portal.get_score())])
                })
        for link in links:
            mapped_level = my_translate(link.get_level(), 0, max_level + 1, 0, 1)
            output.append({
                "type": "polyline",
                "latLngs": [{"lat": portal.lat, "lng": portal.lng} for portal in link.portals],
                "color": "#{:02x}{:02x}{:02x}".format(*[int(my_translate(v, 0,1, 0,255)) for v in color_map_function(mapped_level)])
            })
            for portal in link.get_portals():
                output.append({
                "type": "marker",
                "latLng": {"lat": portal.lat, "lng": portal.lng},
                "color": "#{:02x}{:02x}{:02x}".format(*[int(my_translate(v, 0,1, 0,255)) for v in color_map_function(mapped_level)])
                })    
        for field in fields:
            mapped_level = my_translate(field.get_level(), 0, max_level + 1, 0, 1)
            output.append({
                "type": "polygon",
                "latLngs": [{"lat": portal.lat, "lng": portal.lng} for portal in field.portals],
                "color": "#{:02x}{:02x}{:02x}".format(*[int(my_translate(v, 0,1, 0,255)) for v in color_map_function(mapped_level)])
            })
        
        return output

    @staticmethod
    def create_plan(routes: list[tuple[Portal]], trees: list[Tree]) -> dict:
        AVERAGE_WALKING_SPEED = 84 # meters/minute
        COOLDOWN_BETWEEN_HACKS = 5 # minutes

        all_fields: tuple[Field] = Ingress.flatten_iterable_of_tuples(map(Tree.get_fields, trees))
        all_links: set[Link] = set(Ingress.flatten_iterable_of_tuples(map(Tree.get_links, trees)))

        all_portals = set(Ingress.flatten_iterable_of_tuples(map(Field.get_portals_inclusive, all_fields)))
        route_portals = set(Ingress.flatten_iterable_of_tuples(routes))

        # warn if there are any portals missed by routes
        if (missed_portals_count := len(all_portals) - len(route_portals)) > 0:
            print(f"WARNING: routes missed {missed_portals_count} portal/-s, plan/-s might not be accurate. Missing portals {all_portals.difference(route_portals)}")

        plan_routes = {}
        visited_portals = set()
        created_links = set()
        for route in routes:
            steps = {}
            SBUL_count = 0
            route_links = set()
            for active_portal in route:
                visited_portals.add(active_portal)
                
                connected_links = set(filter(active_portal.is_contained_by_link, all_links))
                outbound_links = [link for link in connected_links.difference(created_links) if visited_portals.issuperset(link.portals)]
                # outbound_links = list(filter(visited_portals.issuperset, links.difference(created_links)))
                route_links.update(outbound_links)
                created_links.update(outbound_links)

                SBUL_count += len(outbound_links) // 9
                if len(outbound_links) > 24: print(f"WARNING: at portal {active_portal.get_label()} outbound link count exceeds 2 SBULs: {len(outbound_links)}")

                # TODO: I hope I have the motivation to rewrite this masterpiece
                # primary sort criteria: filter all the fields that the link is part of and get their levels and pick the min level
                # secondary sort criteria: longest link first
                link_order = sorted(outbound_links, key = lambda outbound_link: (min(map(Field.get_level, filter(outbound_link.is_contained_by_field, all_fields))), -outbound_link.get_length()))
                # link_order = sorted(outbound_links, key = lambda outbound_link: [field for field in all_fields if outbound_link in field.get_links()])
                # link_order = sorted(outbound_links, key = lambda l: (min(map(Field.get_level, filter(lambda f: l in f.get_links(), all_fields))), -tuple(l)[0].distance(tuple(l)[1])))
                
                steps[active_portal.get_label()] = {
                "keys": len(connected_links)-len(outbound_links),
                # TODO: this one's a bit rought aswell
                "links": list(map(Portal.get_label, map(active_portal.get_link_other_portal, link_order)))
                }

            route_length = round(sum(itertools.starmap(Portal.distance, itertools.pairwise(route))), 2)
            total_keys_required = len(route_links)

            plan_routes[f"{route[0]}...{route[-1]}"] = {
                "SBULs_required": SBUL_count,
                "route_length_(meters)": route_length,
                "route_keys_required": total_keys_required,
                "estimated_time_to_complete_(minutes)": round(route_length / AVERAGE_WALKING_SPEED + total_keys_required * COOLDOWN_BETWEEN_HACKS, 2),
                "steps": steps
                }
        
        plan = {"total-total_keys_required": len(all_links),
                "routes": plan_routes}
        return plan

    @staticmethod
    def add_from_bkmrk_file(bkmrk_file_path: str) -> None:
        with open(bkmrk_file_path, "r", encoding='utf-8') as f:
            bkmrk = json.load(f)['portals']['idOthers']['bkmrk']
        for id in bkmrk:
            portal = Portal(bkmrk[id]["label"], *Ingress.parse_lat_comma_lng(bkmrk[id]["latlng"]))
            Ingress.used_portals.append(portal)
    
def help(first_time = False):
    if first_time:
        print("Looks like it's your first time launching main.py")

        open("./input.json", "w").close()
        print("input.json created")

        print("""Install IITC for your browser at http://iitc.me and go to http://www.intel.ingress.com
        For more details on which IITC extentions to install refer to the README.md file\n\n""")
        
    print("Syntax: python main.py [-hol] [-p comma_separated_list[<PV|...|ALL>]] [-c <rainbow|grayscale>]")
    print("""
    Options:
        h: calls this help function
        p: defines which portal groups to add to Ingress.used_portals (only way I could think of to get portal data here)
        a: adds all defined portal groups to Ingress.used_portals
        c: selects the color map to use, (default: rainbow for all layers)
        s: selects the split method to use, (default: hybrid6)
        l: display only the leaf fields, aka the top most layer of each section
        --noplan: skip making a plan
        --nosim: skip simulating created plan
        --nolegend: skip including a legend in output.json
        --input: spiecify path to IITC json file
    """)

def main(opts: list[tuple[str, str]], args):
    # defaults
    make_plan = True
    make_simulation = True
    make_legend = True
    split_method = Ingress.split_methods["hybrid"]
    onlyleaves = False
    color_map = Ingress.color_maps["rainbow"]
    input_path = "./input.json"
    
    # option parsing
    for o, a in opts:
        if o == "-h":
            help()
            sys.exit(2)
        elif o == "-p":
            if a.lower() == "all":
                for file in Ingress.portal_group_map.values():
                    Ingress.add_from_bkmrk_file(file)
            else:
                for portal_group in a.split(","):
                    Ingress.add_from_bkmrk_file(Ingress.portal_group_map[portal_group.strip()])
        elif o == "-c":
            if (color_map := Ingress.color_maps.get(a)) == None:
                print(f"ERROR: color map {a} not recognised, your options are {Ingress.color_maps.keys()}")
                help()
                sys.exit(2)
        elif o == "-s":
            if (split_method := Ingress.split_methods.get(a)) == None:
                print(f"ERROR: split method {a} not recognised, your options are {Ingress.split_methods.keys()}")
                help()
                sys.exit(2)
        elif o == "-l":
            onlyleaves = True
        elif o == "--noplan":
            make_plan = False
        elif o == "--nosim":
            make_simulation = False
        elif o == "--nolegend":
            make_legend = False
        elif o == "--input":
            input_path = a
        else:
            assert False, f"ERROR: unparsed option: {o}"
    
    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"
    print(f"{len(Ingress.used_portals)} portals in Ingress.used_portals")
    try:
        with open(input_path, 'r') as f:
            input: list[dict] = json.load(f)
    except FileNotFoundError:
        help(first_time = True)
        return
    except json.decoder.JSONDecodeError:
        print("input.json is empty, make sure to copy/paste whatever IITC gives you into input.json (and save it)")
        return

    # base_fields get split, routes get applied to them to make a plan, other just gets appended to output
    base_fields, routes, other = Ingress.parse_input(input)

    output = []
    all_trees: tuple[Tree] = tuple(itertools.starmap(Tree, [(base_field, split_method) for base_field in base_fields]))
    all_fields = []
    for tree in all_trees:
        fields = tree.get_fields()

        if onlyleaves:
            fields = list(filter(Field.is_leaf, fields))
        
        all_fields.extend(fields)

    output.extend(Ingress.render(all_fields, color_map))
    # output.extend(Ingress.render(Ingress.merge_iterable_to_set(map(lambda f: f.get_zelda_fields(subdivisions=4), all_fields)), color_map))

    if make_legend:
        legend = Ingress.create_legend(all_fields, onlyleaves)
        output.extend(Ingress.render(legend, color_map))

    Ingress.output_to_json(output + other, "./output.json")
    Ingress.copy_to_clipboard(output + other)

    if make_plan:
        plan = Ingress.create_plan(routes, all_trees)
        Ingress.output_to_json(plan, "./plan.json")
        if make_simulation:
            Ingress.simulate_plan(plan)

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:c:s:l", ["noplan", "nosim", "nolegend"])
    main(opts, args)