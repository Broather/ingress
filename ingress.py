from typing import Iterable, Callable
import math
import json
import numpy as np
import re
from itertools import starmap, combinations, pairwise
import os
import pyperclip
import colorsys

def my_translate(value, from_min, from_max, to_min, to_max):
    """(made by GPT-3.5)"""
    # Figure out how 'wide' each range is
    left_span = from_max - from_min
    right_span = to_max - to_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - from_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return to_min + (value_scaled * right_span)
    
class Portal:
    """represents a point on Earth's surface"""
    def __init__(self, label: str, lat: float, lng: float, value: int = -1) -> None:
        self.label = label.replace(' ', '_')
        self.lat = lat
        self.lng = lng
        self.value = value

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

    def get_latLng(self) -> dict:
        return {"lat": self.lat, "lng": self.lng}

    def get_value(self) -> int:
        """return a convenience value tied to a Portal object"""
        return self.value

    def create_link(self, other: object):
        "create a link from self to other"
        if isinstance(other, Portal):
            return Link(self, other)
        return None

    def is_part_of_link(self, other: object) -> bool:
        if isinstance(other, Link):
            return self in other.portals
        return False

    def is_under_field(self, other: object) -> bool:
        """return True if self is under other: Field.

        Portals that make up the Field are conidered OUTSIDE the Field"""
        if isinstance(other, Field):
            return other.is_in(self)
        return False

    def get_adjacent_portal(self, link: object):
        """return the adjacent Portal that is part of link"""
        assert isinstance(link, Link), "link is not instance of Link"
        if not self.is_part_of_link(link):
            assert False, "ERROR: given link does not contain self, so cannot return other portal"

        if link.portals.index(self) == 0:
            return link.portals[1]
        elif link.portals.index(self) == 1:
            return link.portals[0]

    def find_middle(self, other: object):
        """return the mid point between self and other as a Portal object.

        This is a cruicial function used by get_zelda_fields.
        It's important to only ever divide distance by 2 otherwise
        there will be problems with decimal precision.
        """
        if not isinstance(other, Portal):
            assert False, f"ERROR: Expected type Portal, recieved type {type(other)}"

        assert self != other, f"ERROR: recieved portal identical to self {other}"

        # Create a Portal object based on the normalized vector and distance
        return Portal("anon",
                        lat = (self.lat + other.lat) / 2,
                        lng = (self.lng + other.lng) / 2)


    def distance(self, other_portal: object) -> float:
        """Calculate the Haversine distance between 2 Portals in meters"""
        if isinstance(other_portal, Portal):
            # Convert latitude and longitude from degrees to radians
            lat1, lng1, lat2, lng2 = map(math.radians, [self.lat, self.lng, other_portal.lat, other_portal.lng])

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
        """create a tuple of portals that polyline goes through (marker on first portal)"""
        start_portal = Ingress.find_portal_from_latLng(marker["latLng"])
        route_portals = tuple(Ingress.find_portal_from_latLng(latLng) for latLng in polyline["latLngs"])

        if route_portals[0] == start_portal:
            return route_portals
        if route_portals[-1] == start_portal:
            return route_portals[::-1]
        assert False, "ERROR: a polyline's beginning and end does not match a marker's coordinates"

class Link:
    def __init__(self, frm: Portal, to: Portal, level: int = 0) -> None:
        self.portals = (frm, to)
        self.level = level

    def __repr__(self) -> str:
        return f"Link{self.portals}"

    def __hash__(self) -> int:
        return hash(self.portals)

    def __eq__(self, other: object) -> bool:
        """return True if other is a Link object with the same portals as self, otherwise False"""
        if isinstance(other, Link):
            return self.__hash__() == hash(other.portals) or self.__hash__() == hash(other.portals[::-1])
        return False

    def get_portals(self):
        return self.portals

    def get_frm(self) -> Portal:
        return self.portals[0]
        
    def get_to(self):
        return self.portals[1]

    def get_level(self):
        return self.level

    def get_resulting_fields(self, context: Iterable):
        """return a touple of 0 or 1 or 2 fields that would be created when adding self to the web of links: context"""
        links = list(filter(Link.__instancecheck__, context))
        touching_links = list(filter(self.is_touching, links))
        if len(touching_links) < 2: return tuple()

        potential_fields = []
        for second_link, third_link in combinations(touching_links, 2):
            if self.is_loop(second_link, third_link):
                potential_fields.append(Field.from_links(self, second_link, third_link))
        
        # split fields to 2 sides and return field with max get_area from each side
        one_side = list(filter(self.is_on_side, potential_fields))
        other_side = list(filter(lambda p: not self.is_on_side(p), potential_fields))
        
        if len(one_side) != 0 and len(other_side) != 0:
            return (max(one_side, key=Field.get_area), max(other_side, key=Field.get_area))
        elif len(one_side) == 0 and len(other_side) != 0:
            return (max(other_side, key=Field.get_area), )
        elif len(one_side) != 0 and len(other_side) == 0:
            return (max(one_side, key=Field.get_area), )
        else:
            return tuple()

    def get_length(self) -> float:
        """return length in meters"""
        p1, p2 = self.portals
        return p1.distance(p2)

    def is_within_field(self, other: object) -> bool:
        if isinstance(other, Field):
            return any(map(other.is_in, self.portals))

    def is_part_of_field(self, other: object) -> bool:
        if isinstance(other, Field):
            return self in other.get_links()
        return False

    def is_touching(self, other: object)  -> bool:
        if isinstance(other, Link):
            return bool(set(self.portals).intersection(set(other.portals)))
        return False

    def is_loop(self, one: object, other: object) -> bool:
        if isinstance(one, Link) and isinstance(other, Link):
            return all([self.is_touching(one),
                        self.is_touching(other),
                        one.is_touching(other),
                        one.intersection(other).isdisjoint(self.portals)])
        return False
    
    def is_on_side(self, other: object) -> bool:
        if isinstance(other, Field) and self.is_part_of_field(other):
            adjacent_portal = set(other.portals).difference(set(self.portals)).pop()
            a, b = self.portals
            position = (b.lng - a.lng) * (adjacent_portal.lat - a.lat) - (b.lat - a.lat) * (adjacent_portal.lng - a.lng)
            return position > 0
            # position = (Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax)

    def intersection(self, other: object) -> set:
        if isinstance(other, Link) and self.is_touching(other):
            return set(self.portals).intersection(set(other.portals))
        return set()


class Field:
    """represents an area between 3 distinct points on Earth's surface"""
    def __init__(self, p1: Portal, p2: Portal, p3: Portal, level: int = 0) -> None:
        self.portals: tuple[Portal] = (p1,p2,p3)
        self.level: int = level
        self.split_portal = None
        self.children: list[Field] = []

    def __repr__(self) -> str:
        return f"Field{self.portals}"

    @staticmethod
    def from_links(l1: Link, l2: Link, l3: Link):
        """create a field from 3 links if possible"""
        assert l1.is_loop(l2, l3), f"ERROR: links don't make a closed loop"

        portals = tuple(set(l1.portals).union(set(l2.portals), set(l3.portals)))
        return Field(*portals, min(map(Link.get_level, (l1, l2, l3))))

    @staticmethod
    def from_route(route: tuple[Portal], level: int):
        """create a field with more than 3 points
        
        Only used to make the legend's numbers."""
        field = Field(*route[:3], level)
        field.portals = route

        return field

    @staticmethod
    def from_IITC_polygon(IITC_polygon: dict):
        """creates a field from an IITC polygon with 3 points"""
        assert IITC_polygon["type"] == "polygon", f"ERROR: can't parse element of type {IITC_polygon['type']}"
        assert len(IITC_polygon["latLngs"]) == 3, f"ERROR: can't make field from polygon with {len(IITC_polygon['latLngs'])} points"

        portals = tuple(map(Ingress.find_portal_from_latLng, IITC_polygon["latLngs"]))
        return Field(*portals)

    def get_zelda_fields(self, subdivisions: int = 1) -> tuple:
        """return 3 fields that together look like the triforce logo from Zelda"""
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
        return tuple(starmap(Link, ((p1,p2,self.get_level()) for p1, p2 in combinations(self.portals, 2))))

    def get_MU(self) -> int:
        MU_COEFICIENT = 4.25 * 10**-5
        return math.ceil(MU_COEFICIENT*self.get_area())

    def get_area(self) -> float:
        sides = list(starmap(Portal.distance, combinations(self.portals, 2)))

        # Semi-perimeter of the triangle
        s = sum(sides) / 2

        # Heron's formula for area of a triangle
        area = math.sqrt(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]))

        return area

    def get_level(self) -> int:
        return self.level

    def get_portals(self) -> tuple[Portal]:
        """return all portals that are inside of field (not including the 3 that make up the field)"""
        return tuple(filter(self.is_in, Ingress.used_portals))

    def get_portals_inclusive(self) -> tuple[Portal]:
        """return all portals that are inside of field (including the 3 that make up the field)"""
        return self.get_portals() + self.portals

    def is_leaf(self):
        """return True if field does not have children, otherwise False"""
        return len(self.children) == 0

    def has_portals(self):
        """return True if field has portals within it, otherwise False"""
        return len(self.get_portals()) > 0

    def is_in(self, portal: Portal) -> bool:
        """Check if a Portal is under Field (self) (made by GPT-3.5)"""
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
        """return the portal that splits the field in the most uniform way"""
        potential_split_portals = self.get_portals()
        for portal in potential_split_portals:
            fields = self.split(lambda _: portal)
            portal_counts = tuple(map(tuple.__len__, map(Field.get_portals, fields)))
            count_delta = max(portal_counts) - min(portal_counts)
            portal.value = count_delta

        return min(potential_split_portals, key = Portal.get_value)

    def spiderweb(self) -> Portal:
        """return the portal that's nearest to any of self.portals"""
        return min(self.get_portals(), key = lambda p: min(map(p.distance, self.portals)))

    @staticmethod
    def hybrid(subdiv) -> Callable:
        """return a split method based on subdivisions of zelda fields"""
        def funk(self: Field) -> Portal:
            """return a portal just like spiderweb if there are portals in any of the 3 zelda fields
            
            A mix between spiderweb (subdiv = 2) and homogen (subdiv = 9).
            The higher the subdivisions value, the less likely to find portals within them
            and more likely for the function to behave like homogen."""
            zelda_fields = self.get_zelda_fields(subdivisions = subdiv)
            if any(map(Field.get_portals, zelda_fields)):
                return self.spiderweb()
            return self.homogen()
        return funk
    
    def center(self) -> Portal:
        """return the portal that's nearest to field's center"""
        c = Portal("center", sum(map(Portal.get_lat, self.portals))/3, sum(map(Portal.get_lng, self.portals))/3)
        return min(self.get_portals(), key=c.distance)

    def spiral(self) -> Portal:
        """return the portal that's nearest to any of self.portals that aren't a part of self.parent.portals"""
        

    def split(self, split_method: Callable) -> tuple:
        """creates 3 fields on top of self with a portal chosen by split method"""
        split_portal = split_method(self)
        return tuple(Field(*link.portals, split_portal, self.get_level() + 1) for link in self.get_links())

class BoundingBox:
    def __init__(self, objects: list[Portal|Link|Field], grow_to_square: bool = False, padding: bool = False) -> tuple[Portal, Portal]:
        # TODO: add some debugging squares or something to see how it's growing to square in different situations
        portals = tuple(filter(lambda o: isinstance(o, Portal), objects))
        links = tuple(filter(lambda o: isinstance(o, Link), objects))
        fields = tuple(filter(lambda o: isinstance(o, Field), objects))

        all_portals = Ingress.flatten_tuples(portals,
                    Ingress.flatten_iterable_of_tuples(map(Link.get_portals, links)),
                    Ingress.flatten_iterable_of_tuples(map(Field.get_portals_inclusive, fields)))

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
            delta_latitude = tl.lat - br.lat
            delta_longitude = br.lng - tl.lng
            assert delta_latitude > 0 and delta_longitude > 0, "both deltas AREN'T positive"

            br.lng += delta_longitude * .05
            br.lat -= delta_latitude * .05

            tl.lat += delta_latitude * .05
            tl.lng -= delta_longitude * .05


        self.tl = tl
        self.br = br
        
    def is_in(self, p: Portal):
        return (self.br.lat < p.lat < self.tl.lat) and (self.tl.lng < p.lng < self.br.lng)

class Tree:
    def __init__(self, root: Field, split_method: Callable, announce_self = True) -> None:
        self.root = root
        self.split_method = split_method

        while leaves_with_portals_in_them := tuple(filter(Field.has_portals, filter(Field.is_leaf, self.get_fields()))):
            for leaf in leaves_with_portals_in_them:
                leaf.children = leaf.split(self.split_method)

        if announce_self: print(self)

    def __repr__(self) -> str:
        data = {"mind_units": self.get_MU(),
                "average_level": self.get_mean_level(),
                "standard_deviation": self.get_standard_deviation(),
                "amount_of_links": len(self.get_links()),
                "amount_of_fields": len(self.get_fields()),
                "base_field": self.root}
        
        return "MU: %6d mean lvl: %5.2f stndrd deev: %4.2f links: %3d fields: %3d (%r)" % tuple(data.values())

    def get_fields(self, node: Field = None) -> tuple[Field]:
        """return node and all lower standing nodes, uses root if node not given (made by GTP-3.5).

        Widely used function across the entire project."""
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

    def get_level_variance(self) -> float:
        leaves = tuple(filter(Field.is_leaf, self.get_fields()))
        leaf_areas = np.array(list(map(Field.get_area, leaves)))
        normalized_leaf_areas = leaf_areas / self.root.get_area()

        assert 1-sum(normalized_leaf_areas) < .01, "the sum of normalized leaf areas is off by at least 1%"

        leaf_levels = np.array(list(map(Field.get_level, leaves)))

        return sum(normalized_leaf_areas * (leaf_levels - self.get_mean_level())**2)

    def get_standard_deviation(self) -> float:
        return math.sqrt(self.get_level_variance())

    def get_level_range(self) -> tuple[int, int]:
        """return lowest and highest level of all leaves"""
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
    """contains only static methods used across the entire project"""

    used_portals: tuple[Portal] = None

    color_maps = {"rainbow" : lambda variable: Ingress.hsv_to_rgb(variable, 1, 1),
                "grayscale": lambda variable: Ingress.hsv_to_rgb(1, 0, variable),
                "green": lambda _: (0, 1, 0),
                "blue": lambda _: (0, 0, 1)
                }

    @staticmethod
    def hsv_to_rgb(h,s,v):
        return map(lambda a: int(a*255), colorsys.hsv_to_rgb(h,s,v))

    # TODO: replace with BoundingBox object
    @staticmethod
    def draw_level(number: int, bounding_box: tuple[Portal, Portal], field_level: int = -1) -> tuple[Field]:
        """return field/-s that represent number (one field per digit in number)"""
        # NOTE: consider importing this from a .json file
        DIGIT_MAP = {
                "0":((1,0),(0,1),(1,3),(4,2),(3,1),(0,2),(1,4),(3,4),(4,3),(3,0)),
                "1":((1,0),(1,0),(0,1),(0,3),(3,3),(3,4),(4,4),(4,0),(3,0),(3,1),(1,1)),
                "2":((1,0),(0,1),(0,3),(1,4),(2,4),(3,2),(3,4),(4,4),(4,0),(3,0),(2,2),(1,2),(2,1)),
                "3":((0,0),(0,3),(1,4),(2,3),(3,4),(4,3),(4,1),(2,0),(4,2),(3,3),(2,2),(2,1),(1,0),(2,2),(1,3),(0,3),(1,1)),
                "4":((2,0),(0,2),(0,4),(1,4),(2,3),(0,3),(2,1),(2,4),(3,4),(3,3),(4,3),(4,2),(3,2),(3,0)),
                "5":((0,0),(0,4),(1,4),(1,1),(2,1),(1,2),(2,4),(3,4),(4,3),(4,0),(3,0),(3,3),(2,3),(3,2),(3,1),(2,0)),
                "6":((1,0),(0,1),(0,3),(1,4),(1,3),(0,2),(1,1),(2,4),(3,4),(3,1),(2,1),(3,4),(4,3),(4,1),(3,0)),
                "7":((0,0),(0,4),(1,4),(2,2),(2,3),(3,3),(3,2),(4,2),(4,1),(3,1),(3,0),(2,0),(2,1),(1,3),(1,1)),
                "8":((1,0),(0,1),(1,2),(2,1),(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(4,3),(4,2),(3,1),(2,2),(2,3),(3,3),(4,2),(4,1),(3,0),(2,1)),
                "9":((1,0),(0,1),(0,3),(1,4),(3,4),(4,3),(4,1),(3,0),(2,0),(4,2),(3,3),(1,3),(1,1),(0,2),(1,3),(2,1)),
                    }
        if field_level < 0: field_level = number-1

        digit_bounding_boxes = tuple([bounding_box])
        digits = list(str(number))
        # NOTE: does not support any number of digits, but when the time comes I'll implement split_bounding_box_vertically(splits = 2)
        if len(digits) > 1:
            digit_bounding_boxes = Ingress.split_bounding_box_vertically(digit_bounding_boxes[0])

        output = []
        for digit, bounding_box in zip(digits, digit_bounding_boxes):
            tl, br = bounding_box
            lat_unit = (br.lat - tl.lat)/5
            lng_unit = (br.lng - tl.lng)/5
            portal_grid = {(lat_offset, lng_offset): Portal(str((lat_offset, lng_offset)), tl.lat + lat_offset*lat_unit, tl.lng + lng_offset*lng_unit) for lng_offset in range(5) for lat_offset in range(5)}

            output.append(Field.from_route(
                tuple(map(portal_grid.get, DIGIT_MAP[str(digit)])),
                field_level))
        return tuple(output)

    @staticmethod
    def create_legend(fields: list[Field], onlyleaves: bool) -> list[Field]:
        bb = BoundingBox(fields)
        tr, bl = (Portal("top right", bb.tl.lat, bb.br.lng), Portal("bottom left", bb.br.lat, bb.tl.lng))

        min_field_level = min(map(Field.get_level, fields))
        max_field_level = max(map(Field.get_level, fields))

        # TODO: declare start and end differently based on where to put legend and what order
        # delta_longitude = abs(tl.lng - br.lng)
        delta_latitude = abs(bb.br.lat - bb.tl.lat)
        segment_length = delta_latitude/(max_field_level+1-min_field_level)
        PADDING = delta_latitude*.05
        output = []
        for level in range(min_field_level, max_field_level+1):
            offset = segment_length * (level - min_field_level)
            item_tl = Portal("legend item's top left", bl.lat+offset+segment_length, bl.lng-(segment_length*1.87)-PADDING)
            item_br = Portal("legend item's bottom right", bl.lat+offset, bl.lng-PADDING)
            output.append(Ingress.draw_level(level+1, (item_tl, item_br)))

            if not onlyleaves:
                for lvl in range(level):
                    output.append(Ingress.draw_level(level+1, (item_tl, item_br), field_level = lvl))

        return Ingress.flatten_iterable_of_tuples(output)

    @staticmethod
    def split_bounding_box_vertically(bounding_box: tuple[Portal, Portal]):
        tl, br = bounding_box
        tr, bl = (Portal("top right", tl.lat, br.lng), Portal("bottom left", br.lat, tl.lng))
        right_bb_tl = tl.find_middle(tr)
        left_bb_br = bl.find_middle(br)

        return ((tl, left_bb_br), (right_bb_tl, br))
    @staticmethod
    def flatten_tuples(*tuples: tuple) -> tuple:
        return tuple(element for tuple in tuples for element in tuple)
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
    def read_json(file_path: str, help_method: Callable):
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            help_method(first_time = True)
            return None
        except json.decoder.JSONDecodeError:
            print(f"{file_path} is empty, make sure to copy/paste whatever IITC gives you into input.json (and save it)")
            return None

    @staticmethod
    def output_to_json(o: object, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(o, f, indent=2, ensure_ascii=False)
        print(f"{os.path.basename(file_path)} created successfully")

    @staticmethod
    def copy_to_clipboard(o: object):
        pyperclip.copy(json.dumps(o, ensure_ascii=False))
        print("output copied to clipboard successfully")

    @staticmethod
    def find_portal_from_latLng(latLng: dict) -> Portal:
        for portal in Ingress.used_portals:
            if portal.get_latLng() == latLng:
                return portal

        assert False, f"portal with {latLng} not found in used_portals"

    @staticmethod
    def parse_lat_comma_lng(latLng: str) -> list[float]:
        return [float(coordinate) for coordinate in latLng.split(",")]

    @staticmethod
    def parse_latLng(latLng: dict) -> list[float]:
        return [float(latLng["lat"]), float(latLng["lng"])]

    @staticmethod
    def parse_input(IITC_elements: list[dict]) -> tuple[list[Field], list[list[Portal]], list[dict]]:
        """parse the contents of input.json (which should be full of IITC copy/paste)"""
        white = "#ffffff"

        markers = [e for e in IITC_elements if e["type"] == "marker" and e["color"] == white]
        polylines = [e for e in IITC_elements if e["type"] == "polyline" and e["color"] == white]
        assert len(markers) == len(polylines), f"ERROR: amount of markers and polylines should match, markers: {len(markers)}, polylines: {len(polylines)}"
        # TODO: assert that every portal with a marker is a subset of start and end portals of polylines

        polygons = [e for e in IITC_elements if e["type"] == "polygon" and e["color"] == white]

        base_fields = []
        routes = []

        for polygon in polygons:
            base_fields.append(Field.from_IITC_polygon(polygon))

        # TODO: one last step towards ultimate robustness in terms of "applying routes to fields"
        # TODO: relies on correct creation order
        for marker, polyline in zip(markers, polylines):
            routes.append(Portal.from_IITC_marker_polyline(marker, polyline))

        other = [e for e in IITC_elements if e["color"] != white]

        return (base_fields, routes, other)

    @staticmethod
    def parse_split_method(s: str) -> Callable:
        """function that parses split_methods from strings
        for example: "spiderweb" -> Field.spiderweb and hybrid3 -> Field.hybrid(3)"""
        if match := re.match(r"^hybrid(\d)$", s):
            digit = int(match.group(1))
            return Field.hybrid(digit)
        if re.match(r"^spiderweb$", s):
            return Field.spiderweb
        if re.match(r"^homogen$", s):
            return Field.homogen
        if re.match(r"^center$", s):
            return Field.center
        assert False, "unrecognised split_method, available methods: (spiderweb|hybrid#|homogen)"

    @staticmethod
    def render(objects: tuple[Portal|Link|Field], color_map_function: Callable, output: list = None) -> list[dict]:
        if output is None: output = []
        if len(objects) == 0: return []

        # TODO: or maybe for object in objects and isinstance the type?
        portals = tuple(filter(lambda o: isinstance(o, Portal), objects))
        links = tuple(filter(lambda o: isinstance(o, Link), objects))
        fields = tuple(filter(lambda o: isinstance(o, Field), objects))

        max_level = max(Ingress.flatten_iterable_of_tuples((
                        map(Portal.get_value, portals),
                        map(Link.get_level, links),
                        map(Field.get_level, fields))))

        for portal in portals:
            output.append({
                "type": "marker",
                "latLng": {"lat": portal.lat, "lng": portal.lng},
                "color": "#{:02x}{:02x}{:02x}".format(*color_map_function(portal.get_value()))
                })
        for link in links:
            output.append({
                "type": "polyline",
                "latLngs": [{"lat": portal.lat, "lng": portal.lng} for portal in link.portals],
                "color": "#{:02x}{:02x}{:02x}".format(*color_map_function(link.get_level()/(max_level+1)))
            })
        for field in fields:
            output.append({
                "type": "polygon",
                "latLngs": [{"lat": portal.lat, "lng": portal.lng} for portal in field.portals],
                "color": "#{:02x}{:02x}{:02x}".format(*color_map_function(field.get_level()/(max_level+1)))
            })

        return output

    @staticmethod
    def create_plan(routes: list[tuple[Portal]], trees: list[Tree]) -> dict:
        # TODO: big rework incoming for when portals can be visited multiple times
        AVERAGE_WALKING_SPEED = 84 # meters per minute
        COOLDOWN_BETWEEN_HACKS = 5 # minutes

        all_fields: tuple[Field] = Ingress.flatten_iterable_of_tuples(map(Tree.get_fields, trees))
        all_links: set[Link] = set(Ingress.flatten_iterable_of_tuples(map(Tree.get_links, trees)))
        all_portals = set(Ingress.flatten_iterable_of_tuples(map(Field.get_portals_inclusive, all_fields)))

        route_portals = set(Ingress.flatten_iterable_of_tuples(routes))
        # warn if there are any portals missed by routes
        if (missed_portal_count := len(all_portals) - len(route_portals)) > 0:
            print(f"WARNING: routes missed {missed_portal_count} portal/-s, plan/-s might not be accurate. Missing portals {all_portals.difference(route_portals)}")

        plan_routes = {}
        visited_portals = set()
        created_links = set()
        for route in routes:
            steps = {}
            SBUL_count = 0
            route_links = set()
            for active_portal in route:
                visited_portals.add(active_portal)

                connected_links = set(filter(active_portal.is_part_of_link, all_links))
                outbound_links = [link for link in connected_links.difference(created_links) if visited_portals.issuperset(link.portals)]
                # outbound_links = list(filter(visited_portals.issuperset, links.difference(created_links)))
                route_links.update(outbound_links)
                created_links.update(outbound_links)

                SBUL_count += 0 if len(outbound_links) == 0 else (len(outbound_links) - 1) // 8
                
                # TODO: we're double nested over here, I'm sure there's a way to overcome this
                # primary sort criteria: filter all the fields that the link is part of and get their levels and pick the min level
                # secondary sort criteria: longest link first
                link_order = sorted(outbound_links, key = lambda outbound_link: (min(map(Field.get_level, filter(outbound_link.is_part_of_field, all_fields))), -outbound_link.get_length()))
                # link_order = sorted(outbound_links, key = lambda outbound_link: [field for field in all_fields if outbound_link in field.get_links()])
                # link_order = sorted(outbound_links, key = lambda l: (min(map(Field.get_level, filter(lambda f: l in f.get_links(), all_fields))), -tuple(l)[0].distance(tuple(l)[1])))

                steps[active_portal.get_label()] = {
                "keys": len(connected_links)-len(outbound_links),
                "links": list(map(Portal.get_label, map(active_portal.get_adjacent_portal, link_order)))
                }

            key_count = len(route_links)
            route_length = round(sum(starmap(Portal.distance, pairwise(route))), 2)
            time_to_complete = route_length / AVERAGE_WALKING_SPEED + key_count * COOLDOWN_BETWEEN_HACKS

            plan_routes[f"{route[0]}...{route[-1]}"] = {
                "keys_required": key_count,
                "SBULs_required": SBUL_count,
                "route_length_(meters)": route_length,
                "estimated_time_to_complete_(minutes)": round(time_to_complete, 2),
                "estimated_time_to_complete_(hours)": round(time_to_complete/60, 2),
                "steps": steps
                }

        plan = {
            "portals_involved": len(visited_portals),
            "total_keys_required": sum(map(lambda k: plan_routes[k]["keys_required"], plan_routes)),
            "total_SBULs_required": sum(map(lambda k: plan_routes[k]["SBULs_required"], plan_routes)),
            "routes": plan_routes
            }
        return plan

    @staticmethod
    def simulate_plan(plan: dict, chunk_together: bool = False) -> dict:
        """return route: {bounding_box: BoundingBox, steps: [[<Portal|Link|Field>]]}"""
        output = {}
        context = []
        routes: dict = plan["routes"]
        for route_title in routes:
            plan_steps: dict = routes[route_title]["steps"]
            route_steps = []
            for active_portal in map(Ingress.find_portal, plan_steps):
                active_portal.value = len(plan_steps[active_portal.get_label()]["links"])
                portal_steps = [(active_portal, )]
                portals_to_link_to = tuple(map(Ingress.find_portal, plan_steps[active_portal.get_label()]["links"]))
                links = map(active_portal.create_link, portals_to_link_to)

                for link in links:
                    portal_steps.append((link, *link.get_resulting_fields(context))) 
                    context.append(link)

                if chunk_together: portal_steps = (Ingress.flatten_iterable_of_tuples(portal_steps), )
                route_steps.extend(portal_steps)

            bb = BoundingBox(Ingress.flatten_iterable_of_tuples(route_steps), grow_to_square=True, padding=True)
            output[route_title] = {"bounding_box": bb, "steps": route_steps}

        return output
    
    @staticmethod
    def validate_simulation(simulation: dict):
        # {bounding_box: BoundingBox, steps: [[<Portal|Link|Field>]]}
        all_steps = map(lambda r: simulation[r]["steps"], simulation)
        simulation_objects = Ingress.flatten_iterable_of_tuples(Ingress.flatten_iterable_of_tuples(all_steps))

        claimed_portals: set[Portal] = set()
        links = set()
        fields = set()
        for o in simulation_objects:
            if isinstance(o, Portal):
                claimed_portals.add(o)
            if isinstance(o, Link):
                links.add(o)
                if not set(o.get_portals()).issubset(claimed_portals):
                    print(f"WARNING: link created with unclaimed portal: {o}")
                if not o.get_length() < 2000 and any(map(o.get_portals()[0].is_under_field, fields)):
                    print(f"WARNING: link cannot be created: {o}")
            if isinstance(o, Field):
                fields.add(o)
        
        for p in claimed_portals:
            if not any(map(p.is_under_field, fields)):
                print(f"WARNING: {p} is unused even though it is part of plan")
            outbound_links: tuple[Link] = tuple(filter(lambda l: p.__eq__(l.get_frm()), links))
            if outbound_links:
                longest_link = max(outbound_links, key=lambda l: l.get_length())
                required_level = math.sqrt(math.sqrt(longest_link.get_length()/160))
                if required_level > 4:
                    print(f"NOTE: {p} needs to be at least lvl {required_level:.0f}")
            if len(outbound_links) > 24:
                print(f"WARNING: at portal {p} outbound link count exceeds 2 SBULs: {len(outbound_links)}")

    @staticmethod
    def get_from_bkmrk_file(bkmrk_file_path: str) -> tuple[Portal]:
        with open(bkmrk_file_path, "r", encoding='utf-8') as f:
            bkmrk = json.load(f)['portals']['idOthers']['bkmrk']

        return tuple(Portal(bkmrk[id]["label"], *Ingress.parse_lat_comma_lng(bkmrk[id]["latlng"])) for id in bkmrk)

    @staticmethod
    def load_portals(directory) -> None:
        portal_groups = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    portal_groups.append(Ingress.get_from_bkmrk_file(os.path.join(root, file)))

        Ingress.used_portals = Ingress.flatten_iterable_of_tuples(portal_groups)