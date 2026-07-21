from collections.abc import Sequence
import unittest
from ingress import Portal, Link, Field, Ingress
from itertools import combinations

type Entity = Portal | Link | Field

class CrossingLinksError(Exception):
    pass
class LinkUnderFieldError(Exception):
    pass

def is_overlap(a: Entity, b: Entity):
    # normalize ordering so type(a).__name__ <= type(b).__name__ (Field < Link < Portal)
    if type(a).__name__ > type(b).__name__:
        a, b = b, a

    if isinstance(a, Portal) and isinstance(b, Portal):
        return a == b
    elif isinstance(a, Link) and isinstance(b, Link):
        return a.frm in b.portals or a.to in b.portals
    elif isinstance(a, Field) and isinstance(b, Field):
        return any(
            is_overlap(link1, link2)
            for link1 in a.get_links()
            for link2 in b.get_links()
        )
    elif isinstance(a, Link) and isinstance(b, Portal):
        return b in a.portals
    elif isinstance(a, Field) and isinstance(b, Portal):
        if b in a.portals:
            return False
        return is_portal_in_field(b, a)
    elif isinstance(a, Field) and isinstance(b, Link):
        return is_overlap(a, b.frm) and is_overlap(a, b.to)
    return True

def do_links_cross(link1: Link, link2: Link):
    # return true if given points are in counterclockwise order otherwise false
    ccw = lambda a,b,c: (c.lng-a.lng)*(b.lat-a.lat) > (b.lng-a.lng)*(c.lat-a.lat)

    a,b = link1.portals
    c,d = link2.portals
    return ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)

def are_portals_on_same_side_of_link(link: Link, a: Portal, b: Portal) -> bool:
    cross = lambda a, b, c: (b.lat - a.lat) * (c.lng - a.lng) - (b.lng - a.lng) * (c.lat - a.lat)
    return cross(link.frm, link.to, a) * cross(link.frm, link.to, b) > 0 # non-inclusive

def is_portal_in_field(portal: Portal, field: Field) -> bool:
    a, b, c = field.portals
    return all([are_portals_on_same_side_of_link(Link(a, b), c, portal),
                are_portals_on_same_side_of_link(Link(b, c), a, portal),
                are_portals_on_same_side_of_link(Link(c, a), b, portal)])
    
def get_links(context) -> list[Link]:
    # replace fields with 3 links, pass along existing links and filter out portals
    links = [e for e in context if isinstance(e, Link)]
    fields = [e for e in context if isinstance(e, Field)]
    
    unique_field_links = list(set(Ingress.flatten_iterables(map(Field.get_links, fields))))
    return list(unique_field_links) + list(links)

def add_link(context: Sequence[Entity], new_link: Link) -> Sequence[Entity]:
    fields = [e for e in context if isinstance(e, Field)]
    links = [e for e in context if isinstance(e, Link)]

    if any(map(lambda link: do_links_cross(link, new_link), links)):
        raise CrossingLinksError
    if any(map(lambda field: is_overlap(field, new_link), fields)):
        if new_link.get_length() > 2000:
            raise LinkUnderFieldError
    
    resulting_fields = get_resulting_fields(context, new_link)

    return [*context, new_link, *resulting_fields]

def get_resulting_fields(context: Sequence[Entity], new_link: Link) -> list[Field]:
    touching_links = list(filter(lambda l: is_overlap(l, new_link), get_links(context)))
    if len(touching_links) <= 1: return []

    possible_third_portals: list[Portal] = []
    for one_link, other_link in combinations(touching_links, 2):
        possible_third_portals.extend([third_p for third_p in filter(lambda p: p not in new_link.portals, [*one_link.portals, *other_link.portals]) if is_loop(new_link, one_link, other_link)])

    if len(possible_third_portals) <= 1: return [Field(new_link.frm, new_link.to, possible_third_portals[0])] if possible_third_portals else []

    third_portal_that_makes_the_biggest_possible_field = possible_third_portals.pop(possible_third_portals.index(max(possible_third_portals, key=lambda p: Field(new_link.frm, new_link.to, p).get_area())))
    third_portal_that_makes_the_second_biggest_possible_field_on_other_side_of_link = max(filter(lambda p: not are_portals_on_same_side_of_link(new_link, third_portal_that_makes_the_biggest_possible_field, p), possible_third_portals), key=lambda p: Field(new_link.frm, new_link.to, p).get_area(), default=None)
    
    if third_portal_that_makes_the_second_biggest_possible_field_on_other_side_of_link:
        return [Field(new_link.frm, new_link.to, third_portal_that_makes_the_biggest_possible_field),
                Field(new_link.frm, new_link.to, third_portal_that_makes_the_second_biggest_possible_field_on_other_side_of_link)]
    else:
        return [Field(new_link.frm, new_link.to, third_portal_that_makes_the_biggest_possible_field)]
    
def is_loop(link1, link2, link3):
    return is_overlap(link1, link2) and is_overlap(link2, link3) and is_overlap(link3, link1)  

class TestIngress(unittest.TestCase):
    # lotations from tom scotts series where he makes a video about each province in the uk
    obsevatory = Portal("Jodrel Bank Observatory", 53.24227460218302, -2.3063232907714215)
    castle = Portal("Warwick Castle", 52.27838196580572, -1.5880068312555011)
    iron_bridge = Portal("Iron Bridge", 52.62751740320499, -2.4850676907714218)
    school = Portal("National Ferret School", 53.14537960835833, -1.4948476824569794)
    village = Portal("Abbots Bromley Village", 52.8181452634191, -1.8808374609719551)
    bay = Portal("Morecambe Bay", 54.074586670671, -2.8643857107313155)
    classroom = Portal("Network Rails Manchester Classroom", 53.47707777167849, -2.224206360929744)
    cathedral = Portal("Worcester Cathedral", 52.18891077941245, -2.2203404465959107)

    london_eye = Portal("London Eye", 51.50336422663004, -0.11940699999999997)
    big_ben = Portal("Big Ben", 51.50075223765619, -0.12453627116489778)
    westminster_abbey = Portal("Westminster Abbey", 51.49973326491089, -0.12743414606401254)
    tower_of_london = Portal("Tower of London", 51.50904332613851, -0.0761843093544774)
    british_museum = Portal("The British Museum", 51.51956682988635, -0.12630807751575648)
    

    def test_create_portal(self):
        self.assertIsInstance(self.iron_bridge, Portal)

    def test_create_portal_outside_of_map(self):
        self.assertRaises(ValueError, lambda: Portal("Nowhere land", -34.85562207716252, -180.10514435848482))
        self.assertRaises(ValueError, lambda: Portal("Nowhere land", -90.85562207716252, -79.10514435848482))
        self.assertRaises(ValueError, lambda: Portal("Nowhere land", -90.85562207716252, -180.10514435848482))

    def test_create_link(self):
        l = Link(self.iron_bridge, self.cathedral)
        self.assertIsInstance(l, Link)

    def test_create_field(self):
        f = Field(self.iron_bridge, self.cathedral, self.castle)
        self.assertIsInstance(f, Field)

    def test_create_field_w_context(self):
        link1 = Link(self.iron_bridge, self.cathedral)
        link2 = Link(self.cathedral, self.castle)
        context = [link1, link2]
        
        new_l = Link(self.castle, self.iron_bridge)
        try:
            context = add_link(context, new_l)
        except (CrossingLinksError, LinkUnderFieldError):
            pass
        
        expected_f = Field(self.iron_bridge, self.cathedral, self.castle)
        self.assertIn(expected_f, context)
    
# ----- linking and fielding rules -----

    def test_crossing_links(self):
        # a link cannot cross another link
        l1 = Link(self.iron_bridge, self.castle)
        l2 = Link(self.cathedral, self.village)

        context = [l1]
        self.assertRaises(CrossingLinksError, lambda: add_link(context, l2))

    def test_anchor_portals_outside(self):
        # the three anchor portals that make up a field are considered outside of the control field
        f = Field(self.iron_bridge, self.castle, self.cathedral)
        self.assertFalse(is_overlap(f, self.iron_bridge))
        self.assertFalse(is_overlap(f, self.castle))
        self.assertFalse(is_overlap(f, self.cathedral))

    def test_link_from_anchor_portals_to_inside(self):
        # portals inside the field can be linked to from the anchor portals
        f = Field(self.iron_bridge, self.castle, self.school)
        inside_p = self.village
        new_l = Link(self.iron_bridge, inside_p)

        context = [f]
        try:
            context = add_link(context, new_l)
        except (CrossingLinksError, LinkUnderFieldError):
            pass

        self.assertIn(new_l, context)

    def test_link_far_portals_inside_field(self):
        # portals inside the field can be linked together if the distance between portals does not exceed 2 000 m
        f = Field(self.iron_bridge, self.school, self.bay)
        inside_p1 = self.obsevatory
        inside_p2 = self.classroom
        new_l = Link(inside_p1, inside_p2)

        self.assertTrue(is_overlap(f, new_l))

        context = [f]
        self.assertRaises(LinkUnderFieldError, lambda: add_link(context, new_l))

    def test_link_close_portals_inside_field(self):
        # portals inside the field can be linked together if the distance between portals does not exceed 2 000 m
        f = Field(self.westminster_abbey, self.tower_of_london, self.british_museum)
        inside_p1 = self.london_eye
        inside_p2 = self.big_ben
        new_l = Link(inside_p1, inside_p2)

        context = [f]
        try:
            context = add_link(context, new_l)
        except (CrossingLinksError, LinkUnderFieldError):
            pass
        self.assertIn(new_l, context)

    def test_split_rectangle(self):
        # when splitting a rectangle with a diagonal link, at most 2 of the largest fields are created
        
        # observatory school
        # v            v
        # .____top_l___.
        # |\           |
        # | \  village |
        # |  \     v   |
        # |   \    .   |
        # |    \       |
        # |     \ new_l| right_l
        # |      \     |
        # |       \    |
        # |left_l  \   |
        # |         \  |
        # |          \ |
        # .__bottom_l_\.
        # ^           ^
        # bridge    castle

        # inside_l1 and inside_l2 are not shown because of space constraints
        top_l = Link(self.obsevatory, self.school)
        bottom_l = Link(self.iron_bridge, self.castle)
        left_l = Link(self.obsevatory, self.iron_bridge)
        right_l = Link(self.school, self.castle)

        inside_p = self.village
        inside_l1 = Link(self.obsevatory, inside_p)
        inside_l2 = Link(inside_p, self.castle)

        context = [top_l, bottom_l, left_l, right_l, inside_l1, inside_l2]
        new_l = Link(self.obsevatory, self.castle)
        try:
            context = add_link(context, new_l)
        except (CrossingLinksError, LinkUnderFieldError):
            pass

        expected_f1 = Field(self.obsevatory, self.iron_bridge, self.castle)
        expected_f2 = Field(self.obsevatory, self.school, self.castle)
        unexpected_f = Field(self.obsevatory, self.village, self.castle)
        
        self.assertIn(expected_f1, context)
        self.assertIn(expected_f2, context)
        self.assertNotIn(unexpected_f, context)
        
if __name__ == "__main__":
    unittest.main()