import unittest
from ingress import Portal, Link, Field

def overlap(link, new_link):
    # todo: implement
    return True

def get_links(context):
    # todo: implement
    return context

def add_link(context, new_link):
    if any(map(lambda link: overlap(link, new_link), get_links(context))):
        raise ValueError
    return [*context, new_link]

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

# ----- linking and fielding rules -----

    def test_crossing_links(self):
        # a link cannot cross another link
        l1 = Link(self.iron_bridge, self.castle)
        l2 = Link(self.cathedral, self.village)
        self.assertRaises(ValueError, lambda: add_link([l1], l2))

    def test_anchor_portals_outside(self):
        # the three anchor portals that make up a field are considered outside of the control field
        f = Field(self.iron_bridge, self.castle, self.cathedral)
        # todo: self.assertfalse(overlap(f, self.bridge))
        self.assertFalse(f.is_in(self.iron_bridge))
        self.assertFalse(f.is_in(self.castle))
        self.assertFalse(f.is_in(self.cathedral))

    def test_link_from_anchor_portals_to_inside(self):
        # portals inside the field can be linked to from the anchor portals
        f = Field(self.iron_bridge, self.castle, self.school)
        inside_p = self.village
        new_l = Link(self.iron_bridge, inside_p)

        context = [f]
        context = add_link(context, new_l)
        
        self.assertIn(new_l, context)

    def test_link_far_portals_inside_field(self):
        # portals inside the field can be linked together if the distance between portals does not exceed 2 000 m
        f = Field(self.iron_bridge, self.village, self.bay)
        inside_p1 = self.obsevatory
        inside_p2 = self.classroom
        new_l = Link(inside_p1, inside_p2)

        context = [f]
        try:
            context = add_link(context, new_l)
        except:
            pass

        self.assertNotIn(new_l, context)

    def test_link_close_portals_inside_field(self):
        # portals inside the field can be linked together if the distance between portals does not exceed 2 000 m
        f = Field(self.westminster_abbey, self.tower_of_london, self.british_museum)
        inside_p1 = self.london_eye
        inside_p2 = self.big_ben
        new_l = Link(inside_p1, inside_p2)

        context = [f]
        try:
            context = add_link(context, new_l)
        except:
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
        new_l = Link(self.obsevatory, self.castle)
        context = [top_l, bottom_l, left_l, right_l, inside_l1, inside_l2]
        context = add_link(context, new_l)

        expected_f1 = Field(self.obsevatory, self.iron_bridge, self.castle)
        expected_f2 = Field(self.obsevatory, self.school, self.castle)
        unexpected_f = Field(self.obsevatory, self.village, self.castle)
        
        self.assertIn(expected_f1, context)
        self.assertIn(expected_f2, context)
        self.assertNotIn(unexpected_f, context)
        
if __name__ == "__main__":
    unittest.main()