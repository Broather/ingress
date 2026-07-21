import unittest
from ingress import Portal, Link, Field

class TestIngress(unittest.TestCase):
    bridge = Portal("Iron Bridge", 52.62751740320499, -2.4850676907714218)
    cathedral = Portal("Worcester Cathedral", 52.18891077941245, -2.2203404465959107)
    castle = Portal("Warwick Castle", 52.27838196580572, -1.5880068312555011)

    def test_create_portal(self):
        self.assertIsInstance(self.bridge, Portal)

    def test_create_portal_outside_of_map(self):
        self.assertRaises(ValueError, lambda: Portal("Nowhere land", -90.85562207716252, -180.10514435848482))

    def test_create_link(self):
        l = Link(self.bridge, self.cathedral)
        self.assertIsInstance(l, Link)

    def test_create_field(self):
        f = Field(self.bridge, self.cathedral, self.castle)
        self.assertIsInstance(f, Field)
        
if __name__ == "__main__":
    unittest.main()