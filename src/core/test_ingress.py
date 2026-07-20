import unittest
from ingress import Portal

class TestHTMLNode(unittest.TestCase):
    def test_create_portal_in_london(self):
        p = Portal("London", 51.522198892509834, -0.08104948361448791)
        self.assertIsInstance(p, Portal)
        
    def test_create_portal_in_warshaw(self):
        p = Portal("Warshaw", 52.22784036114757, 21.02844821268945)
        self.assertIsInstance(p, Portal)

    def test_create_portal_in_melbourne(self):
        p = Portal("Melbourne", -37.802271637327316, 144.9520689991028)
        self.assertIsInstance(p, Portal)

    def test_create_portal_in_argentina(self):
        p = Portal("Argentina", -34.85562207716252, -65.10514435848482)
        self.assertIsInstance(p, Portal)

    def test_create_portal_outside_of_map(self):
        self.assertRaises(ValueError, lambda: Portal("Nowhere land", -90.85562207716252, -180.10514435848482))

if __name__ == "__main__":
    unittest.main()