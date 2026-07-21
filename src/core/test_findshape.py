import unittest
from ingress import Portal

def match(self, shape, portals):
    # helper function to find shape in portals
    return True

def no_match(self, shape, portals):
    # helper function to find shape in portals
    return True

class TestFindshape(unittest.TestCase):
    def test_no_match(self):
        # try to find shape from no points
        pass
    def test_translated_match(self):
        # define reasonable offset
        # put the shape’s points on the map
        pass
    def test_translated_match_in_portals(self):
        assert no_match(self, 69, 420)
        # define reasonable offset with range(5) variation that overlaps portals
        # define reasonable scale
        # put the shape’s points on the map overlapping other portals
        assert match(self, 69, 420)
    def test_rotated_match(self):
        # define reasonable offset
        # define rotation with range(1, 360, 10) variation
        # put the shape’s points on the map
        pass
    def test_rotated_match_in_portals(self):
        assert no_match(self, 69, 420)
        # define reasonable offset that overlaps portals
        # define reasonable scale
        # define rotation with range(1, 360, 10) variation
        # put the shape’s points on the map overlaping other portals
        assert match(self, 69, 420)
        pass
    def test_scaled_match_in_portals(self):
        assert no_match(self, 69, 420)
        # define offset that overlaps with portals
        # define scale with reasonable * range(5) variation
        # put the shape’s points on the map overlaping other portals
        assert match(self, 69, 420)
        pass
    def test_translated_rotated_match_in_portals(self):
        pass
    def test_rotated_scaled_match_in_portals(self):
        pass
    def test_translated_scaled_match_in_portals(self):
        pass
    def test_noise(self):
        # each point in shape is slightly moved before placed on map
        pass
    def test_noise_in_portals(self):
        # each point in shape is slightly moved before placed on map overlaping other portals
        pass
    def test_noise_translated_rotated_match_in_portals(self):
        pass
    def test_noise_rotated_scaled_match_in_portals(self):
        pass
    def test_noise_translated_scaled_match_in_portals(self):
        pass
    def test_mirrored_shape(self):
        # do a negative scale
        pass
    def test_mirrored_shape_in_portals(self):
        # do a negative scale
        pass
    def test_valid_shape_map(self):
        # """[{"grid": 4,"0": [{"x": 0,"y": 1},{"x": 0,"y": 2},{"x": 3,"y": 3}],"1": [{"x": 0,"y": 1},{"x": 1,"y": 3},{"x": 3,"y": 2}]},{"grid": 6,"0": [{"x": 0,"y": 1},{"x": 0,"y": 2},{"x": 5,"y": 3},{"x": 0,"y": 2},{"x": 4,"y": 1}],"1": [{"x": 0,"y": 1},{"x": 1,"y": 3},{"x": 3,"y": 5},{"x": 3,"y": 2},{"x": 3,"y": 2}]}]"""
        pass
    def test_invalid_shape_map(self):
        # format for valid shape map [{grid: n > 0, 0-9a-z: [{x: a < n, y: b < n}]}, {grid: n > 0}]
        pass
    def test_match_result(self):
        # For each detected shape, the script shall output:
        # confidence or matching score,
        # portals involved,
        # geometric transformation applied (rotation, scale, translation).
        pass
    def test_result_json(self):
        # encode to json string and assertEquals
        pass

if __name__ == "__main__":
    unittest.main()