import json
import unittest
from ingress import Portal, Field

Shape = tuple[str, list[list[int]]]

def match(shape, portals) -> bool:
    # helper function to find shape in portals
    if portals is None or len(portals) == 0:
        return False
    shape_name, points = shape
    print(f'finding shape "{shape_name}" in {len(portals)} portal(-s)')
    return True

def shape_map(grid_size: int, shape_name: str) -> Shape:
    # helper function to get a shape from shapemap.json
    grids: list[dict] = []
    with open("src/core/shapemap.json", "r") as f:
        grids = json.load(f)

    if len(grids) == 0:
        raise Exception("no grids? *megamind meme picture*")

    grids = list(filter(lambda grid: grid.get("grid") == grid_size, grids))

    assert len(grids) != 0, f"can't find grid of size {grid_size}"
    grid: dict[str, list[list[int]]] = grids[0]
    
    assert shape_name in grid, f"shape {shape_name} doesn't exist in grid of size {grid_size}"
    return (shape_name, grid.get(shape_name, []))
    
class TestFindShape(unittest.TestCase):
    portals = [
        Portal("", 69, 169)
    ]
    shape = shape_map(5, "1")

    def test_no_match(self):
        # try to find shape from no points
        self.assertFalse(match(self.shape, []))
        
    def test_translated_match(self):
        # define reasonable offset
        # put the shape’s points on the map
        pass
    def test_translated_match_in_portals(self):
        self.assertFalse(match(self.shape, []))
        # define reasonable offset with range(5) variation that overlaps portals
        # define reasonable scale
        # put the shape’s points on the map overlapping other portals
        self.assertTrue(match(self.shape, [420]))
    def test_rotated_match(self):
        # define reasonable offset
        # define rotation with range(1, 360, 10) variation
        # put the shape’s points on the map
        pass
    def test_rotated_match_in_portals(self):
        self.assertFalse(match(self.shape, []))
        # define reasonable offset that overlaps portals
        # define reasonable scale
        # define rotation with range(1, 360, 10) variation
        # put the shape’s points on the map overlaping other portals
        self.assertTrue(match(self.shape, [420]))
        pass
    def test_scaled_match_in_portals(self):
        self.assertFalse(match(self.shape, []))
        # define offset that overlaps with portals
        # define scale with reasonable * range(5) variation
        # put the shape’s points on the map overlaping other portals
        self.assertTrue(match(self.shape, [420]))
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