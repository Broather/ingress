import json
import unittest
from ingress import Portal, bounding_box

class Shape():
    def __init__(self, grid_size: int, points: list[list[int]]) -> None:
        self.grid_size = grid_size
        self.points = points

def find(shape: Shape, portals: list[Portal]) -> tuple:
    return tuple()

def match(shape: Shape, portals: list[Portal]) -> bool:
    # helper function to find shape in portals
    if portals is None or len(portals) == 0:
        return False
    
    print(f'finding shape /w {len(shape.points)} points in {len(portals)} portal(-s)')
    return bool(find(shape, portals))

def shape_map(shapemap_string: str, grid_size: int, shape_name: str) -> Shape:
    # helper function to get a shape from shapemap JSON string
    grids: list[dict] = json.loads(shapemap_string)
    assert isinstance(grids, list), "shape map expected to contain list as root element"
    assert len(grids) > 0, "no grids? *megamind meme picture*"

    filtered_grids = list(filter(lambda grid: grid.get("grid") == grid_size, grids))
    assert len(filtered_grids) != 0, f"can't find grid of size {grid_size}"

    grid = filtered_grids[0]
    assert shape_name in grid, f"shape {shape_name} doesn't exist in grid of size {grid_size}"

    return Shape(grid_size, grid.get(shape_name, []))

def new_map():
    points_on_map = []
    
    def inner(shape: Shape|list[Portal], scale: float = 1, rotation: int = 0):
        if isinstance(shape, list):
            points_on_map.extend(shape)
            return points_on_map
        elif isinstance(shape, Shape):
            assert len(points_on_map) >= 2, "overlay requires at least 2 points on the map"
            tl, br = bounding_box(points_on_map)
            shape_size = min(abs(tl.lat - br.lat), abs(tl.lng - br.lng)) * scale
            mapped_points = map(lambda coords: tl + Portal("", -1 * coords[0] * shape_size/shape.grid_size, coords[1] * shape_size/shape.grid_size), shape.points)
            points_on_map.extend(mapped_points)
            return points_on_map
        else:
            assert False, f"unsupported type to overlay: {type(shape[0])}"
        
    return inner

class TestFindShape(unittest.TestCase):
    village_portals = [
        Portal("", 69, 169),
        Portal("", 42, 142)
    ]
    with open("src/core/shapemap.json", "r") as f:
        shape = shape_map(f.read(), 5, "1")

    def test_no_portals(self):
        # try to find shape from no points
        self.assertFalse(match(self.shape, []))

    def test_no_match_in_points(self):
        # fail to find shape from points on map until the shape's points get overlaid
        overlay = new_map()
        self.assertFalse(match(self.shape, overlay(self.village_portals)))
        self.assertTrue(match(self.shape, overlay(self.shape)))
        
    def test_translated_match(self):
        # define reasonable offset
        # put the shape’s points on the map
        pass
    def test_translated_match_in_portals(self):
        # define reasonable offset with range(5) variation that overlaps portals
        # define reasonable scale
        # put the shape’s points on the map overlapping other portals
        pass
    def test_rotated_match(self):
        # define reasonable offset
        # define rotation with range(1, 360, 10) variation
        # put the shape’s points on the map
        pass
    def test_rotated_match_in_portals(self):
        # define reasonable offset that overlaps portals
        # define reasonable scale
        # define rotation with range(1, 360, 10) variation
        # put the shape’s points on the map overlaping other portals
        pass
    def test_scaled_match_in_portals(self):
        # define offset that overlaps with portals
        # define scale with reasonable * range(5) variation
        # put the shape’s points on the map overlaping other portals
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