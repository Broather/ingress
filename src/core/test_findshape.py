import json
import unittest
import testdata
from ingress import Portal, bounding_box, Shape

def find(shape: Shape, portals: list[Portal], fixed_rotation = False):
    pass

def match(shape: Shape, portals: list[Portal]) -> bool:
    # helper function to find shape in portals
    if portals is None or len(portals) == 0:
        return False
    
    # print(f'finding shape /w {len(shape.points)} points in {len(portals)} portal(-s)')
    return bool(find(shape, portals))

def get_shape(shapemap_string: str, grid_size: int, shape_name: str) -> Shape:
    # helper function to get a shape from shapemap JSON string
    grids: list[dict] = json.loads(shapemap_string)
    assert isinstance(grids, list), "shape map expected to contain list as root element"
    assert len(grids) > 0, "no grids? *megamind meme*"

    filtered_grids = list(filter(lambda grid: grid.get("grid") == grid_size, grids))
    assert len(filtered_grids) != 0, f"can't find grid of size {grid_size}"

    grid = filtered_grids[0]
    assert shape_name in grid, f"shape {shape_name} doesn't exist in grid of size {grid_size}"

    return Shape(grid_size, grid.get(shape_name, []))

def new_map():
    points_on_map = []
    
    def inner(shape: Shape|list[Portal], origin: Portal|None = None, scale: float = 1, rotation: int = 0):
        if isinstance(shape, list) and all(map(lambda o: isinstance(o, Portal), shape)):
            # TODO: could use origin to translate all portals to a certain place
            points_on_map.extend(shape)
            return points_on_map
        elif isinstance(shape, Shape):
            assert len(points_on_map) >= 2, "overlay requires at least 2 points on the map"
            tl, br = bounding_box(points_on_map)
            cell_size = min(abs(tl.lat - br.lat), abs(tl.lng - br.lng)) * scale/shape.grid_size

            if not origin: origin = tl
            mapped_points = map(lambda coords: origin + Portal("", -1 * coords[0] * cell_size, coords[1] * cell_size), shape.points)
            points_on_map.extend(mapped_points)
            return points_on_map
        else:
            assert False, f"unsupported type to overlay: {type(shape[0])}"
        
    return inner

class TestFindShape(unittest.TestCase):
    town_portals = testdata.matlock
    shape = testdata.square

    def test_create_shape(self):
        square = Shape(4, [[1,1], [2,1], [2,2], [1,2]])
        self.assertEqual(square.__repr__(), "....\n.##.\n.##.\n....\n")

    def test_get_shape(self):
        shapemap = """[{
        "grid": 4,
        "square": [[1,1], [2,1], [2,2], [1,2]]
    }]"""
        square = get_shape(shapemap, 4, "square")
        self.assertEqual(square.__repr__(), "....\n.##.\n.##.\n....\n")

    def test_no_portals(self):
        # fail to find shape from no points
        test_map = new_map()
        self.assertFalse(match(self.shape, test_map([])))

    @unittest.skip("not implemented")
    def test_match(self):
        # fail to find shape from points on map until the shape's points get overlaid
        test_map = new_map()
        self.assertFalse(match(self.shape, test_map(self.town_portals)))
        self.assertTrue(match(self.shape, test_map(self.shape)))
        
    @unittest.skip("not implemented")
    def test_translated_match(self):
        # define reasonable offset
        # put the shape’s points on the map
        test_map = new_map()
        portals = test_map(self.town_portals)
        self.assertFalse(match(self.shape, portals))

        tl, br = bounding_box(portals)
        middle = tl.find_middle(br)

        self.assertTrue(match(self.shape, test_map(self.shape, origin=middle, scale=.5)))

    def test_translated_match_in_town(self):
        # define reasonable offset with range(5) variation that overlaps portals
        # define reasonable scale
        # put the shape’s points on the map overlapping other portals
        pass
    def test_rotated_match(self):
        # define reasonable offset
        # define rotation with range(1, 360, 10) variation
        # put the shape’s points on the map
        pass
    def test_rotated_match_in_town(self):
        # define reasonable offset that overlaps portals
        # define reasonable scale
        # define rotation with range(1, 360, 10) variation
        # put the shape’s points on the map overlaping other portals
        pass
    def test_scaled_match_in_town(self):
        # define offset that overlaps with portals
        # define scale with reasonable * range(5) variation
        # put the shape’s points on the map overlaping other portals
        pass
    def test_translated_rotated_match_in_town(self):
        pass
    def test_rotated_scaled_match_in_town(self):
        pass
    def test_translated_scaled_match_in_town(self):
        pass
    def test_noise(self):
        # each point in shape is slightly moved before placed on map
        pass
    def test_noise_in_town(self):
        # each point in shape is slightly moved before placed on map overlaping other portals
        pass
    def test_noise_translated_rotated_match_in_town(self):
        pass
    def test_noise_rotated_scaled_match_in_town(self):
        pass
    def test_noise_translated_scaled_match_in_town(self):
        pass
    def test_mirrored_shape(self):
        # do a negative scale
        pass
    def test_mirrored_shape_in_town(self):
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