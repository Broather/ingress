import json
import sys
import itertools

path = r'./portals/pavilosta.json'

class Portal:
    portal_map = {}
    def __init__(self, coordinates: dict):
        # coordinatesExample = {
        #     'lat': 69.69696969,
        #     'lng': 420.4204204
        # }
        self.coordinates = coordinates
        self.label = Portal.portal_map.get(f"{coordinates['lat']},{coordinates['lng']}", "UNKNOWN").replace(" ", "_")
    def __repr__(self) -> str:
        return f"{self.label}"

class Link:
    def __init__(self, p1: Portal, p2: Portal) -> None:
        self.p1 = p1
        self.p2 = p2
    def __repr__(self) -> str:
        return f"{self.p1}-{self.p2}"

class Field:
    def __init__(self, portals: list[Portal], color: str):
        self.portals = portals
        self.color = color

        combinations = list(itertools.combinations(self.portals, 2))
        self.links = list(itertools.starmap(Link, combinations))

    def __repr__(self) -> str:
        return f"{self.portals}"
        
    @staticmethod
    def fromPolygon(polygon: dict):
        # polygonExample = {
        #     'type': 'polygon',
        #     'latLngs': [{'lat': 56.889879, 'lng': 21.179382}, {'lat': 56.889475, 'lng': 21.181191}, {'lat': 56.890217, 'lng': 21.180515}], 
        #     'color': '#bbbbbb'
        #     }
        assert len(polygon['latLngs']) == 3, f"polygon must have 3 sides, has {len(polygon['latLngs'])}"
        
        portals = []
        for coordinates in polygon['latLngs']:
            portals.append(Portal(coordinates))
            
        return Field(portals, "#bbbbbb")
        
if __name__ == "__main__":
    
# based on option, append portals to hash table

    with open(path, 'r', encoding='utf-8') as p1:
        input: list[dict] = json.load(p1)
    
    bkmrk = input['portals']['idOthers']['bkmrk']
    for key in bkmrk:
        portal = bkmrk[key]
        Portal.portal_map[portal['latlng']] = portal['label']

# import plan with route part

    with open('./p_input.json', 'r', encoding='utf-8') as p1:
        input: list[dict] = json.load(p1)

    route = [elm for elm in input if elm['type'] == 'polyline'] 
    polygons = [elm for elm in input if elm['type'] == 'polygon']

    assert len(route) == 1, f"only 1 route allowed... for now, {len(route)} detected"
    route = route[0]

    # Field and Link classes do operations with Portal class which has a static parameter portal_map with all included portals
    # throws warnings when len(polygon['latLangs'] != 3)
    fields: list[Field] = [Field.fromPolygon(polygon) for polygon in polygons]
    links: list[Link] = []
    for field in fields:
        links += field.links
    print(links)