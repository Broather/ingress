import getopt
import sys
import json
import matplotlib.pyplot as plt
from main import Ingress, Portal

def help():
    print("syntax: render.py [-h] path/to/output.json")

def main(opts, args):
    for o, a in opts:
        if o == "-h":
            help()
            return
        elif o == "-p":
            for portal_group in a.split(","):
                with open(Ingress.portal_group_map[portal_group.strip()], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])

    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"

    assert len(args) == 1, f"only 1 positional argument allowed, {len(args)} detected"
    path = args[0]

    with open(path, "r", encoding="utf-8") as f:
        input = json.load(f)

    longitudes = list(map(Portal.get_lng, Ingress.used_portals))
    latitudes = list(map(Portal.get_lat, Ingress.used_portals))

    plt.scatter(longitudes, latitudes, color="orange", label="label")
    plt.show()
    # plt.savefig("output.png")
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:", [])
    main(opts, args)