import getopt
import sys
import json
import matplotlib.pyplot as plt
from main import Ingress, Portal

def plot_IITC_elements(input):
    MARGIN = 5*10**-4
    all_longitudes = []
    all_latitudes = []
    for IITC_element in input:
        longitudes = list(map(lambda e: e["lng"], IITC_element["latLngs"]))
        latitudes = list(map(lambda e: e["lat"], IITC_element["latLngs"]))
        all_longitudes.extend(longitudes)
        all_latitudes.extend(latitudes)

        if IITC_element["type"] == "polyline":
            plt.plot(longitudes, latitudes, color=IITC_element["color"], zorder=1)
        elif IITC_element["type"] == "polygon":
            plt.fill(longitudes, latitudes, facecolor=IITC_element["color"], edgecolor=IITC_element["color"], linewidth=2, alpha=0.3)
        else:
            print(f"WARNING: plot_IITC_elements attepting to plot IITC element of type {IITC_element['type']}")
    return ((min(all_longitudes)-MARGIN, max(all_longitudes)+MARGIN), (min(all_latitudes)-MARGIN, max(all_latitudes)+MARGIN))

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

    plt.figure(facecolor='#262626')
    plt.axis("off")

    xlim, ylim = plot_IITC_elements(input)
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    longitudes = list(map(Portal.get_lng, Ingress.used_portals))
    latitudes = list(map(Portal.get_lat, Ingress.used_portals))
    plt.scatter(longitudes, latitudes, color="#ff6600")

    plt.show()
    # plt.savefig("output.png")
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hp:", [])
    main(opts, args)