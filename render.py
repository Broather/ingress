import getopt
import sys
import json
import matplotlib.pyplot as plt
import os
import imageio
import math
from main import Ingress, Portal
from snapshot import render_plan 

def plot_portals(*args: Portal, color = "#ff6600"):
    if all(map(lambda p: isinstance(p, Portal), args)):
        longitudes = list(map(Portal.get_lng, args))
        latitudes = list(map(Portal.get_lat, args))
        plt.plot(longitudes, latitudes, "o", color=color)
    else:
        assert False, f"plot_portals was given argument of type NOT Portal, args: {args}"

def pythagoras(a):
    return math.sqrt(a**2+a**2)

def create_gif(images_folder: str, output_gif_path: str, fps: int = 5):
    """creates a gif from a directory of .png images (made by GPT-3.5 modified by me)"""
    images = []
    
    png_filenames: list[str] = list(filter(lambda filename: filename.endswith(".png"), os.listdir(images_folder)))
    png_filenames.sort(key=lambda x: int(x.split('.')[0]))
    # Read all PNG images in the folder
    for filename in png_filenames:
        filepath = os.path.join(images_folder, filename)
        images.append(imageio.v2.imread(filepath))

    # Create GIF
    imageio.mimwrite(output_gif_path, images, fps=fps, loop=0)

# TODO: put in a class with self.axis_limits = None
def clear_and_setup_plot(axis_limits: tuple[tuple, tuple] = None) -> None:
    plt.close()

     # general plot setup
    plt.figure(facecolor='#262626')
    plt.axis("off")

    if axis_limits:
        plt.xlim(axis_limits[0])
        plt.ylim(axis_limits[1])

def create_directory(path: str) -> None:
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)
        
def plot_IITC_elements(input: list[dict], recenter: bool = False) -> None:
    all_longitudes = []
    all_latitudes = []
    
    for IITC_element in input:
        if latLngs := IITC_element.get("latLngs"):
            longitudes = list(map(lambda e: e["lng"], latLngs))
            latitudes = list(map(lambda e: e["lat"], latLngs))
            all_longitudes.extend(longitudes)
            all_latitudes.extend(latitudes)

        if IITC_element["type"] == "polyline":
            plt.plot(longitudes, latitudes, color=IITC_element["color"], zorder=1)
        elif IITC_element["type"] == "polygon":
            plt.fill(longitudes, latitudes, facecolor=IITC_element["color"], edgecolor=IITC_element["color"], linewidth=2, alpha=0.2)
        elif IITC_element["type"] == "marker":
            plt.plot(IITC_element["latLng"]["lng"], IITC_element["latLng"]["lat"], "o", color=IITC_element["color"], zorder=3)
        else:
            print(f"WARNING: plot_IITC_elements attepting to plot IITC element of type {IITC_element['type']}")

    if recenter and all_longitudes and all_latitudes:
        # TODO: make sure the fact that im assuming northern hemesphere and eastern hemesphere doesnt screw me over
        tl = Portal("top left", lat = max(all_latitudes), lng = min(all_longitudes))
        br = Portal("bottom right", lat = min(all_latitudes), lng = max(all_longitudes))
        center = Portal("center", lat = tl.lat-(abs(tl.lat - br.lat)/2), lng = tl.lng+(abs(tl.lng-br.lng)/2))
        # plot_portals(tl, br, center, color="red")
        
        delta_longitude = abs(tl.lng - br.lng)
        delta_latitude = abs(br.lat - tl.lat)
        if delta_longitude > delta_latitude:
            PADDING = delta_longitude*.02
            # put points on left and right side centers
            ptl = Portal("top or left center point", lat = center.lat, lng = tl.lng)
            pbr = Portal("bottom or right center point", lat = center.lat, lng = br.lng)
            tl.transform(ptl, delta_longitude/2 - delta_latitude/2)
            br.transform(pbr, delta_longitude/2 - delta_latitude/2)
        elif delta_latitude > delta_longitude:
            PADDING = delta_latitude*.02
            # put points on top and bottom side centers
            ptl = Portal("top or left center point", lat = tl.lat, lng = center.lng)
            pbr = Portal("bottom or right center point", lat = br.lat, lng = center.lng)
            tl.transform(ptl, delta_latitude/2 - delta_longitude/2)
            br.transform(pbr, delta_latitude/2 - delta_longitude/2)
        else:
            PADDING = delta_latitude*.02
            # deltas have 1:1 ratio, no action required
            pass

        tl.transform(center, pythagoras(PADDING)) # moves tl outwards by sqrt(PADDING**2+PADDING**2)
        br.transform(center, pythagoras(PADDING)) # moves tl outwards by sqrt(PADDING**2+PADDING**2)
        plt.xlim(tl.lng, br.lng)
        plt.ylim(br.lat, tl.lat)

def help():
    print("syntax: render.py [-h] path/to/output.json")

def main(opts, args):
    # defaults
    only_links = False

    for o, a in opts:
        if o == "-h":
            help()
            return
        elif o == "-l":
            only_links = True
        elif o == "-p":
            for portal_group in a.split(","):
                with open(Ingress.portal_group_map[portal_group.strip()], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])

    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"

    assert len(args) == 1, f"only 1 positional argument allowed, {len(args)} detected"
    path = args[0]
    try:
        with open(path, "r", encoding="utf-8") as f:
            input: list[dict] = json.load(f)
    except FileNotFoundError:
        print(f"{path} not found")
        return
    
    clear_and_setup_plot()

    if input[0].get("type"):
        print("making a png from IITC elements")
        plot_IITC_elements(input, recenter=True)
        plot_portals(*Ingress.used_portals)

        plt.savefig("output.png")
        print("output.png created successfuly")
    elif steps := input[0].get("Steps"):
        print("making a gif from a plan\'s steps")
        # render the last step 
        # TODO: Renderer.plot_step?(step_to_stop_at) that plt.savefig aswell (seems like a waste to plot and not save it)
        last_step = render_plan(steps, len(steps), only_links)
        plot_IITC_elements(last_step)
        # get the axis limits 
        axis_limits = (plt.xlim(), plt.ylim())
        # use them to make the rest of the frames 
        for step_to_stop_at in range(1, len(steps)+1):
            clear_and_setup_plot(axis_limits)

            IITC_elements = render_plan(steps, step_to_stop_at, only_links)
            plot_IITC_elements(IITC_elements)

            # add portals on top
            plot_portals(*Ingress.used_portals)

            create_directory("./gif_source")
            plt.savefig(f"./gif_source/{step_to_stop_at}.png")

        create_gif("./gif_source", "./gif_source/_gif.gif", )

    else:
        assert False, f"ERROR: unrecognised input file {path}"
        
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hlp:", [])
    main(opts, args)