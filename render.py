import getopt
import sys
import json
import matplotlib.pyplot as plt
import os
import imageio
from main import Ingress, Portal
from snapshot import render_plan 

def create_gif(images_folder: str, output_gif_path: str, fps: float = 5):
    """creates a gif from a directory of .png images (made by GPT-3.5)"""
    images = []
    
    png_filenames: list[str] = list(filter(lambda filename: filename.endswith(".png"), os.listdir(images_folder)))
    png_filenames.sort(key=lambda x: int(x.split('.')[0]))
    # Read all PNG images in the folder
    for filename in png_filenames:
        filepath = os.path.join(images_folder, filename)
        images.append(imageio.v2.imread(filepath))

    # Create GIF
    imageio.mimwrite(output_gif_path, images, fps=fps, loop=0)
    
def clear_and_setup_plot() -> None:
    plt.close()

     # general plot setup
    plt.figure(facecolor='#262626')
    plt.axis("off")

def create_directory(path: str) -> None:
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)
        
def plot_IITC_elements(input, center) -> None:
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
    if center and (all_longitudes or all_latitudes):
        plt.xlim((min(all_longitudes)-MARGIN, max(all_longitudes)+MARGIN))
        plt.ylim((min(all_latitudes)-MARGIN, max(all_latitudes)+MARGIN))

def plot_plan(plan):
    pass
    
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
        # center around IITC elements
        xlim, ylim = plot_IITC_elements(input)
        plt.xlim(*xlim)
        plt.ylim(*ylim)

        # add portals on top
        longitudes = list(map(Portal.get_lng, Ingress.used_portals))
        latitudes = list(map(Portal.get_lat, Ingress.used_portals))
        plt.scatter(longitudes, latitudes, color="#ff6600")
        
        plt.savefig("output.png")
    elif steps := input[0].get("Steps"):
        print("making a gif from a plan\'s steps")
        for step_to_stop_at in range(1, len(steps)+1):
            clear_and_setup_plot()

            IITC_elements = render_plan(steps, step_to_stop_at, only_links)
            plot_IITC_elements(IITC_elements, True)
            
            # add portals on top
            longitudes = list(map(Portal.get_lng, Ingress.used_portals))
            latitudes = list(map(Portal.get_lat, Ingress.used_portals))
            plt.scatter(longitudes, latitudes, color="#ff6600")
            
            create_directory("./gif_source")
            plt.savefig(f"./gif_source/{step_to_stop_at}.png")

        create_gif("./gif_source", "./gif_source/_gif.gif", )

    else:
        assert False, f"ERROR: unrecognised input file {path}"
        
    
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hlp:", [])
    main(opts, args)