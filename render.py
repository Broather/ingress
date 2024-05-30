import getopt
import itertools
import sys
import json
import matplotlib.pyplot as plt
import os
import imageio
from ingress import Ingress, Portal, BoundingBox

def plot_portals(*portals: Portal, color = "#ff6600", zorder: int = 1):
    if all(map(lambda p: isinstance(p, Portal), portals)):
        longitudes = list(map(Portal.get_lng, portals))
        latitudes = list(map(Portal.get_lat, portals))
        plt.scatter(longitudes, latitudes, s=10, c=color, zorder=zorder)
    else:
        assert False, f"plot_portals was given argument of type NOT Portal: {portals}"

def create_gif(images_folder: str, output_gif_path: str, fps: int = 5):
    """creates a gif from a directory of .png images (made by GPT-3.5 modified by me)"""
    images = []

    # sorted_file_names = sorted(filter(lambda f_name: f_name.endswith(".png"), os.listdir(image_folder_path)), key=lambda f_name: tuple(map(int, f_name.removesuffix(".png").split("-"))))
    sorted_png_file_names = sorted(filter(lambda f_name: f_name.endswith(".png"), os.listdir(images_folder)), key=lambda f_name: tuple(map(int, f_name.removesuffix(".png").split("-"))))
    # Read all PNG images in the folder
    for file_name in sorted_png_file_names:
        file_path = os.path.join(images_folder, file_name)
        images.append(imageio.v2.imread(file_path))

    # Create GIF
    imageio.mimwrite(output_gif_path, images, fps=fps, loop=0)

def set_plot_bounds(bb: BoundingBox) -> None:
    plt.xlim((bb.tl.lng, bb.br.lng))
    plt.ylim((bb.br.lat, bb.tl.lat))

def clear_and_setup_plot() -> None:
    plt.close()

    # and set it back up
    plt.figure(facecolor='#262626')
    plt.axis("off")

def create_directory(dir_path: str) -> None:
    exists = os.path.exists(dir_path)
    if not exists:
        os.makedirs(dir_path)
    else:
        png_file_names = filter(lambda f_name: f_name.endswith(".png"), os.listdir(dir_path))
        for file_name in png_file_names:
            file_path = f"{dir_path}/{file_name}"
            os.remove(file_path)
        
def plot_IITC_elements(input: list[dict]) -> None:
    for IITC_element in input:
        if latLngs := IITC_element.get("latLngs"):
            longitudes = list(map(lambda e: e["lng"], latLngs))
            latitudes = list(map(lambda e: e["lat"], latLngs))

        if IITC_element["type"] == "polyline":
            # default zorder is 2
            plt.plot(longitudes, latitudes, color=IITC_element["color"], zorder=1)
        elif IITC_element["type"] == "polygon":
            plt.fill(longitudes, latitudes, facecolor=IITC_element["color"], edgecolor=IITC_element["color"], linewidth=2, alpha=0.2)
        elif IITC_element["type"] == "marker":
            plot_portals(Ingress.find_portal_from_latLng(IITC_element["latLng"]), color=IITC_element["color"], zorder=3)
        else:
            print(f"WARNING: plot_IITC_elements attepting to plot IITC element of type {IITC_element['type']}")

def help():
    print("syntax: render.py [-hc] [-p comma] path/to/plan.json")

def main(opts, args):
    # defaults
    image_folder_path = "./gif_source"
    chunk_steps = False
    plan_path = "./plan.json"

    for o, a in opts:
        if o == "-h":
            help()
            return
        elif o == "-c":
            chunk_steps = True
        elif o == "-p":
            if a.lower() == "all":
                for file in Ingress.portal_group_map.values():
                    Ingress.add_from_bkmrk_file(file)
            else:
                for portal_group in a.split(","):
                    Ingress.add_from_bkmrk_file(Ingress.portal_group_map[portal_group.strip()])
        elif o == "--plan":
            plan_path = a
        else:
            assert False, f"ERROR: unhandled option: {o}"

    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"

    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan: list[dict] = json.load(f)
    except FileNotFoundError:
        print(f"{plan_path} not found")
        return

    create_directory(image_folder_path)

    simulation = Ingress.simulate_plan(plan, chunk_steps=chunk_steps)
    route_nr = 0
    for route in simulation:
        clear_and_setup_plot()
        bb: BoundingBox = simulation[route]["bounding_box"]
        set_plot_bounds(bb)
        plot_portals(*filter(bb.is_in, Ingress.used_portals))
        step_nr = 0
        for active_step in simulation[route]["steps"]:
            active_step = Ingress.render(active_step, Ingress.color_maps["green"])
            plot_IITC_elements(active_step)
            plt.savefig(f"{image_folder_path}/{route_nr}-{step_nr}.png", dpi=150)
            step_nr += 1
        route_nr += 1

    create_gif(image_folder_path, f"{image_folder_path}/_gif.gif", )

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hcp:", ["plan="])
    main(opts, args)