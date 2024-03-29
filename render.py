import getopt
import itertools
import sys
import json
import matplotlib.pyplot as plt
import os
import imageio
from main import Ingress, Portal, Link, Field

def plot_portals(*portals: Portal, color = "#ff6600", zorder: int = 1):
    if all(map(lambda p: isinstance(p, Portal), portals)):
        longitudes = list(map(Portal.get_lng, portals))
        latitudes = list(map(Portal.get_lat, portals))
        plt.scatter(longitudes, latitudes, s=10, c=color, zorder=zorder)
    else:
        assert False, f"plot_portals was given argument of type NOT Portal, portals: {portals}"

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
    
def clear_and_setup_plot(bounding_box: tuple[Portal, Portal]) -> None:
    plt.close()

    # and set it back up
    plt.figure(facecolor='#262626')
    plt.axis("off")
    tl, br = bounding_box
    plt.xlim((tl.lng, br.lng))
    plt.ylim((br.lat, tl.lat))

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
            plt.plot(longitudes, latitudes, color=IITC_element["color"], zorder=1)
        elif IITC_element["type"] == "polygon":
            plt.fill(longitudes, latitudes, facecolor=IITC_element["color"], edgecolor=IITC_element["color"], linewidth=2, alpha=0.2)
        elif IITC_element["type"] == "marker":
            plot_portals(Ingress.find_portal_from_latLng(IITC_element["latLng"]), color=IITC_element["color"], zorder=3)
        else:
            print(f"WARNING: plot_IITC_elements attepting to plot IITC element of type {IITC_element['type']}")

def help():
    print("syntax: render.py [-h] path/to/output.json")

def main(opts, args):
    # defaults
    image_folder_path = "./gif_source"
    simulation_slice = None

    for o, a in opts:
        if o == "-h":
            help()
            return
        elif o == "-s":
            if len(a.split(",")) == 3:
                route_index, from_step, to_step = map(int, a.split(","))
                simulation_slice = (route_index, from_step, to_step)
            else: 
                print("ERROR: -s option expects comma separated list of 3 integers: route_index,from_step,to_step")
                return
        elif o == "-a":
            for portal_group_file_path in Ingress.portal_group_map.values():
                Ingress.add_from_bkmrk_file(portal_group_file_path)
        elif o == "-p":
            for portal_group in a.split(","):
                Ingress.add_from_bkmrk_file(Ingress.portal_group_map[portal_group.strip()])

    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"

    assert len(args) == 1, f"ERROR: only 1 positional argument allowed, {len(args)} detected"
    path = args[0]
    try:
        with open(path, "r", encoding="utf-8") as f:
            plan: list[dict] = json.load(f)
    except FileNotFoundError:
        print(f"{path} not found")
        return
    
    simulation = Ingress.simulate_plan(plan)
    if simulation_slice:
        route_index, from_step, to_step = simulation_slice
        sliced_steps = simulation[route_index][1][from_step:to_step]
        portals = map(Ingress.find_portal, list(plan["routes"][route_index]["steps"].keys())[from_step:to_step])
        simulation = (tuple([Ingress.bounding_box(portals, grow_to_square=True), sliced_steps]), )

    create_directory(image_folder_path)
    
    previous_route_steps = []
    route_nr = 0
    for bounding_box, steps in simulation:
        step_nr = 0
        for active_step, leading_steps in zip(steps, itertools.accumulate(steps)):
            clear_and_setup_plot(bounding_box)
            plot_portals(*Ingress.used_portals)
            rendered_leading_steps = Ingress.render(tuple(previous_route_steps) + leading_steps, lambda _: (0, 1, 0))
            rendered_active_step = Ingress.render(active_step, lambda _: (0, 0, 1))
            plot_IITC_elements(rendered_leading_steps)
            plot_IITC_elements(rendered_active_step)
            plt.savefig(f"{image_folder_path}/{route_nr}-{step_nr}.png", dpi=150)
            step_nr += 1
        previous_route_steps.extend(Ingress.flatten_iterable_of_tuples(steps))
        route_nr += 1

    create_gif(image_folder_path, f"{image_folder_path}/_gif.gif", )

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hs:ap:", [])
    main(opts, args)