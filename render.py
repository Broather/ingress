import getopt
import itertools
import sys
import json
import matplotlib.pyplot as plt
import os
import imageio
from main import Ingress, Portal, Link, Field

def simulate_step(active_portal: Portal, portals_to_link_to: tuple[Portal], previous_steps: list[tuple[Portal|Link|Field]]) -> tuple[Link|Field]:
    links = tuple(map(active_portal.create_link, portals_to_link_to))

    output = []
    for link in links:
        touching_links = tuple(filter(link.is_touching, Ingress.flatten_iterable_of_tuples(previous_steps + output)))
        fields = link.get_fields(touching_links)
        output.append((link,) + fields)

    return Ingress.flatten_iterable_of_tuples(output)

def simulate_plan(plan: dict):
    """returns [[(Portal, Portal), ([Portal], [Link], [Link, Link], ...)],
                ((Portal, Portal), ([Portal], [Link], [Link, Link], ...))]"""

    output = []
    all_steps = []
    routes = plan["routes"]
    for route in routes:
        route_steps = []
        steps = route["steps"]
        for active_portal in map(Ingress.find_portal, steps):
            portals_to_link_to = tuple(map(Ingress.find_portal, steps[active_portal.get_label()]["links"]))
            if len(portals_to_link_to) == 0: 
                route_steps.append((active_portal,))
                continue

            route_steps.append(simulate_step(active_portal, portals_to_link_to, all_steps+route_steps))
            

        bounding_box = Ingress.bounding_box(Ingress.flatten_iterable_of_tuples(route_steps))
        all_steps.extend(route_steps)
        output.append((bounding_box, route_steps))
    
    return output

def plot_portals(portals: Portal, color = "#ff6600"):
    if all(map(lambda p: isinstance(p, Portal), portals)):
        longitudes = list(map(Portal.get_lng, portals))
        latitudes = list(map(Portal.get_lat, portals))
        plt.plot(longitudes, latitudes, "o", color=color)
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
            plt.plot(IITC_element["latLng"]["lng"], IITC_element["latLng"]["lat"], "o", color=IITC_element["color"], zorder=3)
        else:
            print(f"WARNING: plot_IITC_elements attepting to plot IITC element of type {IITC_element['type']}")

def help():
    print("syntax: render.py [-h] path/to/output.json")

def main(opts, args):
    # defaults
    only_links = False
    image_folder_path = "./gif_source"

    for o, a in opts:
        if o == "-h":
            help()
            return
        elif o == "-l":
            only_links = True
        elif o == "-a":
            for portal_group in Ingress.portal_group_map:
                with open(Ingress.portal_group_map[portal_group], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])
        elif o == "-p":
            for portal_group in a.split(","):
                with open(Ingress.portal_group_map[portal_group.strip()], "r", encoding='utf-8') as f:
                    Ingress.add_from_bkmrk(json.load(f)['portals']['idOthers']['bkmrk'])

    assert Ingress.used_portals, f"no portals selected to split with, make sure you are using -p"

    assert len(args) == 1, f"ERROR: only 1 positional argument allowed, {len(args)} detected"
    path = args[0]
    try:
        with open(path, "r", encoding="utf-8") as f:
            plan: list[dict] = json.load(f)
    except FileNotFoundError:
        print(f"{path} not found")
        return
    
    # TODO: there are no fields in the simulation
    simulation = simulate_plan(plan)
    create_directory(image_folder_path)
    
    previous_route_steps = []
    route_nr = 0
    for bounding_box, steps in simulation:
        step_nr = 0
        for active_step, leading_steps in zip(steps, itertools.accumulate(steps)):
            clear_and_setup_plot(bounding_box)
            plot_portals(Ingress.used_portals)
            rendered_leading_steps = Ingress.render(tuple(previous_route_steps) + leading_steps, lambda _: (0, 1, 0))
            rendered_active_step = Ingress.render(active_step, lambda _: (0, 0, 1))
            plot_IITC_elements(rendered_leading_steps)
            plot_IITC_elements(rendered_active_step)
            plt.savefig(f"{image_folder_path}/{route_nr}-{step_nr}.png")
            step_nr += 1
        previous_route_steps.extend(Ingress.flatten_iterable_of_tuples(steps))
        route_nr += 1

    create_gif(image_folder_path, f"{image_folder_path}/_gif.gif", )

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hlap:", [])
    main(opts, args)