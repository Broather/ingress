import getopt
import sys
import json
import os
import imageio
import matplotlib.pyplot as plt
from ingress import Ingress, Portal, Link, Field, BoundingBox

def plot_portals(*portals: Portal, color = "#ff6600", zorder: int = 1, portal_method_to_show: callable = None):
    if all(map(lambda p: isinstance(p, Portal), portals)):
        longitudes = list(map(Portal.get_lng, portals))
        latitudes = list(map(Portal.get_lat, portals))
        plt.scatter(longitudes, latitudes, s=10, c=color, zorder=zorder)
        if portal_method_to_show:
            for p in portals:
                plt.annotate(portal_method_to_show(p), (p.get_lng(), p.get_lat() - .0001),
                            color = "white",
                            fontsize = 7,
                            ha = "center")
    else:
        assert False, f"plot_portals was given argument of type NOT Portal: {portals}"

def create_gif(images_folder: str, output_gif_path: str, fps: int = 5):
    """creates a gif from a directory of .png images (made by GPT-3.5 modified by me)"""

    png_file_names = list(filter(lambda file_name: file_name.endswith(".png"), os.listdir(images_folder)))
    png_file_names.sort(key=lambda file_name: tuple(map(int, file_name.removesuffix(".png").split("-"))))

    # Read all PNG images in the folder
    images = []
    for file_name in png_file_names:
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

        if IITC_element["type"] == "polyline":
            longitudes = list(map(lambda e: e["lng"], IITC_element["latLngs"]))
            latitudes = list(map(lambda e: e["lat"], IITC_element["latLngs"]))

            # default zorder is 2
            plt.plot(longitudes, latitudes,
                    color = IITC_element["color"],
                    zorder = 1)
        elif IITC_element["type"] == "polygon":
            longitudes = list(map(lambda e: e["lng"], IITC_element["latLngs"]))
            latitudes = list(map(lambda e: e["lat"], IITC_element["latLngs"]))

            plt.fill(longitudes, latitudes,
                    facecolor = IITC_element["color"],
                    edgecolor = IITC_element["color"],
                    linewidth = 2,
                    alpha = 0.2)
        elif IITC_element["type"] == "marker":
            plot_portals(Ingress.find_portal_from_latLng(IITC_element["latLng"]),
                        color = IITC_element["color"],
                        zorder = 3)
        else:
            print(f"WARNING: plot_IITC_elements attepting to plot IITC element of type {IITC_element['type']}")

def test_continuity(simulation: dict):
    # {bounding_box: BoundingBox, steps: [[<Portal|Link|Field>]]}
    all_steps = map(lambda r: simulation[r]["steps"], simulation)
    simulation_objects = Ingress.flatten_iterable_of_tuples(Ingress.flatten_iterable_of_tuples(all_steps))

    claimed_portals = set()
    fields = set()
    for o in simulation_objects:
        if isinstance(o, Portal):
            claimed_portals.add(o)
        if isinstance(o, Link):
            if not set(o.get_portals()).issubset(claimed_portals):
                print(f"WARNING: link created with unclaimed portal: {o}")
            if not o.get_length() < 2000 and any(map(o.get_portals()[0].is_within_field, fields)):
                print(f"WARNING: link cannot be created because it is within field (and is long): {o}")
        if isinstance(o, Field):
            fields.add(o)

def assistance():
    print("syntax: render.py [-hc] [-p comma] path/to/plan.json")

def main(options):
    # defaults
    image_folder_path = "./gif_source"
    chunk_together = False
    show_portal_names = False
    plan_path = "./plan.json"

    for o, a in options:
        if o == "-h":
            assistance()
            return
        elif o == "-c":
            chunk_together = True
        elif o == "-n":
            show_portal_names = True
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

    assert Ingress.used_portals, "no portals selected to split with, make sure you are using -p"

    plan = Ingress.read_json(plan_path, assistance)

    create_directory(image_folder_path)

    simulation = Ingress.simulate_plan(plan, chunk_together=chunk_together)
    test_continuity(simulation)
    route_nr = 0
    for route_data in simulation.values():
        clear_and_setup_plot()
        bb: BoundingBox = route_data["bounding_box"]
        set_plot_bounds(bb)
        plot_portals(*filter(bb.is_in, Ingress.used_portals), portal_method_to_show = Portal.get_value)
        step_nr = 0
        for active_step in route_data["steps"]:
            active_step = Ingress.render(active_step, Ingress.color_maps["green"])
            plot_IITC_elements(active_step)
            plt.savefig(f"{image_folder_path}/{route_nr}-{step_nr}.png", dpi=150)
            step_nr += 1
        route_nr += 1

    create_gif(image_folder_path, f"{image_folder_path}/_gif.gif", )

if __name__ == "__main__":
    opts, _ = getopt.getopt(sys.argv[1:], "hcnp:", ["plan="])
    main(opts)
