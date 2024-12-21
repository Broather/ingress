import sys
import getopt
import json
from main import Ingress

def assistance():
    print("Syntax: python rekey.py [-h] path/to/plan.json")
    print("""
    Modifies plan.json with the correct amount of keys reqired from each portal
    Options:
        h: calls this help function
    """)
def main(opts, args):
    assert len(args) == 1, f"ERROR: rekey only accepts one positional argument, {len(args)} were given"

    for o in opts:
        if o == "-h":
            assistance()
            return

    with open(args[0], "r", encoding="utf-8") as f:
        input = json.load(f)
    steps: dict = input[0]["Steps"]

    keys_required = {key: 0 for key in steps}

    for key in steps:
        for link in steps[key]["links"]:
            keys_required[link] += 1

    for key in steps:
        steps[key]["keys"] = keys_required[key]

    input[0]["Steps"] = steps
    input[0]["Total_keys_required"] = sum(keys_required.values())

    with open("rekey.json", "w", encoding="utf-8") as f:
        json.dump(input, f, ensure_ascii=False, indent=2)
        print("rekey.json created successfuly")

    Ingress.copy_to_clipboard(input)

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", [])
    main(opts, args)