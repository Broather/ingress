import sys
import getopt
import json

def main(opts, args):
    assert len(args) == 1, f"ERROR: rekey only accepts one positional argument, {len(args)} were given"

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

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "", [])
    main(opts, args)

# {
#     "key1": [],
#     "key2": ["key1"],
#     "key3": ["key1", "key2"],
#     "key4": ["key2", "key3"],
#  }
# {
#     "key1": 2,
#     "key2": 2,
#     "key3": 1,
#     "key4": 0,
#  }