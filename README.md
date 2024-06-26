python CLI tool that automates planning fields in the mobile game Ingress made by Niantic

in it's current state, the script relies on the use of the browser add-on [IITC](https://github.com/IITC-CE/ingress-intel-total-conversion) and it's "Draw tools" plug-in aswell as the "Bookmarks for maps and portals" plug-in. ([setup instuctions](https://iitc.app/download_desktop))

once setup, you will need to add any and all portals you wish to make fields with and export them to a .json file under the "portals" directory

## quickstart
### create a virtual environment
```
python -m venv venv
```
### activate the virtual environment
#### in git bash
```
. venv/Scripts/activate
```
#### in powershell
```
.\venv\Scripts\activate
```
### install requirements
```
python -m pip install -r requirements.txt
```
### run program for the first time
```
python main.py
```

## additional reading
### list of scripts in the project:
- *main.py* (takes contents of input.json and produces output.json and plan.json)
- *analyze.py* (takes contents of input.json or plan.json and produces a matplotlib pyplot)
- *render.py* (takes contents of plan.json and produces a .gif animation)
- *rekey.py* (takes contents of plan.json and modifies it with correct "keys" values)

each has a -h option for a list of available options
