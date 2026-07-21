A terminal UI to create 2D shapes and search for them in a set of points

Made for use with the geo-location based mobile game "Ingress" made by "Niantic"

In it's current state, the script relies on the use of the browser add-on [IITC](https://iitc.app/download_desktop) along with the following plugins:

- [Draw tools](https://iitc.app/download_desktop#draw-tools-by-breunigs)
- [Bookmarks for maps and portals](https://iitc.app/download_desktop#bookmarks-by-ZasoGD)

Once setup, add any and all portals to bookmarks and copy/paste the data to a `.json` file in a `portals` directory in project root. 

Use IITC's right hand side menu > `Bookmarks Opt` > `Copy bookmarks`

## quickstart
### create a virtual environment
```
python -m venv venv
```
### activate the virtual environment
#### in git bash
```
. venv/bin/activate
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
