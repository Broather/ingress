python CLI tool that automates planning fields in the mobile game Ingress made by Niantic

in it's current state, the script relies on the use of the browser add-on [IITC](https://github.com/IITC-CE/ingress-intel-total-conversion) and it's "Draw tools" plug-in aswell as the "Bookmarks for maps and portals" plug-in.

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
### install requirements (just pyperclip for now)
```
python -m pip install -r requirements.txt
```
### run it for the first time
```
python main.py
```
