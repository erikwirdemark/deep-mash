## Dev setup
Clone the repo and setup a virtual environment, for example with
```
python -m venv .venv 
source .venv/bin/activate
```
And then install the project in editable mode:
```
pip install -e .
```
Or alternatively if you use [uv](https://docs.astral.sh/uv/) (much faster and generally nicer than plain pip):
```
uv venv 
uv sync
```
