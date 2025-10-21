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
Or alternatively if you use [uv](https://docs.astral.sh/uv/) (much faster and generally nicer than
plain pip imo):
```
uv venv
source .venv/bin/activate
uv sync
```

## Log
**MUSDB** 
- La till MUSDB18-dataset (laddade ner från https://zenodo.org/records/1117372), tänkte kanske kunde
  använda som testset sen, då en nackdel med att bara testa på gtzan-stems är att modellen kanske
  bara lär sig ngn detalj i separationsmodellen, speciellt då separeringen de använt eventuellt
  inte var jättebra (kan tex höra instrumentals ganska tydligt i bakgrunden på vissa vocals). MUSDB
  skapades tydligen direkt från multitrack recordings, så har inte samma problem. 
- Bara 150 låtar, men de flesta är mycket längre än 30s - med 15s chunks får jag 2084 chunks för hela
  musdb, jmft med 2000 för gtzan-stems. 

**Model** 
- Började på models/cocola_cnn.py, tänker vi kan utgå från deras eftersom dom fick det att funka,
  och sen ändra/lägga till saker om vi vill.

