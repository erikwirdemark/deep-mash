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
**GTZAN-stems-preprocessing**:
- Tog några delar från Olles dataloader.py och gjorde en funktion GTZANStemsDataset._preprocess som
  downsamplar samt mixar non-vocals och sparar som vocals.pt & non-vocals.pt (tensorer) i ett nytt
  gtzan-stems-processed directory för varje låt, så man slipper upprepa @ runtime
- Tar ~1min på min dator, går från 20G till 7.2G
- Downsamplar till 16kHz då det är det dom använder i cocola, men skulle ev kunna testa högre senare också 
- Jag lyssnade igenom ett par av låtarna, verkar som att mixning av non-vocals genom att bara
  summera och normalisera funkar bra. Ett möjligt problem är att vissa låtar verkar ha rätt lite
  vocals med längre perioder av tystnad, men borde vara lungt så länge vi tar tillräckligt långa
  chunks (skulle säga minst 10s, kanske helst hela 30s ifall det funkar). Och antagligen borde man
  ta bort all klassisk musik, och ksk försöka filtrera bort låtar med för lite vocals (typ genom
  threshold på genomsnitt / nån kvantil av ljudvågen?)
