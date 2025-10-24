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
plain pip):
```
uv venv
source .venv/bin/activate
uv sync
```

## Getting started
This package includes the DeepMash model, with the purpose of learning how to embed instrumental and vocal stems in an embedding space so that corresponding stems are embedded closely together. Along with the model architecture, the package includes functionality to train the model, preprocess data, save embeddings, query a database of saved embeddings and create mash ups of vocal and instrumental stems. 

To get started, firstly take a look at the ``config`` folder. Here we have placed a template for a configuration file for running an experiment, before moving forward assert that all is to your liking and that paths are modified to point to where you have stored your dataset.

## Training a model
To train a model, simply run the ``main.py`` script and pass your configuration file along with the train command:
```
python main.py -c config/<your config file> train
```
Or if you are using UV:
```
uv run main.py -c config/<your config file> train
```
If you have configured your model to save the embeddings, all instrumental embeddings of your dataset will be saved to your device.

## Querying the database
Once you have a trained model, you must specify the file path of the saved checkpoint in your configuration file. Once you have done that, you can finally start making mashups! To find the best matching instrumentals to your vocal wav or mp3 file, run the following:
```
uv run main.py -c config/<your config file> -q <path to your vocal stem>
```
You will be presented with a choice of instrumentals to mix with, and once selected your mashup will be outputted in the folder you have configured in your yaml file. 

### Note

In order to use this package, you need to download the GTZAN Stems dataset. The dataset is available via Kaggle at the following [link](https://www.kaggle.com/datasets/mantasu/gtzan-stems).
