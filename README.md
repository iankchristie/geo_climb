# Adventure is Out There!

Here is the readme for our Geospatial Machine Learning project GEOClimb. It's goal is to predict unexplore rock climbing areas throughout the world.

## Architecture:

The model architecture receives embeddings as input and uses a simple MLP to produce a probability of climbing within an area given the embeddings.

Downloaders are responsible for downloading the data and storing it in the data directory. Data Cleaners are responsible for transforming the raw data to clean and standardize it before the encoding process. Encoding will take the raw data and produce embeddings. Model components will run the model and use Weights and Biases as a logger. Vizualization files are responsible for all visualizations of the data.

## Getting Started:

1. Download Conda
2. Create Conda environment: `conda env create -f environment.yml`
3. Initialize environment: `conda activate geo_climb`

## Working:

- If you install any packages, make sure to `conda env export --name geo_climb --no-builds > environment.yml`
- If you pull and packages need to be installed, pull them from the conda environment.yml `conda env update --file environment.yml --prune`

## Dataset

The dataset can be downloaded from this link: https://drive.google.com/drive/folders/1lItS74OOocI-ppHx5q0SmEKHP-G_-gmg?usp=drive_link

It contains raw sentinel2 RGB images, DEM, and lithology information for 12,742 climbing (labeled) locations and 12,692 unlabeled locations. It also contains multiple generated embeddings for the raw data. The model uses the generated embeddings as input.

## Creating the Dataset

If you want to build the dataset from scratch you'll need to follow the steps below.

### Climbing Data

1. Download mountain project data from Kaggle here: https://www.kaggle.com/datasets/pdegner/mountain-project-rotues-and-forums
2. Load the mp_routes.csv into `data/labeled/climbing` directory (You may need to create it).
3. You can clean and run the data using `python3 data/labeled/mtp_cleaner.py`

### Earth Engine Data

1. Create your earth engine account: https://earthengine.google.com/
2. Authenticate with earth engine `earthengine authenticate`. (This binary should already be installed via conda)
3. Then you can edit and run `downloaders/downloader.py` to download Sentinel2 and DEM data. There are further instructions in the file.

### Lithology Data

When running `downloaders/downloader.py` to download lithology data, _please_ do not use the parallel functionality. You'll start to hit the macrostrat server at 40 or more QPS. We have not coordinated with the authors of that dataset to ensure they can handle the load. Please just run synchronously.

## Encoders

Encoders produce embeddings from raw data. It is key that these embeddings follow the file format:

```
<descriptive_directory_name>/<data_type {dem, sen, lit}>_<lat>_<lon>.npy
```

We currently provide embeddings for:

Sentinel2:

- gaussian RCF
- empirical RCF
- MOSAIKS
- flattened

DEM:

- gaussian RCF
- empirical RCF
- flattened

Lithology:

- SciBERT Encoding including description
- SciBERT Encoding excluding description

## Model

The model is entirely defined by the `model/config.yml`. The datasets are generified so that all that is required to use them is to provide a list of embedding directories.

The name of the experiment will be a contatenation of the embedding directories. This is the key that the dataset and evaluation scripts require.
