# Adventure is Out There!

Here is the readme for our Geospatial Machine Learning project GEOClimb. It's goal is to predict unexplore rock climbing areas throughout the world.

## Getting Started:

1. Download Conda
2. Create Conda environment: `conda env create -f environment.yml`
3. Initialize environment: `conda activate geo_climb`

## Working:

- If you install any packages, make sure to `conda env export --name geo_climb --no-builds > environment.yml`
- If you pull and packages need to be installed, pull them from the conda environment.yml `conda env update --file environment.yml --prune`

## Dataset

The dataset can be downloaded from this link: https://drive.google.com/drive/folders/1lItS74OOocI-ppHx5q0SmEKHP-G_-gmg?usp=drive_link

It contains sentinel2 RGB images, DEM, and lithology information for 12,742 climbing (labeled) locations and 12,692 unlabeled locations.

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
