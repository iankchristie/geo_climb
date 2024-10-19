Adventure is Out There!

Here is the readme for our Geospatial Machine Learning project GEOClimb. It's goal is to predict unexplore rock climbing areas throughout the world.

Getting Started:

1. Download Conda
2. Create Conda environment: `conda env create -f environment.yml`
3. Initialize environment: `conda activate geo_climb`
   _. Create your earth engine account
   _. Authenticate with earth engine `earthengine authenticate`
   \*.

Working:

- If you install any packages, make sure to `conda env export --name geo_climb --no-builds > environment.yml`
- If you pull and packages need to be installed, pull them from the conda environment.yml `conda env update --file environment.yml --prune`
