# GeoVector-Map-Digitization-AI-Agent
**Authors:** Nadeem Usmani | GSP Working Group

GeoVector is an AI Agent designed to automate the digitization of geological/thematic maps from georeferenced raster imagery into clean, topological vector shapefiles (GIS geometries).

**Now features 3-Channel Architecture:** Seamlessly supports both 1-channel Grayscale and 3-channel RGB Colored Maps for highest accuracy texture prediction.

## 🚀 Intelligent Workflow
This tool consists of a completely automated, multi-stage AI pipeline:
1. **Map Intelligence (LLM Integration):** Uses Claude 3 API to dynamically parse map legend components and map color tokens dynamically.
2. **Dynamic Tensor Generation:** It automatically scans the active Map Raster and asks if you wish to run deep-learning directly on rich RGB colors matrices, or strictly Black & White pixel data.
3. **Deep Learning Segmentation (PyTorch U-Net):** Given limited user-drawn bounding boxes, it generates balanced, jittered image texture chips in (C, H, W) shapes to train a specialized CNN architecture on the fly, sliding-window segmenting the whole map.
4. **Spatial Topology & Smoothing:** Leverages SciKit-Image and Shapely for topological noise-reduction (edge closing) and advanced coverage-simplification (Douglas-Peucker scaling) to snap AI pixel boundaries into perfectly rigid, human-like straight GIS vectors.
5. **GeoPandas Integration:** Automates output encoding, CRS-preservation, and outputs separate/exploded unit shapefiles directly imported into mapping standards.

## 📦 Setup Instructions
To run this pipeline, you must have Python 3.9+ installed natively or via Conda.

**1. Clone or Download the Tool**
Download the `GeoVector_Agent` zip folder and extract it to your working directory. Ensure your georeferenced map (e.g. `.tif` file) is in the root of the extracted folder.

**2. Install Dependencies**
Open your terminal (PowerShell, CMD, or bash) and run:
```bash
pip install -r requirements.txt
```

**3. Run the Agent**
Execute the pipeline:
```bash
python geovector_unet_os.py
```

## 🧠 Usage Walkthrough
1. **API Key Phase**: Optionally provide your Anthropic API Key for automated legend-scraping. If skipped, it asks you to manually define the rock units.
2. **Color Profile**: The code checks the map dimensions and automatically switches Matplotlib canvas and tensor arrays to 3-Channel mode if the map supports Color RGB.
3. **Training Context Phase**: A UI window will open. Click and drag boxes across the raster to highlight different formations for the active map class.
4. **Testing Phase**: Once segmentation completes, it validates against an isolated accuracy assessment geometry before proceeding.
5. **Export**: Shapefiles will be populated in `output_geovector/`. Load these into ArcGIS or QGIS!
