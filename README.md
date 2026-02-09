# Personalized Value-for-Money (VFM) Recommender for Airbnb

This repository contains the full implementation of a personalized Value-for-Money (VFM) recommendation system for Airbnb listings.

The project integrates:
- Airbnb listing data (price, ratings, amenities)
- Neighborhood context derived from OpenStreetMap (OSM) POIs
- User-controlled preference weights for personalization
- Multiple VFM scoring functions and evaluation metrics

This work was developed as part of the **Data Collection & Management Lab (00940290)** final project, Technion – Winter 2026.

---

## Repository Structure



---

## How to Run the Project

1. Clone the repository.
2. Open the notebooks in **Databricks** (recommended) or a local Spark environment.
3. Run notebooks **in order (Stage 1 -> Stage 2 -> Main notebook)**.
4. Raw data loading paths are documented in the notebooks.

> **Note:** Raw datasets are **not included** in this repository due to size and licensing constraints.

---

## Data Availability

- **Raw OSM data (PBF, ~11GB)** and processed POI datasets are stored in an **Azure Blob Container** (link provided in the final report).
- Airbnb data was provided by the course staff and is **not redistributed**.

---

## Video Demonstration

A short system overview video (user perspective) is linked in the final report PDF.

---

## Authors

- Gil Yashayev  
- Yaniv Steiner
- Yehoraz Ben Yehuda
