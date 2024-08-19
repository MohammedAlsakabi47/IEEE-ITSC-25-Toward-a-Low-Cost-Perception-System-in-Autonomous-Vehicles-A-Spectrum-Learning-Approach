<p align="center">
  <img src = "Figures/Carnegie_Mellon_University.png" width="60%">
</p>

# AAAI-25
#### Mohammed Alsakabi, Aidan Erickson, John M. Dolan, Ozan K. Tonguz

This repository contains the code, documentation, and examples corresponding to the paper *A Camera-Assisted Radar 3D Scene Generation via Spectrum Learning** by Alsakabi et. al, submitted to AAAI-25.

<div align="center">
<p float="center">
<img src="Figures/SoTA_vs_Proposed.gif" alt="Proposed method" width="600"/>
<br />
<b>Example scene of our method compared to the current state-of-the-art (SOTA).</b>
</p>
</div>

### Files

`Example Frames/`: the 10 random example frames including camera images, radar point clouds, lidar point clouds, radar spectrum, lidar spectrum, camera spectrum, and camera semantic segmentations

`Algorithm1.m`: Algorithm 1, labeled in the paper as Pixel Encoding and Spectrum Estimation, in MATLAB

`ResNet_Training.py`: code for training ResNet101

`semantic_segmentaion.py`: the code for Deeplab v3 semantic segmentation model

`unidirectional_chamfer_distance.py`: function of unidirectional chamfer distance

`visualization_examples.ipynb`: example visualization and calculation of UCD

### How to run

To obtain a visualization of our method, simply run `visualization_examples.ipynb` with the default settings. This will render a Unidirectional Chamfer Distance measure as well as the visualizations.
