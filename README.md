<p align="center">
  <img src = "Figures/Carnegie_Mellon_University.png" width="60%">
</p>

# IEEE ITSC-25: Toward a Low-Cost Perception System in Autonomous Vehicles: A Spectrum Learning Approach
#### Mohammed Alsakabi, Aidan Erickson, John M. Dolan, Ozan K. Tonguz

This repository contains the code, documentation, and examples corresponding to the paper *Toward a Low-Cost Perception System in Autonomous Vehicles: A Spectrum Learning Approach* by Alsakabi et. al, submitted to IEEE ITSC-25.

<div align="center">
<p float="center">
<img src="Figures/scene2.gif" alt="Proposed method" width="600"/>
<br />
<b>Example scene of our method compared to the current state-of-the-art (SOTA).</b>
</p>
</div>

### Files

`Example Frames/`: the 10 random example frames including camera images, radar point clouds, lidar point clouds, radar spectrum, lidar spectrum, camera spectrum, and camera semantic segmentations

`Algorithm1.m`: Algorithm 1, labeled in the paper as Pixel Encoding and Spectrum Estimation, and correlation calculation for the provided example frames in MATLAB

`ResNet_Training.py`: code for training ResNet101

`semantic_segmentaion.py`: the code for Deeplab v3 semantic segmentation model

`unidirectional_chamfer_distance.py`: function of unidirectional chamfer distance

`visualization_and_UCD.ipynb`: example visualization and calculation of UCD

### How to run
To obtain a visualization of camera segmentations, radar depth map, camera spectrum and radar spectrum, simply rum Algorithm1.m as they are input and output of Algorithm 1 in the paper.

To obtain a general visualization of our method, simply run `visualization_and_UCD.ipynb` with the default settings.

To access the complete RaDelft dataset, refer to [RaDelft Dataset](https://data.4tu.nl/datasets/4e277430-e562-4a7a-adfe-30b58d9a5f0a) and this [repository](https://github.com/RaDelft/RaDelft-Dataset)
