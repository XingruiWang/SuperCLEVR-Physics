# SuperCLEVR Physics

[![arXiv](https://img.shields.io/badge/arXiv-2406.00622-b31b1b.svg)](https://arxiv.org/abs/2406.00622) ![License](https://img.shields.io/github/license/XingruiWang/SuperCLEVR-Physics)


A dynamical 3D scene understanding dataset for Video Question Answering. The scenes are annotated with objects' (1) static properties (**shape**, **color**) and (2) 3D dynamical properties (**3D position**, **velocities**, **external forces**), and (3) physical properties (**mass**, **frictions**, **restitution**); and Collision Event (objects involved, frame). 

<img alt="SuperCLEVR Physics Videos" src="https://github.com/XingruiWang/SuperCLEVR-Physics/blob/master/imgs/merged_animated_grid.gif?raw=true">
<p align="center"><small>(Note: the color space is compressed for visualization)</small></p>

### Related works
- [SuperCLEVR](https://github.com/Lizw14/Super-CLEVR). Visual question answering (VQA) dataset for domain robustness in four factors: visual complexity, question redundancy, concept distribution, concept compositionality.

- [SuperCLEVR-3D](https://github.com/XingruiWang/superclevr-3D-question). A VQA dataset for 3D awareness scene understanding the objects from images including 3D poses, parts, and occlusions. 

## How to generate your own data

<details>

<summary>Setup Environment</summary>

### Python version

I use python version 3.10. The python version will affect the compatibility of bpy packages.

### Install Dependency

Our repo is build upon Kubric. Please use the following steps to install kubric packages. We modify the original package for controlling more dynamical properties. 

```
pip install -r requirements.txt

```

### Install bpy

This is the python package for blender software. The bpy is now able to be installed from pip

```
pip install bpy==3.5
```
If 3.5 is not applicable, 3.4 should also compatible to this repo.

</details>





<!--
## Video Question Answering

### 1. Factual questions

### 2. Predictive questions

### 3. Counterfactual questions
-->
