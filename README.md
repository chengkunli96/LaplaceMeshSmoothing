# LaplaceMeshSmoothing

![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![GitHub stars](https://img.shields.io/github/stars/chengkunli96/LaplaceMeshSmoothing.svg?style=social)](https://github.com/chengkunli96/LaplaceMeshSmoothing/stargazers)

This repository is about Laplacian filtering in the context of mesh denoising. The main contributions are
* compute the mean curvature of a mesh by Uniform Laplace and Cotangent Laplace respectively.
* compute mesh's Gaussian curvature.
* reconstruct the mesh with k eigenvectors.
* implementation of explicit and implicit Laplacian mesh smoothing.

More details can be found in this [report](https://github.com/mremilien/LaplaceMeshSmoothing/blob/main/doc/report.pdf).

## Requirements 
This code has been test under python 3.6.12, and the libraries I used as following,
* open3d==0.12.0
* trimesh==3.9.1
* numpy==1.19.2
* matplotlib==3.3.2
* argparse==1.4.0
* scipy==1.5.4

## File structure
* `data` the data for testing our algorithm.
* `docs` the report of my experiment.
* `src` includes all of the source code.
   * `part*.py` in this files, I've done several experiment (check the report, you'll understand what I've done).
   * `tools` floder is a package I build to implement core algorithm
        * `LaplaceOperator.py` - the core code
        * `tools.py` - some tool function, like show mesh in open3d gui
 
## Easy using
Fistly, you should import the nessary module.
``` python
import trimesh
import tools.LaplaceOperator  # our python package
```
For computing mean curvature matrix H and Gaussian curvature K of the mesh, you can do:
``` python
tm = trimesh.load(mesh_file_path)  # load mesh
Obj = LaplaceOperator.LaplaceOperator(tm)
H = Obj.getMeanCurvature(mode='uniform')  # also, you can use 'cotan'
K = Obj.getGaussianCurvature()
```
For spectral analysis which reconstruct the mesh with k eigenvectors, you can do:
``` python
newtm = Obj.decompose(k)  # k is the num of eigenvectors you want
```
For mesh smoothing, do following
``` python
# lr is the step size for every iteration. maxIter is the number of iteration
# you can also change mode to 'explicit' to do smooth
newtm = Obj.smooth(lr=step_size, maxIter=iteration, mode='implicit')
```
 
## Experiment
You can read the code of my experiments to find out the usage with more detail.
  
The first experiment is about Uniform Laplace mean curvature and Gaussian curvature.
``` python
# Argument '--obj' is used to indicate the 3D object we use, and the mesh I use here can be found in data folder.
# and use'--mode mean' is to show the mean curvature.
python part1.py --obj lilium_s.obj --mode mean
python part1.py --obj plane.obj --mode mean
 
# Argument ‘--mode gaussian’ is to show the mean curvature.
python part1.py --obj plane.obj --mode gaussian
python part1.py --obj lilium_s.obj --mode gaussian
```
And then for the mean curvature by Cotangent laplace
``` python
python part3.py --obj plane.obj
python part3.py --obj lilium_s.obj
```
As for doing spectral analysis,
``` python
# Use '--k' to set the number of eigenvectors
pyhon part4.py --k 60
```
In terms of the explicit Laplacian mesh smoothing,
``` python
# '--step_size' is to set the step size lambda of iteration
# '--iteration' is to set the number of iteration
python part5.py --obj fandisk_ns.obj --step_size 1e-5 --iteration 50
```
To see the performace of the implicit Laplacian mesh smoothing,
``` python
# '--step_size' is to set the step size lambda of iteration
# '--iteration' is to set the number of iteration
python part5.py --obj fandisk_ns.obj --step_size 1e-5 --iteration 50
```

## Some samples
For spectral analysis, if you run `part4.py`, finnaly you will get as following.

![spectral analysis](https://user-images.githubusercontent.com/22772725/112777180-6a8b9400-9039-11eb-8096-3589c6f0b031.png)

Besides, the picture below shows the results of the explicit Laplacian mesh smoothing.

![smoothing by explicit laplace](https://user-images.githubusercontent.com/22772725/112777992-6e201a80-903b-11eb-8df1-ca1ce36482c6.png)


