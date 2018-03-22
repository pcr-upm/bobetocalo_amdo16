# Head-Pose Estimation using a Random Forest

We provide C++ code in order to replicate the head-pose experiments in our paper https://link.springer.com/chapter/10.1007/978-3-319-41778-3_3

If you use this code for your own research, you must reference our AMDO paper:

```
Head-Pose Estimation In-the-Wild Using a Random Forest
Roberto Valle, José M. Buenaposada, Antonio Valdés, Luis Baumela.
Conference on Articulated Motion and Deformable Objects, 9th International Conference, AMDO 2016, pp. 24-33, Palma de Mallorca, Spain, July 13-15, 2016.
```

#### Requisites
- faces_framework https://github.com/bobetocalo/faces_framework

#### Installation
This repository must be located inside the following directory:
```
faces_framework
    └── headpose 
        └── bobetocalo_amdo16
```
You need to have a C++ compiler (supporting C++11):
```
> mkdir release
> cd release
> cmake ..
> make -j$(nproc)
> cd ..
```
#### Usage
```
> ./release/face_headpose_bobetocalo_amdo16_test
```
