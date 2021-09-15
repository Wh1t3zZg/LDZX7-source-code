# MSc project - Learned 3D representations: point clouds, depth maps and holograms  
**author: LDZX7**  
**supervisor: Kaan Akşit**  


# Code Description
[read_save.ipynb](read_save.ipynb): this notebook provides various ways of reading and saving point clouds, 3D meshes with help of multiple packages.   

[compression_translation.ipynb](compression_translation.ipynb): this notebook provides: 1. 3D mesh(standford bunny, and reconstructed Owlii dancer mesh version) compression by Google Draco; 2. translation methods of point-cloud-to-mesh or mesh-to-point-cloud. The implementations come with their results such as: visualization, encode and decode times, translation times.  

[marching_cube.py](marching_cube.py): this code uses marching cube algorithm to generate mesh from signed distance function for visualization or further uses.    

[demo.ipynb](demo.ipynb): this notebook shows point cloud compression process implementation mentioned in the report, including compression rate in terms of bpp (bits per point), encode and decode time.   

[plane_detection.py](plane_detection.py): detect the plane in point cloud by RANSAC.   

[multiplane.py](multiplane.py): this code turn point cloud into its multiplane representation and layer-based hologram encoding method. And a hologram class including error diffusion. And a point cloud transforom class (e.g. rotation, translation, scale). And some useful utility functions.   

[multiplane_demo.ipynb](multiplane_demo.ipynb): demo run for functions in multiplane.py  

[data_prepare.ipynb](data_prepare.ipynb): codes for preparing the custom dataset   

[utils.py](utils.py): defining some metric for deep learning model selection and comparison. And linear interpolation function for recovering the gaps of extracted point cloud.   

[network.py](network.py): the codes for defining vanilla u-net and attention r2u-net. And parameter initialization function.       

[main.py](main.py): main file for training/validating/testing the deep neural network  
