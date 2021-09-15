# MSc project - Learned 3D representations: point clouds, depth maps and holograms  
**author: LDZX7**  
**supervisor: Kaan Akşit**  


# Code Description
[read_save.ipynb](read_save.ipynb): this notebook provides various ways of reading and saving point clouds, 3D meshes with help of multiple packages.    
[compression.ipynb](compression.ipynb): this notebook provides: 1. 3D mesh(standford bunny, and reconstructed Owlii dancer mesh version) compression by Google Draco; 2. translation methods of point-cloud-to-mesh or mesh-to-point-cloud. The implementations come with their results such as: visualization, encode and decode times, translation times. The compressed *.draco* file is uploaded to *dancer* folder.     
[marching_cube.py](marching_cube.py): this code uses marching cube algorithm to generate mesh from signed distance function for visualization or further uses.    
[demo.ipynb](demo.ipynb): this notebook shows point cloud compression process implementation mentioned in the report, including compression rate in terms of bpp (bits per point), encode and decode time. The compressed files are contained in the *compressed* folder in the *dancer* folder.    
[multiplane.py](multiplane.py): this code turn point cloud into its multiplane representation.   
[multiplane_demo.ipynb](multiplane_demo.ipynb): demo run for functions in multiplane.py  
[model.py](model.py): the codes for defining vanilla u-net and attention r2u-net.   
[data_prepare.ipynb](data_prepare.ipynb): codes for preparing the custom dataset  
[reconstruction.py](reconstruction.py): codes for recovering the extracted point cloud from digital hologram by linear interpolation.   
[reconstruction_demo.ipynb](reconstruction_demo.ipynb): demo run for functions in reconstruction.py  
[network.py](network.py): defining neural network structure  
[utils.py](utils.py): defining some useful metric for deep learning model selection and comparison  
[main.py](main.py): main file for training deep neural network  
