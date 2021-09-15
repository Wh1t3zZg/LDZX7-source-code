import torch
from multiplane import *


def post_process(grid_list,
                 num_points=500000):
    """receive the outputs through NN, and return the reconstructed point cloud"""

    resolution = grid_list[0].shape
    dz = np.sum(resolution) / ( 2 * (len(grid_list)-1) )

    interp = np.arange(1, dz, 10)
    interp_len = len(interp)

    pcd = np.zeros((num_points, 3))
    dist = 0

    for i, grid in enumerate(grid_list):
        if i == 0:
            idx = np.where( grid > 0 )

            pcd = np.concatenate( (idx[1].reshape(-1,1), idx[0].reshape(-1,1), dist*np.ones((len(idx[0]),1))),
                                  axis=1 )

            grid[np.where(grid==0)] = np.inf

        else:
            idx = np.where( grid > 0 )

            coordn = np.concatenate( (idx[1].reshape(-1,1), idx[0].reshape(-1,1), dist*np.ones((len(idx[0]), 1))),
                                     axis=1 )

            # cover the gap between layers by interpolating the overlapped pixels
            overlap = np.where( grid == grid_list[i-1] )
            overlap_len = len( overlap[0] )

            overlap_coordn = np.concatenate( ( np.tile(overlap[1], [interp_len,1]).reshape(-1,1),
                                              np.tile(overlap[0], [interp_len,1]).reshape(-1,1),
                                              np.repeat(interp+dist, overlap_len). reshape(-1,1) ),
                                             axis=1 )

            pcd = np.concatenate((pcd, coordn, overlap_coordn),
                                 axis=0)

            grid[np.where(grid == 0)] = np.inf

            dist += dz

    pcd = normalize(pcd)

    # point cloud to 3d triangle mesh
    # pcd = prepareO3D(pcd)

    # pcd.estimate_normals()

    # poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
    #                                                                          depth=10,
    #                                                                          width=0,
    #                                                                          scale=1.1,
    #                                                                          linear_fit=False)[0]

    # cropping
    # bbox = pcd.get_axis_aligned_bounding_box()
    # mesh_crop = poisson_mesh.crop(bbox)

    # up-sampling
    # pcd = mesh_crop.sample_points_uniformly(number_of_points=num_points)

    return pcd   # np.array(pcd.points, dtype=np.float64)


def tensor2img(x):
    img = (x[:,0,:,:]>x[:,1,:,:]).float()
    img = img*255
    
    return img


def num_params(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
        

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc


def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # print(torch.sum(SR),torch.sum(GT))

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE


def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP


def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC


def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1


def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS


def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC
        