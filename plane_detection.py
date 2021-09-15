from multiplane import *


# plane detection
def planeDetect(pcd, threshold=0.01, init_n=3, iter=1000):
    """ single plane detection using RANSAC

    Params:
        pcd: N*3 point cloud
        threshold: distance threshold
        init_n: number of initial points to be considered inliers in each iteration
        iter: number of iteration

    Returns:
         w: 1*4 list of plane coefficients
         index: list, plane index of each point
    """
    PointCloud = prepareO3D(pcd)
    w, index = PointCloud.segment_plane(threshold, init_n, iter)

    return w, index


def multiplePlaneDetect(pcd, min_ratio=0.05, threshold=0.01, iterations=1000):
    """ multiple planes detection from given point clouds

    Params:
        pcd: input point cloud array
        min_ratio: The minimum left points ratio to end the Detection
        threshold: RANSAC threshold in (m)

    Returns:
        plane_list: plane equation coefficients and plane point index
    """
    N = len(pcd)
    target = pcd.copy()
    count = 0
    plane_list = []
    while count < (1 - min_ratio) * N:
        w, index = planeDetect(target, threshold=threshold, init_n=3, iter=iterations)

        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)

    return plane_list
