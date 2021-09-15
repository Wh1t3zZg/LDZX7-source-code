import numpy as np
import open3d as o3d
from odak.wave import wavenumber,propagate_beam


# utility
def preparePCD(xyz):
    """raw point cloud data to [N,3]-shape numpy array"""
    pcd = np.asarray(xyz)

    if pcd.shape[1] != 3:
        return pcd.T

    else:
        return pcd


def prepareO3D(pcd):
    """point cloud data to PointCloud instance in open3d"""
    PointCloud = o3d.geometry.PointCloud()
    PointCloud.points = o3d.utility.Vector3dVector(pcd)

    return PointCloud


def showGeometry(pcd, width=800, height=600):
    """visualize the given point cloud list in one plot, click to rotate, slide to zoom"""
    PointCloud = prepareO3D(pcd)
    o3d.visualization.draw_geometries([PointCloud], width=width, height=height)


def removeNoise(pcd, nb_neighbors=20, std_ratio=2.0):
    """remove the noise in point cloud by statistical noise removal method"""
    PointCloud = prepareO3D(pcd)
    result, _ = PointCloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    return np.asarray(result.points)


def downSample(pcd, voxel_size=0.003):
    """ down sample the point cloud"""
    p = prepareO3D(pcd).voxel_down_sample(voxel_size=voxel_size)

    return np.asarray(p.points)


# multiple-plane representation
def getBoundary(pcd):
    """get the largest range of given point cloud"""
    pcd = preparePCD(pcd)

    x_min, x_max = pcd[:, 0].min(), pcd[:, 0].max()
    y_min, y_max = pcd[:, 1].min(), pcd[:, 1].max()
    z_min, z_max = pcd[:, 2].min(), pcd[:, 2].max()

    return [x_min, x_max, y_min, y_max, z_min, z_max], pcd


def normalize(pcd, width=1, height=1, length=1):
    """make the size of point cloud model in meters, and move the point cloud model to """
    ranges, pcd = getBoundary(pcd)

    pcd -= np.array([ranges[0], ranges[2], ranges[-2]])
    pcd /= np.array([ ranges[1]-ranges[0], ranges[3]-ranges[2], ranges[-1]-ranges[-2] ])
    pcd *= np.array([width, height, length])

    return pcd


def padding(grid, margin_scale=2):
    """double the size of input array, i.e. from X*Y to 2X*2Y"""
    height, width = grid.shape[0], grid.shape[1]

    pad_x = int( np.ceil( width/margin_scale) )
    pad_y = int( np.ceil(height/margin_scale) )

    return np.pad(grid, ([pad_y, pad_y], [pad_x, pad_x]), constant_values=(0,0))


# modified version (without loops)
def multiplePlane(pcd,
                  N,
                  sort_by_plane=False):
    """change the point cloud to its multi-plane representation version

    Param:
        pcd: input point cloud data
        N: number of planes
        sort_by_plane: if True, return a list contained 2d location of each point on its plane, sorted by plane index

    Return:
         plane_list/pts_list: resulted plane/points list. For each point, [x, y], x&y are corresponding pixel location
         plane_idx: plane index of the corresponding point
         dz: float number represents depth difference between planes (in meters)
    """
    ranges, pcd = getBoundary(pcd)
    plane_pos = np.linspace(ranges[-2], ranges[-1], num=N)
    dz = plane_pos[1] - plane_pos[0]

    # normalization depth channel to get plane index of each point
    depth = pcd.copy()[:, -1]
    depth -= ranges[-2]
    depth /= ranges[-1] - ranges[-2]

    plane_idx = np.around(depth * (N-1)).astype(int)

    if not sort_by_plane:
        pts_list = pcd.copy()

        return pts_list, plane_idx

    else:
        plane_list = pcd.copy()[:,0:-1]

        # sort by depth
        sort_idx = np.argsort(plane_idx)
        plane_list = plane_list[sort_idx]

        # split points into different array by its plane index
        split_idx = np.where(np.diff(np.sort(plane_idx)) != 0)
        plane_list = np.split(plane_list, split_idx[0] + 1)

        return plane_list, dz


# map the points into pixel in the grid format (ndarrays), points are stored by the binary value in each pixel location
def multipleGrid(pcd,
                 N,
                 resolution,
                 size,
                 keep_ratio=True):
    """turn the point list of each plane into 2D binary image-like array

    Param:
        pcd: input point cloud data
        N: number of layers
        resolution (ncol, nrow): resolution of each grid layer
        size (hx, hy, hz): size of the point cloud model in meters
        keep_ratio: let the grid resolution match the size of the 3D scene

    Return:
        grid_list: list of binary width*height ndarray, which shows the point location
        dx: pixel pitch (in meters)
        dz: distance between layers (in meters)
    """
    height, width = resolution[0], resolution[1]
    hx, hy, hz = size[0], size[1], size[2]

    if keep_ratio:
        ratio = hx/hy
        resolution[0] = int( np.ceil(ratio * resolution[1]) )

    pcd = normalize(pcd, hx, hy, hz)

    # find pixel pitch dx
    dx = np.minimum(hx/width, hy/height)
    # separate layers, and get dz
    plane_list, dz = multiplePlane(pcd, N, sort_by_plane=True)

    grid_list = []
    for plane in plane_list:
        grid = np.zeros(resolution)

        plane *= ( np.array([width-1,height-1]) / np.array([hx,hy]) )

        pts_idx = np.round(plane).astype(int)

        grid[ pts_idx[:,0].tolist(), pts_idx[:,1].tolist() ] = 1

        grid_list.append(grid)

    return grid_list    # , dx, dz


# overall implementation， fused multiPlane and multiGrid, solved some issues
def multiLayerDF3D(pcd,
                   N,
                   resolution,
                   keep_ratio=True,
                   amplitude=1):
    """depth-fused 3d + layer-slicing + grid mapping

    Param:
        pcd: input point cloud data
        N: number of planes
        resolution (ncol, nrow): resolution of each grid layer
        keep_ratio: let the grid resolution match the size of the 3D scene

    Return:
        grid_list (ncol, nrow): list of binary ndarray, which shows the point location
    """


    # prepare
    ranges, pcd = getBoundary(pcd)

    hx, hy, hz = ranges[1]-ranges[0], ranges[3]-ranges[2], ranges[-1]-ranges[-2]
    pcd -= np.array([ranges[0], ranges[2], ranges[-2]])   # pcd = normalize(pcd, hx, hy, hz)

    plane_dist = np.linspace(ranges[-2], ranges[-1], num=N)   
    dz = plane_dist[1]-plane_dist[0]   # distance between layers

    height, width = resolution[0], resolution[1]
    if keep_ratio:
        ratio = hx/hy
        height = int( np.ceil(ratio * width) )


    # normalization depth channel to get plane index of each point
    depth = pcd.copy()[:, -1]

    depth -= ranges[-2]
    depth /= ranges[-1] - ranges[-2]

    plane_idx = np.floor(depth * (N-1)).astype(int)   # N-1 since

    # slicing the point cloud
    plane_list = pcd.copy()

    # sort by depth
    plane_list = plane_list[np.argsort(plane_idx)]
    # split points into different array according to their plane index
    split_idx = np.where(np.diff(np.sort(plane_idx)) != 0)

    plane_list = np.split(plane_list, split_idx[0] + 1)   # (N+1) * n * 3


    # encode df3d 2d-grid
    grid_list = np.zeros( (len(plane_list), height, width) )
    for i, plane in enumerate(plane_list):
        pixel_idx = plane.copy()[:, :-1]
        depth_fused = plane.copy()[:, -1]

        # df3d weights, note: w1 w2 are inversely prop to distance, since distance increases, intensity decreases
        w2 = (depth_fused - plane_dist[i]) / dz
        w1 = np.ones(w2.shape) - w2

        # normalizing to get pixel index
        pixel_idx *= ( np.array( [height-1,width-1] ) / np.array( [hx,hy] ) )
        pts_idx = np.round(pixel_idx).astype(int)
        grid_list[i][pts_idx[:,0].tolist(), pts_idx[:,1].tolist() ] += w1#**2

        if i != len(plane_list)-1:
            grid_list[i+1][pts_idx[:, 0].tolist(), pts_idx[:, 1].tolist()] += w2#**2
        else:
            pass  # print('ready')

    return list( grid_list * amplitude )
    

# simplest version
def occlusionMask(mask, new_plane):
    """update binary occlusion mask according to the point geometry in the next plane"""

    mask[ np.where(np.abs(new_plane) > 0) ] = 0

    return mask


def layer_based_hologram(pcd,
                         N,
                         resolution,
                         initial_dist=0.15,
                         layer_dist=0.004,
                         wavelength=650*1e-9,
                         pixel_pitch=8*1e-6,
                         propagation_type='Angular Spectrum',
                         df3d=True):
    """layer-based hologram simulation

    Param:
        pcd: input point cloud
        N: number of layers
        resolution: resolution of resulted hologram
        initial_dist: closest distance between point cloud and zero-plane
        layer_dist: distance between layers
        wavelength:
        pixel_pitch:
        propagation_type:

    Return:
        hologram: resulted hologram
        reconstruction: single reconstruction from -init_dist
    """
    if df3d:
        grid_list = multiLayerDF3D(pcd, N, resolution, keep_ratio=False)
    else:
        grid_list = multipleGrid(pcd, N, resolution,(1,1,1), keep_ratio=False)
    
    dz, dx = layer_dist, pixel_pitch
    
    k = wavenumber(wavelength)
    distance = initial_dist

    resolution = grid_list[0].shape   # since the resolution may be adjusted to keep ratio

    mask = np.ones(resolution)
    hologram = np.zeros(resolution, dtype=np.complex64)
    
    # padding
    hologram = padding(hologram)
    
    i = 0
    for grid in grid_list: #reversed(grid_list):
        grid = grid.astype(np.complex64)

        random_phase = np.pi * np.random.random(resolution)
        field = grid*np.cos(random_phase) + 1j*grid*np.sin(random_phase)
        field = grid

        field *= mask
        mask = occlusionMask(mask, grid)
        
        # padding
        field = padding(field)

        subhologram = propagate_beam(field,
                                     k,
                                     distance+i*dz,
                                     dx,
                                     wavelength,
                                     propagation_type)
        hologram += subhologram
        
        i += 1
    
    # reconstruction = propagate_beam(hologram,
    #                                 k,
    #                                 -distance,
    #                                 dx,
    #                                 wavelength,
    #                                 propagation_type)
    
    return hologram  # , reconstruction


class Hologram:
    def __init__(self, hologram):
        self.hologram = hologram

        self.amplitude = np.abs(self.hologram)
        self.phase = np.arctan( self.hologram.imag / self.hologram.real )

        self.POH = np.zeros(self.hologram.shape, dtype=type(self.hologram[0,0]))

    def getAmplitude(self, hologram):
        amplitude = np.abs(hologram)

        return amplitude

    def getPhase(self, hologram):
        phase = np.arctan(hologram.imag / hologram.real)

        return phase

    def errorDiffusion(self,
                       w_1=7/16,
                       w_2=3/16,
                       w_3=5/16,
                       w_4=1/16):
        """return phase-only version by bidirectional error diffusion"""

        H = self.hologram.copy()
        H_p = self.hologram / self.amplitude

        E = H - H_p   # error

        # padding extra one
        POH = np.zeros((self.hologram.shape[0]+1, self.hologram.shape[1]+2), dtype=type(self.hologram[0,0]))
        POH[0:-1,1:-1] = H

        # error diffusion
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):

                if i%2 != 0:
                    # even rows
                    j += 1

                    h = POH[i,j]
                    h_p = h / self.getAmplitude(h)

                    e = h - h_p

                    POH[i,   j+1]   +=   w_1 * E[i, j-1]
                    POH[i+1, j-1]   +=   w_2 * E[i, j-1]
                    POH[i+1, j]     +=   w_3 * E[i, j-1]
                    POH[i+1, j+1]   +=   w_4 * E[i, j-1]

                else:
                    # odd rows, direction reversed
                    j += 2   # 1 for reverse index, 1 for padded column
                    j *= -1

                    h = POH[i,j]
                    h_p = h / self.getAmplitude(h)

                    e = h - h_p

                    POH[i,   j-1]   +=   w_1 * E[i, j+1]
                    POH[i+1, j+1]   +=   w_2 * E[i, j+1]
                    POH[i+1, j]     +=   w_3 * E[i, j+1]
                    POH[i+1, j-1]   +=   w_4 * E[i, j+1]

        POH = POH[0:-1, 1:-1]
        self.POH = POH

        return self.POH


# transformation
class Transform:
    def __init__(self, pcd):
        """class instance for point cloud transformation"""

        # point cloud before transformation
        self.ori_pcd = preparePCD(pcd)
        # current point cloud (after transformation)
        self.pcd = preparePCD(self.ori_pcd)
        # overall transformation matrix (homogenous)
        self.T = np.eye(4)
        self.num_trans = 0

    def translation(self, dist):
        """translation
        Params:
            dist: [dx,dy,dz], translation distance in xyz-direction
        """
        T = np.eye(4)

        dx, dy, dz = dist[0], dist[1], dist[2]
        T[3, 0] = dx
        T[3, 1] = dy
        T[3, 2] = dz

        self.transform(T)

        return self.pcd

    def rotation(self, theta, axis):
        """rotate w.r.t. to the target axis
        Params:
            theta: rotation angle
            axis: rotation axis, 0=x-axis; 1=y-axis; 2=z-axis
        """
        T = np.eye(4)

        if axis == 0:
            T[1, 1] = np.cos(theta)
            T[2, 2] = np.cos(theta)
            T[2, 1] = -np.sin(theta)
            T[1, 2] = np.sin(theta)

        elif axis == 1:
            T[0, 0] = np.cos(theta)
            T[2, 2] = np.cos(theta)
            T[0, 2] = -np.sin(theta)
            T[2, 0] = np.sin(theta)

        elif axis == 2:
            T[0, 0] = np.cos(theta)
            T[1, 1] = np.cos(theta)
            T[0, 1] = -np.sin(theta)
            T[1, 0] = np.sin(theta)

        else:
            print('wrong input axis value, should be one from {0, 1, 2}')

        self.transform(T)

        return self.pcd

    def scale(self, factor):
        """scale up/down w.r.t. origin
        Params:
            factor: scale factor
        """
        T = np.eye(4)
        T[0:3, 0:3] *= factor

        self.transform(T)

        return self.pcd

    def transform(self, T):
        """change the point cloud to homogenous coordinate in order to apply transformation matrix, then change back

        Params:
            T: 4*4 transformation matrix
        """
        N = self.pcd.shape[0]
        self.num_trans += 1

        homo_pcd = np.ones((N, 4))
        homo_pcd[:, 0:3] = self.pcd
        homo_pcd = homo_pcd @ T
        resulted_pcd = homo_pcd[:, 0:3]

        self.T = self.T @ T
        self.pcd = resulted_pcd

    def restart(self):
        self.pcd = self.ori_pcd
        self.T = np.eye(4)
        self.num_trans = 0

    def planeTransform(self, coef, T):
        """transform the plane coefficient w.r.t. the transformation matrix

        Param:
            coeff: coefficients of general plane function, Ax+By+Cz+D=0, 1*4
            T: transformation matrix, 4*4

        Return:
             transformed coefficients, 1*4
        """
        return np.asarray(coef).reshape((1, 4)) @ np.linalg.inv(T)







