import torch
from pytorch_sdf import BVH
from typing import NewType
from pytorch_sdf import solid_angles_cuda
try:
    import pytorch3d
    from pytorch3d.ops import knn_points
except ImportError:
    print("pytorch3d not found")

Tensor = NewType('Tensor', torch.Tensor)


class SDF:
    def __init__(self, distance_method: str = "bvh"):
        """
        Initialize the SDF class.
        """

        self.mesh_bvh = BVH()

        if distance_method == "bvh":
            self.distance_function = self.bvh_distance
        elif distance_method == "knn":
            self.distance_function = self.knn_distance
        elif distance_method == "cdist":
            self.distance_function = self.cdist_distance
        else:
            raise ValueError("Invalid distance method")


    def fast_winding_numbers(self, query_pts: Tensor, triangles: Tensor, thresh: float = 1e-8) -> Tensor:
        """
        Compute the winding numbers of the triangles with respect to the query_pts.

        :param query_pts: Tensor of shape (B, N, 3)
        :param triangles: Tensor of shape (B, M, 3, 3)
        :param thresh: Threshold for the winding number computation
        :return: Tensor of shape (B, N)
        """
        solid_angles = solid_angles_cuda.solid_angles_cuda(query_pts, triangles, thresh)
        return solid_angles.sum(dim=-1) / (4 * torch.pi)


    def bvh_distance(self, query_pts: Tensor, vertices: Tensor, triangles: Tensor) -> Tensor:
        """
        Compute the distance of the query_pts to the mesh defined by the triangles using the BVH.

        :param query_pts: Tensor of shape (B, N, 3)
        :param vertices: Tensor of shape (B, V, 3) Not used
        :param triangles: Tensor of shape (B, M, 3, 3)
        :return: distance of shape (B, N) and distance threshold where < min_dist_threshold is considered inside
        """

        min_dist, _, _, _ = self.mesh_bvh(triangles.contiguous(), query_pts.contiguous())
        # min_dist is squared distance
        min_dist = torch.sqrt(torch.relu(min_dist))
        min_dist_threshold = 0.
        return min_dist, min_dist_threshold


    def knn_distance(self, query_pts: Tensor, vertices: Tensor, triangles: Tensor = None) -> Tensor:
        """
        Compute the distance of the query_pts to the vertices using the k-nearest neighbors.

        :param query_pts: Tensor of shape (B, N, 3)
        :param vertices: Tensor of shape (B, V, 3)
        :param triangles: Tensor of shape (B, M, 3, 3) Not used
        :return: distance of shape (B, N) and distance threshold where < min_dist_threshold is considered inside
        """

        dist, _, _ = knn_points(query_pts, vertices, K=1,)
        min_dist = torch.sqrt(torch.relu(dist.squeeze(-1)))
        min_dist_threshold = 0.

        return min_dist, min_dist_threshold


    def cdist_distance(self, query_pts: Tensor, vertices: Tensor, triangles: Tensor = None) -> Tensor:
        """
        Compute the distance of the query_pts to the vertices using the cdist function from pytorch.
        It is fast, but has numerical precision issue https://github.com/pytorch/pytorch/issues/42479

        :param query_pts: Tensor of shape (B, N, 3)
        :param vertices: Tensor of shape (B, V, 3)
        :param triangles: Tensor of shape (B, M, 3, 3) Not used
        :return: distance of shape (B, N) and distance threshold where < min_dist_threshold is considered inside
        """

        dist = torch.cdist(query_pts, vertices,)
        min_dist, _ = torch.min(dist, dim=-1)
        # min_dist_threshold is set to -0.0008 due to numerical precision issues with cdist
        min_dist_threshold = -0.0008

        return min_dist, min_dist_threshold


    def sdf_with_winding_numbers(self, query_pts: Tensor, triangles: Tensor, vertices: Tensor) -> Tensor:
        """
        Compute the signed distance function of the query_pts to the mesh defined by the triangles.

        :param query_pts: Tensor of shape (B, N, 3)
        :param triangles: Tensor of shape (B, M, 3, 3)
        :param vertices: Tensor of shape (B, V, 3)
        :return: distance of shape (B, N) and inside bool of shape (B, N)
        """

        with torch.no_grad():
            winding_number = self.fast_winding_numbers(query_pts, triangles)
            # 1.-2*winding_number is usually used, but might add noise when multiplied by the distance
            sign = torch.where(winding_number < .5, 1, -1)
        min_dist, min_dist_threshold = self.distance_function(query_pts, vertices, triangles)

        min_dist = sign * min_dist
        inside = min_dist < min_dist_threshold

        return min_dist, inside
