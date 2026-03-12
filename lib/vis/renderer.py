import os
import pickle
import cv2
import torch
import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation

from .tools import get_colors, checkerboard_geometry

# ---------------------------------------------------------------------------
# Part-segmentation helpers
# ---------------------------------------------------------------------------
_PART_SEG_PATH = os.path.join(os.path.dirname(__file__), 'smpl_partSegmentation_mapping.pkl')

# Color map: SMPL part ID → RGB (0-1 range)
# Parts that share the same colour produce left/right symmetry highlighting.
_PART_COLOR_MAP = {
    1:  [1.0, 0.0, 1.0],   # L_Hip      – Magenta
    2:  [0.0, 1.0, 1.0],   # R_Hip      – Cyan
    4:  [1.0, 0.0, 1.0],   # L_Knee     – Magenta  (same as L_Hip)
    5:  [0.0, 1.0, 1.0],   # R_Knee     – Cyan     (same as R_Hip)
    16: [0.0, 1.0, 0.0],   # L_Shoulder – Lime
    17: [1.0, 0.0, 0.0],   # R_Shoulder – Red
    18: [0.0, 1.0, 0.0],   # L_Elbow    – Lime     (same as L_Shoulder)
    19: [1.0, 0.0, 0.0],   # R_Elbow    – Red      (same as R_Shoulder)
}
_BODY_DEFAULT_COLOR = [0.8, 0.75, 0.7]  # neutral beige for the torso/head


def _load_part_mapping(path=_PART_SEG_PATH):
    """Load SMPL vertex→part-ID mapping from the pkl file.
    Returns a numpy array of length N_verts, or None on failure."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if 'smpl_index' in data:
            return np.array(data['smpl_index'])
        return None
    except Exception as e:
        print(f'[Renderer] Could not load part segmentation: {e}')
        return None


# Load once at import time so every Renderer instance can share it.
_PART_MAPPING = _load_part_mapping()


def create_vertex_colors(num_vertices, part_mapping=_PART_MAPPING,
                         color_map=_PART_COLOR_MAP,
                         default_color=_BODY_DEFAULT_COLOR):
    """Return (num_vertices, 3) float32 array with per-body-part RGB colours."""
    colors = np.ones((num_vertices, 3), dtype=np.float32) * np.array(default_color, dtype=np.float32)
    if part_mapping is not None and len(part_mapping) == num_vertices:
        for part_id, color in color_map.items():
            indices = np.where(part_mapping == part_id)[0]
            if len(indices) > 0:
                colors[indices] = np.array(color, dtype=np.float32)
    return colors


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    roi_image[mask] = image[mask]
    out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype
    
    K = torch.zeros((K_org.shape[0], 4, 4)
    ).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1
    
    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = (right - left)
    height = (bottom - top)

    new_left = torch.clamp(cx - width/2 * scaleFactor, min=0, max=img_w-1)
    new_right = torch.clamp(cx + width/2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h-1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(),
                        new_right.detach(), new_bottom.detach())).int().float().T
    
    return bbox


class Renderer():
    def __init__(self, width, height, focal_length, device, faces=None):

        self.width = width
        self.height = height
        self.focal_length = focal_length

        self.device = device
        if faces is not None:
            self.faces = torch.from_numpy(
                (faces).astype('int')
            ).unsqueeze(0).to(self.device)

        self.initialize_camera_params()
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                    blur_radius=1e-5),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            )
        )

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False)


    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        # Extrinsics
        self.R = torch.diag(
            torch.tensor([1, 1, 1])
        ).float().to(self.device).unsqueeze(0)

        self.T = torch.tensor(
            [0, 0, 0]
        ).unsqueeze(0).float().to(self.device)

        # Intrinsics
        self.K = torch.tensor(
            [[self.focal_length, 0, self.width/2],
            [0, self.focal_length, self.height/2],
            [0, 0, 1]]
        ).unsqueeze(0).float().to(self.device)
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()
        
        
    def set_ground(self, length, center_x, center_z):
        device = self.device
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y"))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]


    def update_bbox(self, x3d, scale=2.0, mask=None):
        """ Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(self,):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, background, colors=[0.8, 0.8, 0.8],
                    vertex_colors=None):
        """Render a single mesh onto *background*.

        Args:
            vertices:      (N_verts, 3) torch tensor in camera space.
            background:    HxWx3 numpy image.
            colors:        Flat RGB list [r, g, b] used when *vertex_colors* is None.
            vertex_colors: Optional (N_verts, 3) numpy array with per-vertex RGB
                           values in [0, 1].  When provided, each vertex gets its
                           own colour (enabling body-part colouring).
        """
        self.update_bbox(vertices[::50], scale=1.2)
        vertices = vertices.unsqueeze(0)          # (1, N_verts, 3)

        if vertex_colors is not None:
            # Per-vertex colouring path
            verts_features = torch.tensor(
                vertex_colors, dtype=vertices.dtype, device=vertices.device
            ).unsqueeze(0)                        # (1, N_verts, 3)
            flat_color = _BODY_DEFAULT_COLOR      # used only for Materials
        else:
            # Flat single-colour path (original behaviour)
            if colors[0] > 1:
                colors = [c / 255. for c in colors]
            verts_features = torch.tensor(colors).reshape(1, 1, 3).to(
                device=vertices.device, dtype=vertices.dtype
            ).repeat(1, vertices.shape[1], 1)    # (1, N_verts, 3)
            flat_color = colors

        textures = TexturesVertex(verts_features=verts_features)

        mesh = Meshes(
            verts=vertices,
            faces=self.faces,
            textures=textures,
        )

        materials = Materials(
            device=self.device,
            specular_color=(flat_color,),
            shininess=0,
        )

        results = torch.flip(
            self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
            [1, 2]
        )
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
        self.reset_bbox()
        return image
    
    
    def render_with_ground(self, verts, faces, colors, cameras, lights):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        
        # (B, V, 3), (B, F, 3), (B, V, 3)
        verts, faces, colors = prep_shared_geometry(verts, faces, colors)
        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        verts = list(torch.unbind(verts, dim=0)) + [gv]
        faces = list(torch.unbind(faces, dim=0)) + [gf]
        colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]]
        mesh = create_meshes(verts, faces, colors)

        materials = Materials(
            device=self.device,
            shininess=0
        )
        
        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            
        return image
    
    
def prep_shared_geometry(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (F, 3)
    :param colors (B, 4)
    """
    B, V, _ = verts.shape
    F, _ = faces.shape
    colors = colors.unsqueeze(1).expand(B, V, -1)[..., :3]
    faces = faces.unsqueeze(0).expand(B, F, -1)
    return verts, faces, colors


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(verts, device, distance=5, position=(-5.0, 5.0, 0.0)):
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1)
    
    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions
    
    rotation = look_at_rotation(positions, targets, ).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)
    
    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


def _get_global_cameras(verts, device, min_distance=3, chunk_size=100):
    
    # split into smaller chunks to visualize
    start_idxs = list(range(0, len(verts), chunk_size))
    end_idxs = [min(start_idx + chunk_size, len(verts)) for start_idx in start_idxs]
    
    Rs, Ts = [], []
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        vert = verts[start_idx:end_idx].clone()
        import pdb; pdb.set_trace()