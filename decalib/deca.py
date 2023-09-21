import sys
sys.path.append('/nfs/github/recurrent/out/utils')
import paddle_aux
import paddle
import os, sys
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from .utils.renderer import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .utils.tensor_cropper import transform_points
from .datasets import datasets
from .utils.config import cfg
False = True


class DECA(paddle.nn.Layer):

    def __init__(self, config=None, device='cuda'):
        super(DECA, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.place = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

    def _setup_renderer(self, model_cfg):
        set_rasterizer(self.cfg.rasterizer_type)
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.
            topology_path, uv_size=model_cfg.uv_size, rasterizer_type=self.
            cfg.rasterizer_type).to(self.place)
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.0
        mask = paddle.to_tensor(data=mask[:, :, (0)])[(None), (None), :, :]
        self.uv_face_eye_mask = paddle.nn.functional.interpolate(x=mask,
            size=[model_cfg.uv_size, model_cfg.uv_size]).to(self.place)
        mask = imread(model_cfg.face_mask_path).astype(np.float32) / 255.0
        mask = paddle.to_tensor(data=mask[:, :, (0)])[(None), (None), :, :]
        self.uv_face_mask = paddle.nn.functional.interpolate(x=mask, size=[
            model_cfg.uv_size, model_cfg.uv_size]).to(self.place)
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = paddle.to_tensor(data=fixed_dis).astype(dtype=
            'float32').to(self.place)
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32
            ) / 255.0
        mean_texture = paddle.to_tensor(data=mean_texture.transpose(2, 0, 1))[(
            None), :, :, :]
        self.mean_texture = paddle.nn.functional.interpolate(x=mean_texture,
            size=[model_cfg.uv_size, model_cfg.uv_size]).to(self.place)
        self.dense_template = np.load(model_cfg.dense_template_path,
            allow_pickle=True, encoding='latin1').item()

    def _create_model(self, model_cfg):
        self.n_param = (model_cfg.n_shape + model_cfg.n_tex + model_cfg.
            n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light)
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.
            n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.
            param_list}
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.place)
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.place)
        self.flame = FLAME(model_cfg).to(self.place)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.place)
        self.D_detail = Generator(latent_dim=self.n_detail + self.n_cond,
            out_channels=1, out_scale=model_cfg.max_z, sample_mode='bilinear'
            ).to(self.place)
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = paddle.load(path=model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint[
                'E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint[
                'E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint[
                'D_detail'])
        else:
            print(f'please check model path: {model_path}')
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

    def decompose_code(self, code, num_dict):
        """ Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        """
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape((code_dict[key].
                    shape[0], 9, 3))
        return code_dict

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        """ Convert displacement map into detail normal map
        """
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
        uv_z = uv_z * self.uv_face_eye_mask
        uv_detail_vertices = (uv_coarse_vertices + uv_z * uv_coarse_normals +
            self.fixed_uv_dis[(None), (None), :, :] * uv_coarse_normals.
            detach())
        dense_vertices = uv_detail_vertices.transpose(perm=[0, 2, 3, 1]
            ).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render
            .dense_faces.expand(shape=[batch_size, -1, -1]))
        uv_detail_normals = uv_detail_normals.reshape([batch_size,
            uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]
            ).transpose(perm=[0, 3, 1, 2])
        uv_detail_normals = (uv_detail_normals * self.uv_face_eye_mask + 
            uv_coarse_normals * (1.0 - self.uv_face_eye_mask))
        return uv_detail_normals

    def visofp(self, normals):
        """ visibility of keypoints, based on the normal direction
        """
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:, :, 2:] < 0.1).astype(dtype='float32')
        return vis68

    def encode(self, images, use_detail=True):
        if use_detail:
            with paddle.no_grad():
                parameters = self.E_flame(images)
        else:
            parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        if use_detail:
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode
        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:, 3:].clone()
            posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict

    def decode(self, codedict, rendering=True, iddict=None, vis_lmk=True,
        return_vis=True, use_detail=True, render_orig=False, original_image
        =None, tform=None, add_light=True, th=0, align_ffhq=False,
        return_ffhq_center=False, ffhq_center=None, light_type='point',
        render_norm=False):
        images = codedict['images']
        batch_size = images.shape[0]
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict[
            'shape'], expression_params=codedict['exp'], pose_params=
            codedict['pose'])
        if align_ffhq and ffhq_center is not None or return_ffhq_center:
            lm_eye_left = landmarks2d[:, 36:42]
            lm_eye_right = landmarks2d[:, 42:48]
            lm_mouth_outer = landmarks2d[:, 48:60]
            eye_left = paddle.mean(x=lm_eye_left, axis=1)
            eye_right = paddle.mean(x=lm_eye_right, axis=1)
            eye_avg = (eye_left + eye_right) * 0.5
            mouth_left = lm_mouth_outer[:, (0)]
            mouth_right = lm_mouth_outer[:, (6)]
            mouth_avg = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg
            center = eye_avg + eye_to_mouth * 0.1
            if return_ffhq_center:
                return center
            if align_ffhq:
                delta = ffhq_center - center
                verts = verts + delta
        if self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:
            albedo = paddle.zeros(shape=[batch_size, 3, self.uv_size, self.
                uv_size])
        landmarks3d_world = landmarks3d.clone()
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,
            :, :2]
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]
        trans_verts = util.batch_orth_proj(verts, codedict['cam'])
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        opdict = {'verts': verts, 'trans_verts': trans_verts, 'landmarks2d':
            landmarks2d, 'landmarks3d': landmarks3d, 'landmarks3d_world':
            landmarks3d_world}
        if (return_vis and render_orig and original_image is not None and 
            tform is not None):
            points_scale = [self.image_size, self.image_size]
            _, _, h, w = original_image.shape
            trans_verts = transform_points(trans_verts, tform, points_scale,
                [h, w])
            landmarks2d = transform_points(landmarks2d, tform, points_scale,
                [h, w])
            landmarks3d = transform_points(landmarks3d, tform, points_scale,
                [h, w])
            images = original_image
        else:
            h, w = self.image_size, self.image_size
        if rendering:
            ops = self.render(verts, trans_verts, albedo, codedict['light'],
                h=h, w=w, add_light=add_light, th=th, light_type=light_type,
                render_norm=render_norm)
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']
            opdict['albedo_images'] = ops['albedo_images']
        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo
        return opdict, _

    def visualize(self, visdict, size=224, dim=2):
        """
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        """
        assert dim == 1 or dim == 2
        grids = {}
        for key in visdict:
            _, _, h, w = visdict[key].shape
            if dim == 2:
                new_h = size
                new_w = int(w * size / h)
            elif dim == 1:
                new_h = int(h * size / w)
                new_w = size
>>>            grids[key] = torchvision.utils.make_grid(paddle.nn.functional.
                interpolate(x=visdict[key], size=[new_h, new_w]).detach().cpu()
                )
        grid = paddle.concat(x=list(grids.values()), axis=dim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, (
            [2, 1, 0])]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8
            )
        return grid_image

    def save_obj(self, filename, opdict):
        """
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        """
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 +
            0.5)
        util.write_obj(filename, vertices, faces, texture=texture, uvcoords
            =uvcoords, uvfaces=uvfaces, normal_map=normal_map)
        texture = texture[:, :, ([2, 1, 0])]
        normals = opdict['normals'][i].cpu().numpy()
        displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze(
            )
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices
            , normals, faces, displacement_map, texture, self.dense_template)
        util.write_obj(filename.replace('.obj', '_detail.obj'),
            dense_vertices, dense_faces, colors=dense_colors,
            inverse_face_order=True)

    def run(self, imagepath, iscrop=True):
        """ An api for running deca given an image path
        """
        testdata = datasets.TestData(imagepath)
        images = testdata[0]['image'].to(self.place)[None, ...]
        codedict = self.encode(images)
        opdict, visdict = self.decode(codedict)
        return codedict, opdict, visdict

    def model_dict(self):
        return {'E_flame': self.E_flame.state_dict(), 'E_detail': self.
            E_detail.state_dict(), 'D_detail': self.D_detail.state_dict()}
