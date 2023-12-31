# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

import numpy as np
import paddle
from paddle.utils.download import get_weights_path_from_url
from skimage import color, io

from .models import FAN, ResNetDepth
from .utils import *


def load_url(url):
    cache_file = get_weights_path_from_url(url)
    return paddle.load(cache_file)


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """

    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


models_urls = {
    # '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
    "2DFAN-4": "https://test1model.oss-cn-beijing.aliyuncs.com/diffusion-rig/paddle_2DFAN4-11f355bf06.pth.tar",
    # '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
    "3DFAN-4": "https://test1model.oss-cn-beijing.aliyuncs.com/diffusion-rig/paddle_3DFAN4-7835d9f11d.pth.tar",
    "depth": "https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar",
}


class FaceAlignment:
    def __init__(
        self,
        landmarks_type,
        network_size=NetworkSize.LARGE,
        device="cuda",
        flip_input=False,
        face_detector="sfd",
        verbose=False,
    ):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        # if 'cuda' in device:
        #     paddle.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__(
            "face_alignment_paddle.detection." + face_detector,
            globals(),
            locals(),
            [face_detector],
            0,
        )
        self.face_detector = face_detector_module.FaceDetector(
            device=device, verbose=verbose
        )

        # Initialise the face alignemnt networks
        self.face_alignment_net = FAN(network_size)
        if landmarks_type == LandmarksType._2D:
            network_name = "2DFAN-" + str(network_size)
        else:
            network_name = "3DFAN-" + str(network_size)

        fan_weights = load_url(
            models_urls[network_name],
            #    map_location=lambda storage, loc: storage
        )
        self.face_alignment_net.set_state_dict(fan_weights)

        self.face_alignment_net.to(paddle.CUDAPlace(0))
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()

            depth_weights = load_url(
                models_urls["depth"], map_location=lambda storage, loc: storage
            )
            depth_dict = {
                k.replace("module.", ""): v
                for k, v in depth_weights["state_dict"].items()
            }
            self.depth_prediciton_net.load_state_dict(depth_dict)

            self.depth_prediciton_net.to(device)
            self.depth_prediciton_net.eval()

    def get_landmarks(self, image_or_path, detected_faces=None):
        """Deprecated, please use get_landmarks_from_image

        Arguments:
            image_or_path {string or numpy.array or paddle.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        return self.get_landmarks_from_image(image_or_path, detected_faces)

    @paddle.no_grad()
    def get_landmarks_from_image(self, image_or_path, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or paddle.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        if isinstance(image_or_path, str):
            try:
                image = io.imread(image_or_path)
            except IOError:
                print("error opening file :: ", image_or_path)
                return None
        elif isinstance(image_or_path, paddle.Tensor):
            image = image_or_path.detach().cpu().numpy()
        else:
            image = image_or_path

        if image.ndim == 2:
            image = color.gray2rgb(image)
        elif image.ndim == 4:
            image = image[..., :3]

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_image(
                image[..., ::-1].copy()
            )

        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        landmarks = []
        for i, d in enumerate(detected_faces):
            center = paddle.to_tensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0],
                dtype="float32",
            )
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = paddle.to_tensor(inp.transpose((2, 0, 1))).astype("float32")

            inp = inp.cuda()
            inp = inp.divide(paddle.to_tensor(255.0)).unsqueeze(0)

            out = self.face_alignment_net(inp)[-1].detach()
            if self.flip_input:
                out += flip(
                    self.face_alignment_net(flip(inp))[-1].detach(), is_label=True
                )
            out = out.cpu()

            pts, pts_img = get_preds_fromhm(out, center, scale)
            pts, pts_img = pts.reshape((68, 2)) * 4, pts_img.reshape((68, 2))

            if self.landmarks_type == LandmarksType._3D:
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for i in range(68):
                    if pts[i, 0] > 0 and pts[i, 1] > 0:
                        heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)
                heatmaps = paddle.to_tensor(heatmaps).unsqueeze(0)

                heatmaps = heatmaps.to(self.device)
                depth_pred = (
                    self.depth_prediciton_net(paddle.concat((inp, heatmaps), 1))
                    .cpu()
                    .reshape((68, 1))
                )
                pts_img = paddle.concat(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1
                )

            landmarks.append(pts_img.numpy())

        return landmarks

    @paddle.no_grad()
    def get_landmarks_from_batch(self, image_batch, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image in a batch in parallel.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_batch {paddle.tensor} -- The input images batch

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_batch(image_batch)

        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        landmarks = []
        # A batch for each frame
        for i, faces in enumerate(detected_faces):
            landmark_set = []
            for face in faces:
                center = paddle.to_tensor(
                    [(face[2] + face[0]) / 2.0, (face[3] + face[1]) / 2.0],
                    dtype="float32",
                )

                center[1] = center[1] - (face[3] - face[1]) * 0.12
                scale = (
                    face[2] - face[0] + face[3] - face[1]
                ) / self.face_detector.reference_scale
                image = image_batch[i].cpu().numpy()

                image = image.transpose(1, 2, 0)

                inp = crop(image, center, scale)
                inp = paddle.to_tensor(inp.transpose((2, 0, 1))).float()

                inp = inp.to(self.device)
                inp.div_(255.0).unsqueeze_(0)

                out = self.face_alignment_net(inp)[-1].detach()
                if self.flip_input:
                    out += flip(
                        self.face_alignment_net(flip(inp))[-1].detach(), is_label=True
                    )  # patched inp_batch undefined variable error
                out = out.cpu()
                pts, pts_img = get_preds_fromhm(out, center, scale)

                # Added 3D landmark support
                if self.landmarks_type == LandmarksType._3D:
                    pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
                    heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                    for i in range(68):
                        if pts[i, 0] > 0:
                            heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)
                    heatmaps = paddle.to_tensor(heatmaps).unsqueeze_(0)

                    heatmaps = heatmaps.to(self.device)
                    depth_pred = (
                        self.depth_prediciton_net(paddle.cat((inp, heatmaps), 1))
                        .data.cpu()
                        .view(68, 1)
                    )
                    pts_img = paddle.cat(
                        (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1
                    )
                else:
                    pts, pts_img = pts.view(-1, 68, 2) * 4, pts_img.view(-1, 68, 2)
                landmark_set.append(pts_img.numpy())
            if 0 != len(landmark_set):
                landmark_set = np.concatenate(landmark_set, axis=0)
            landmarks.append(landmark_set)
        return landmarks

    def get_landmarks_from_directory(
        self, path, extensions=[".jpg", ".png"], recursive=True, show_progress_bar=True
    ):
        detected_faces = self.face_detector.detect_from_directory(
            path, extensions, recursive, show_progress_bar
        )

        predictions = {}
        for image_path, bounding_boxes in detected_faces.items():
            image = io.imread(image_path)
            preds = self.get_landmarks_from_image(image, bounding_boxes)
            predictions[image_path] = preds

        return predictions
