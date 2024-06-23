import json
import sys
import time
from calendar import c
from functools import partial
from unittest.mock import NonCallableMagicMock

import cv2
import h5py
import numpy as np
import tqdm
from pydub import AudioSegment


class SMCReader:

    def __init__(self, file_path):
        """Read SenseMocapFile endswith ".smc".

        Args:
            file_path (str):
                Path to an SMC file.
        """
        self.smc = h5py.File(file_path, "r")
        self.__calibration_dict__ = None
        self.actor_id = self.smc.attrs["actor_id"]
        self.performance_part = self.smc.attrs["performance_part"]
        self.capture_date = self.smc.attrs["capture_date"]
        self.actor_info = dict(
            age=self.smc.attrs["age"],
            color=self.smc.attrs["color"],
            gender=self.smc.attrs["gender"],
            height=self.smc.attrs["height"],
            weight=self.smc.attrs["weight"],
        )
        self.Camera_info = dict(
            num_device=self.smc["Camera"].attrs["num_device"],
            num_frame=self.smc["Camera"].attrs["num_frame"],
            resolution=self.smc["Camera"].attrs["resolution"],
        )

    ###info
    def get_actor_info(self):
        return self.actor_info

    def get_Camera_info(self):
        return self.Camera_info

    ### Calibration
    def get_Calibration_all(self):
        """Get calibration matrix of all cameras and save it in self

        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict(
                Camera_id : Matrix_type : value
              )
            Notice:
                Camera_id(str) in {'00' ... '59'}
                Matrix_type in ['D', 'K', 'RT']
        """
        if self.__calibration_dict__ is not None:
            return self.__calibration_dict__
        self.__calibration_dict__ = dict()
        for ci in self.smc["Calibration"].keys():
            self.__calibration_dict__.setdefault(ci, dict())
            for mt in ["D", "K", "RT"]:
                self.__calibration_dict__[ci][mt] = self.smc["Calibration"][ci][mt][()]
        return self.__calibration_dict__

    def get_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain camera by its type and id

        Args:
            Camera_id (int/str of a number):
                CameraID(str) in {'00' ... '60'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'RT']
        """
        Camera_id = str(Camera_id)
        assert Camera_id in self.smc["Calibration"].keys(), f"Invalid Camera_id {Camera_id}"
        rs = dict()
        for k in ["D", "K", "RT"]:
            rs[k] = self.smc["Calibration"][Camera_id][k][()]
        return rs

    ### RGB image
    def __read_color_from_bytes__(self, color_array):
        """Decode an RGB image from an encoded byte array."""
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    def get_img(self, Camera_id, Image_type, Frame_id=None, disable_tqdm=True):
        """Get image its Camera_id, Image_type and Frame_id

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in
                    {'00'...'59'}
            Image_type(str) in
                    {'Camera': ['color','mask']}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame'-1
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence
        Returns:
            a single img :
                'color': HWC(2048, 2448, 3) in bgr (uint8)
                'mask' : HW (2048, 2448) (uint8)
            multiple imgs :
                'color': NHWC(N, 2048, 2448, 3) in bgr (uint8)
                'mask' : NHW (N, 2048, 2448) (uint8)
        """
        Camera_id = str(Camera_id)
        assert Camera_id in self.smc["Camera"].keys(), f"Invalid Camera_id {Camera_id}"
        assert Image_type in self.smc["Camera"][Camera_id].keys(), f"Invalid Image_type {Image_type}"
        assert isinstance(Frame_id, (list, int, str, type(None))), f"Invalid Frame_id datatype {type(Frame_id)}"
        if isinstance(Frame_id, (str, int)):
            Frame_id = str(Frame_id)
            assert Frame_id in self.smc["Camera"][Camera_id][Image_type].keys(), f"Invalid Frame_id {Frame_id}"
            if Image_type in ["color", "mask"]:
                img_byte = self.smc["Camera"][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
            if Image_type == "mask":
                img_color = np.max(img_color, 2).astype(np.uint8)
            return img_color
        else:
            if Frame_id is None:
                Frame_id_list = sorted([int(l) for l in self.smc["Camera"][Camera_id][Image_type].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_img(Camera_id, Image_type, fi))
            return np.stack(rs, axis=0)

    def get_audio(self):
        """
        Get audio data.
        Returns:
            a dictionary of audio data consists of:
                audio_np_array: np.ndarray
                sample_rate: int
        """
        if "s" not in self.performance_part.split("_")[0]:
            print(f"no audio data in the performance part: {self.performance_part}")
            return None
        data = self.smc["Camera"]["00"]["audio"]
        return data

    def writemp3(self, f, sr, x, normalized=False):
        """numpy array to MP3"""
        channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(x * 2**15)
        else:
            y = np.int16(x)
        song = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
        song.export(f, format="mp3", bitrate="320k")

    ###Keypoints2d
    def get_Keypoints2d(self, Camera_id, Frame_id=None):
        """Get keypoint2D by its Camera_group, Camera_id and Frame_id
        PS: Not all the Camera_id/Frame_id have detected keypoints2d.

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in {18...32}
                    Not all the view have detection result, so the key will miss too when there are no lmk2d result
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence
        Returns:
            single lmk2d : (106, 2)
            multiple lmk2d : (N, 106, 2)
            if no data,return None
        """
        Camera_id = str(Camera_id)
        if Camera_id not in [f"%02d" % i for i in range(18, 33)]:
            return None
        # assert Camera_id in [f"%02d" % i for i in range(18, 33)], f"Invalid Camera_id {Camera_id}"
        assert isinstance(Frame_id, (list, int, str, type(None))), f"Invalid Frame_id datatype: {type(Frame_id)}"

        if Camera_id not in self.smc["Keypoints2d"].keys():
            print(f"not lmk2d result in camera id {Camera_id}")
            return None
        if isinstance(Frame_id, (str, int)):
            Frame_id = int(Frame_id)
            assert (
                Frame_id >= 0 and Frame_id < self.smc["Keypoints2d"].attrs["num_frame"]
            ), f"Invalid frame_index {Frame_id}"
            Frame_id = str(Frame_id)
            if (
                Frame_id not in self.smc["Keypoints2d"][Camera_id].keys()
                or self.smc["Keypoints2d"][Camera_id][Frame_id] is None
                or len(self.smc["Keypoints2d"][Camera_id][Frame_id]) == 0
            ):
                print(f"not lmk2d result in Camera_id/Frame_id {Camera_id}/{Frame_id}")
                return None
            return self.smc["Keypoints2d"][Camera_id][Frame_id]
        else:
            if Frame_id is None:
                return self.smc["Keypoints2d"][Camera_id]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                kpt2d = self.get_Keypoints2d(Camera_id, fi)
                if kpt2d is not None:
                    rs.append(kpt2d)
            return np.stack(rs, axis=0)

    ###Keypoints3d
    def get_Keypoints3d(self, Frame_id=None):
        """Get keypoint3D Frame_id
        PS: Not all the Frame_id have keypoints3d.

        Args:
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence
        Returns:
            Keypoints3d tensor: np.ndarray of shape ([N], ,3)
            if data do not exist: None
        """
        if isinstance(Frame_id, (str, int)):
            Frame_id = int(Frame_id)
            assert (
                Frame_id >= 0 and Frame_id < self.smc["Keypoints3d"].attrs["num_frame"]
            ), f"Invalid frame_index {Frame_id}"
            if str(Frame_id) not in self.smc["Keypoints3d"].keys() or len(self.smc["Keypoints3d"][str(Frame_id)]) == 0:
                print(f"get_Keypoints3d: data of frame {Frame_id} do not exist.")
                return None
            return self.smc["Keypoints3d"][str(Frame_id)]
        else:
            if Frame_id is None:
                return self.smc["Keypoints3d"]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                kpt3d = self.get_Keypoints3d(fi)
                if kpt3d is not None:
                    rs.append(kpt3d)
            return np.stack(rs, axis=0)

    ###FLAME
    def get_FLAME(self, Frame_id=None):
        """Get FLAME (world coordinate) computed by flame-fitting processing pipeline.
        FLAME is only provided in expression part.

        Args:
            Frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            dict:
                "global_pose"                   : double (3,)
                "neck_pose"                     : double (3,)
                "jaw_pose"                      : double (3,)
                "left_eye_pose"                 : double (3,)
                "right_eye_pose"                : double (3,)
                "trans"                         : double (3,)
                "shape"                         : double (100,)
                "exp"                           : double (50,)
                "verts"                         : double (5023,3)
                "albedos"                       : double (3,256,256)
        """
        if "e" not in self.performance_part.split("_")[0]:
            print(f"no flame data in the performance part: {self.performance_part}")
            return None
        if "FLAME" not in self.smc.keys():
            print("not flame parameters, please check the performance part.")
            return None
        flame = self.smc["FLAME"]
        if Frame_id is None:
            return flame
        elif isinstance(Frame_id, list):
            frame_list = [str(fi) for fi in Frame_id]
            rs = []
            for fi in tqdm.tqdm(frame_list):
                rs.append(self.get_FLAME(fi))
            return np.stack(rs, axis=0)
        elif isinstance(Frame_id, (int, str)):
            Frame_id = int(Frame_id)
            assert Frame_id >= 0 and Frame_id < self.smc["FLAME"].attrs["num_frame"], f"Invalid frame_index {Frame_id}"
            return flame[str(Frame_id)]
        else:
            raise TypeError("frame_id should be int, list or None.")

    ###uv texture map
    def get_uv(self, Frame_id=None, disable_tqdm=True):
        """Get uv map (image form) computed by flame-fitting processing pipeline.
        uv texture is only provided in expression part.

        Args:
            Frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            a single img: HWC in bgr (uint8)
        """
        if "e" not in self.performance_part.split("_")[0]:
            print(f"no uv data in the performance part: {self.performance_part}")
            return None
        if "UV_texture" not in self.smc.keys():
            print("not uv texture, please check the performance part.")
            return None
        assert isinstance(Frame_id, (list, int, str, type(None))), f"Invalid Frame_id datatype {type(Frame_id)}"
        if isinstance(Frame_id, (str, int)):
            Frame_id = str(Frame_id)
            assert Frame_id in self.smc["UV_texture"].keys(), f"Invalid Frame_id {Frame_id}"
            img_byte = self.smc["UV_texture"][Frame_id][()]
            img_color = self.__read_color_from_bytes__(img_byte)
            return img_color
        else:
            if Frame_id is None:
                Frame_id_list = sorted([int(l) for l in self.smc["UV_texture"].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_uv(fi))
            return np.stack(rs, axis=0)

    ###scan mesh
    def get_scanmesh(self):
        """
        Get scan mesh data computed by Dense Mesh Reconstruction pipeline.
        Returns:
            dict:
                'vertex': np.ndarray of vertics point (n, 3)
                'vertex_indices': np.ndarray of vertex indices (m, 3)
        """
        if "e" not in self.performance_part.split("_")[0]:
            print(f"no scan mesh data in the performance part: {self.performance_part}")
            return None
        data = self.smc["Scan"]
        return data

    def get_scanmask(self, Camera_id=None):
        """Get image its Camera_id

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in
                    {'00'...'59'}
        Returns:
            a single img : HW (2048, 2448) (uint8)
            multiple img: NHW (N, 2048, 2448)  (uint8)
        """
        if Camera_id is None:
            rs = []
            for i in range(60):
                rs.append(self.get_scanmask(f"{i:02d}"))
            return np.stack(rs, axis=0)
        assert isinstance(Camera_id, (str, int)), f"Invalid Camera_id type {Camera_id}"
        Camera_id = str(Camera_id)
        assert Camera_id in self.smc["Camera"].keys(), f"Invalid Camera_id {Camera_id}"
        img_byte = self.smc["ScanMask"][Camera_id][()]
        img_color = self.__read_color_from_bytes__(img_byte)
        img_color = np.max(img_color, 2).astype(np.uint8)
        return img_color


### test func
if __name__ == "__main__":
    actor_part = sys.argv[1]
    st = time.time()
    print("reading smc: {}".format(actor_part))
    rd = SMCReader(f"/mnt/lustre/share_data/pandongwei/RenFace_waic_20230718/{actor_part}.smc")
    print("SMCReader done, in %f sec\n" % (time.time() - st), flush=True)

    # basic info
    print(rd.get_actor_info())
    print(rd.get_Camera_info())

    # img
    Camera_id = "25"
    Frame_id = 0

    image = rd.get_img(Camera_id, "color", Frame_id)  # Load image for the specified camera and frame
    print(f"image.shape: {image.shape}")  # (2048, 2448, 3)
    images = rd.get_img("04", "color", disable_tqdm=False)
    print(f"color {images.shape}, {images.dtype}")

    # mask
    mask = rd.get_img(Camera_id, "mask", Frame_id)
    print(f"mask.shape: {mask.shape}")  # (2048, 2448)
    l = [10, 13]
    mask = rd.get_img(13, "mask", l, disable_tqdm=False)
    mask = rd.get_img(13, "mask", disable_tqdm=False)
    print(f" mask {mask.dtype} {mask.shape}")

    # camera
    cameras = rd.get_Calibration_all()
    print(f"all_calib 30 RT: {cameras['30']['RT']}")
    camera = rd.get_Calibration(15)
    print(" split_calib ", camera)

    # audio
    if "_s" in actor_part:
        audio = rd.get_audio()
        print("audio", audio["audio"].shape, "sample_rate", np.array(audio["sample_rate"]))
        sr = int(np.array(audio["sample_rate"]))
        arr = np.array(audio["audio"])
        rd.writemp3(f="./test.mp3", sr=sr, x=arr, normalized=True)

    # landmark
    lmk2d = rd.get_Keypoints2d("25", 4)
    print("kepoint2d", lmk2d.shape)
    lmk2ds = rd.get_Keypoints2d("26", [1, 2, 3, 4, 5])
    print(f"lmk2ds.shape: {lmk2ds.shape}")
    lmk3d = rd.get_Keypoints3d(4)
    print(f"kepoint3d shape: {lmk3d.shape}")
    lmk3d = rd.get_Keypoints3d([1, 2, 3, 4, 5])
    print(f"kepoint3d shape: {lmk3d.shape}")

    # flame
    if "_e" in actor_part:
        flame = rd.get_FLAME(56)
        print(f"keys: {flame.keys()}")
        print(f"global_pose: {flame['global_pose'].shape}")
        print(f"neck_pose: {flame['neck_pose'].shape}")
        print(f"jaw_pose: {flame['jaw_pose'].shape}")
        print(f"left_eye_pose: {flame['left_eye_pose'].shape}")
        print(f"right_eye_pose: {flame['right_eye_pose'].shape}")
        print(f"trans: {flame['trans'].shape}")
        print(f"shape: {flame['shape'].shape}")
        print(f"exp: {flame['exp'].shape}")
        print(f"verts: {flame['verts'].shape}")
        print(f"albedos: {flame['albedos'].shape}")
        flame = rd.get_FLAME()
        print(f"keys: {flame.keys()}")

    # uv texture
    if "_e" in actor_part:
        uv = rd.get_uv(Frame_id)
        print(f"uv shape: {uv.shape}")
        uv = rd.get_uv()
        print(f"uv shape: {uv.shape}")

    # scan mesh
    if "_e" in actor_part:
        scan = rd.get_scanmesh()
        print(f"keys: {scan.keys()}")
        print(f"vertex: {scan['vertex'].shape}")
        print(f"vertex_indices: {scan['vertex_indices'].shape}")
        rd.write_ply(scan, "./test_scan.ply")

    # scan mask
    if "_e" in actor_part:
        scanmask = rd.get_scanmask("03")
        print(f"scanmask.shape: {scanmask.shape}")
        scanmask = rd.get_scanmask()
        print(f"scanmask.shape all: {scanmask.shape}")
