import json
import os
import re

import cv2
import numpy as np
from tqdm import tqdm

from smc_reader import SMCReader
from utils import ITEM2EXT, ITEM2FORLDER, directory, write_ply


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_general_data(savepath, raw_smc, anno_smc, item, f_id, c_id):
    if item == "image":
        image = raw_smc.get_img(c_id, "color", f_id)
        cv2.imwrite(savepath, image)
    elif item == "masked_image":
        image = raw_smc.get_img(c_id, "color", f_id)
        mask = anno_smc.get_img(c_id, "mask", f_id)
        cv2.imwrite(savepath, image * (mask / 255.0)[..., None])
    elif item == "mask":
        mask = anno_smc.get_img(c_id, "mask", f_id)
        cv2.imwrite(savepath, mask)
    elif item == "uv":
        uv = anno_smc.get_uv(f_id)
        if uv is not None:
            cv2.imwrite(savepath, uv)
    elif item == "scan":
        scan = anno_smc.get_scanmesh()
        if scan is not None:
            write_ply(scan, savepath)
    elif item == "lmk_2d":
        lmk2d = anno_smc.get_Keypoints2d(c_id, f_id)
        if lmk2d is not None:
            np.save(savepath, lmk2d)
    elif item == "lmk_3d":
        lmk3d = anno_smc.get_Keypoints3d(f_id)
        if lmk3d is not None:
            np.save(savepath, lmk3d)
    else:
        print("item {} has not been implemented.".format(item))


DATA_ROOT = "/path/to/RenderMe360/OpenXDLab___RenderMe-360/"
ACTOR_ID = "0026"
UNFOLD_LIST = [
    "image",
    "masked_image",
    "mask",
    "uv",
    # "scan",
    # "lmk_2d",
    # "lmk_3d",
    # "audio",
]  # Choose from these items: "image", "mask", "uv", "scan", "lmk_2d", "lmk_3d", "audio"
SKIP_SEQ = []

out_dir = os.path.join(DATA_ROOT, "preprocess", ACTOR_ID)
directory(out_dir)

anno_dir = os.path.join(DATA_ROOT, "anno", ACTOR_ID)
seqs = []
for file in os.listdir(anno_dir):
    if "anno" not in file:
        continue

    pattern = r"{}_(.*)_anno.smc".format(ACTOR_ID)
    seq = re.findall(pattern, file)[0]
    seqs.append(seq)
seqs.sort()  # Get all sequences

for seq in seqs:
    if seq in SKIP_SEQ:
        continue

    raw_file = os.path.join(DATA_ROOT, "raw", ACTOR_ID, f"{ACTOR_ID}_{seq}_raw.smc")
    anno_file = os.path.join(DATA_ROOT, "anno", ACTOR_ID, f"{ACTOR_ID}_{seq}_anno.smc")
    raw_reader = SMCReader(raw_file)
    anno_reader = SMCReader(anno_file)

    cam_info = raw_reader.get_Camera_info()
    actor_info = raw_reader.get_actor_info()
    n_frame = cam_info["num_frame"]
    n_frame = 1  # For test: only see frame 0
    n_cam = cam_info["num_device"]

    print("Processing seq '{}' ... ".format(seq))
    for item in UNFOLD_LIST:
        seq_dir = os.path.join(out_dir, seq, ITEM2FORLDER[item])
        directory(seq_dir)

        bar = tqdm(range(n_frame * n_cam))
        bar.set_description("Unfold {}".format(item))

        for f_id in range(n_frame):
            for c_id in range(n_cam):
                savepath = os.path.join(seq_dir, "{:05}_{:02}{}".format(f_id, c_id, ITEM2EXT[item]))
                save_general_data(savepath, raw_reader, anno_reader, item, f_id, "{:02}".format(c_id))
                bar.update()

    # Construct cam file
    json_contents = {
        "actor_id": raw_reader.actor_id,
        "performance_part": raw_reader.performance_part,
        "capture_date": raw_reader.capture_date,
        "age": actor_info["age"],
        "color": actor_info["color"],
        "gender": actor_info["gender"],
        "height": actor_info["height"],
        "weight": actor_info["weight"],
        "img_h": cam_info["resolution"][0],
        "img_w": cam_info["resolution"][1],
        "n_frames": cam_info["num_frame"],
        "n_cams": cam_info["num_device"],
        "frames": [],
    }
    for f_id in range(n_frame):
        for c_id in range(n_cam):
            calib = anno_reader.get_Calibration("{:02}".format(c_id))
            frame = {
                "timestep_index": f_id,
                "camera_index": c_id,
                "cx": calib["K"][0, 2],
                "cy": calib["K"][1, 2],
                "fx": calib["K"][0, 0],
                "fy": calib["K"][1, 1],
                "img_h": cam_info["resolution"][0],
                "img_w": cam_info["resolution"][1],
                "transform_matrix": calib["RT"],
            }
            json_contents["frames"].append(frame)

    cam_path = os.path.join(out_dir, seq, "calib.json")
    with open(cam_path, "w") as fp:
        json.dump(json_contents, fp, cls=NpEncoder, indent=4)
