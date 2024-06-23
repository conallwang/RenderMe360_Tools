import os

import cv2
import numpy as np

from smc_reader import SMCReader
from utils import directory, vislmks, write_obj, write_ply

DATA_ROOT = "/path/to/RenderMe360/OpenXDLab___RenderMe-360/"
ACTOR_ID = "0026"
ACTION_TYPE = "e"  # 'e' for expressions, 's' for speeches, and 'h' for hair
ACTION_ID = "0"
SUB_ACTION_ID = ""

OUTDIR = "./test_data"
directory(OUTDIR)

raw_file = os.path.join(DATA_ROOT, "raw", ACTOR_ID, f"{ACTOR_ID}_{ACTION_TYPE}{ACTION_ID}_{SUB_ACTION_ID}_raw.smc")
anno_file = os.path.join(DATA_ROOT, "anno", ACTOR_ID, f"{ACTOR_ID}_{ACTION_TYPE}{ACTION_ID}_{SUB_ACTION_ID}_anno.smc")
if SUB_ACTION_ID == "":
    raw_file = raw_file[:-9] + raw_file[-8:]
    anno_file = anno_file[:-10] + anno_file[-9:]
raw_reader = SMCReader(raw_file)
anno_reader = SMCReader(anno_file)

camera_id = "25"
frame_id = 0

# Audio
audio = raw_reader.get_audio()

# Image
image = raw_reader.get_img(camera_id, "color", frame_id)
mask = anno_reader.get_img(camera_id, "mask", frame_id)
cv2.imwrite(os.path.join(OUTDIR, "{:05}_{}.png".format(frame_id, camera_id)), image)
cv2.imwrite(os.path.join(OUTDIR, "{:05}_{}_mask.png".format(frame_id, camera_id)), mask)

img_h, img_w, _ = image.shape

# Calibration
calibration = anno_reader.get_Calibration(camera_id)
print(calibration["K"].shape, calibration["D"].shape, calibration["RT"].shape)

# Landmark 2d
lmk2d = anno_reader.get_Keypoints2d(camera_id, frame_id)
vislmks(os.path.join(OUTDIR, "{:05}_{}_lmk2d.png".format(frame_id, camera_id)), lmk2d, img_h, img_w, image)

# UV
uv = anno_reader.get_uv(frame_id)
cv2.imwrite(os.path.join(OUTDIR, "{:05}_{}_uv.png".format(frame_id, camera_id)), uv)

# Scan
scan = anno_reader.get_scanmesh()
write_ply(scan, os.path.join(OUTDIR, "{:05}_{}.ply".format(frame_id, camera_id)))
