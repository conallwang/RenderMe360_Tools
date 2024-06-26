import os

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

ITEM2FORLDER = {
    "image": "images",
    "masked_image": "masked_images",
    "mask": "masks",
    "uv": "uvs_256",
    "scan": "scans",
    "lmk_2d": "lmks_2d",
    "lmk_3d": "lmks_3d",
    "audio": "audios",
}

ITEM2EXT = {
    "image": ".png",
    "masked_image": ".png",
    "mask": ".png",
    "uv": ".png",
    "scan": ".ply",
    "lmk_2d": ".npy",
    "lmk_3d": ".npy",
    "audio": ".wav",
}


def directory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


def vislmks(filename, lmks_2d, img_h, img_w, bg_img=None):
    if bg_img is None:
        bg_img = np.zeros((img_h, img_w, 3))
    # img_h, img_w = bg_img.shape[:2]
    bg_img[lmks_2d[:, 1].astype(np.int64), lmks_2d[:, 0].astype(np.int64), :] = 255.0

    cv2.imwrite(filename, bg_img)


def write_obj(filepath, verts, tris=None, log=True):
    """将mesh顶点与三角面片存储为.obj文件,方便查看

    Args:
        verts:      Vx3, vertices coordinates
        tris:       n_facex3, faces consisting of vertices id
    """
    fw = open(filepath, "w")
    # vertices
    for vert in verts:
        fw.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

    if not tris is None:
        for tri in tris:
            fw.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    fw.close()
    if log:
        print(f"mesh has been saved in {filepath}.")


def write_ply(scan, outpath):
    vertex = np.empty(len(scan["vertex"]), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i in range(len(scan["vertex"])):
        vertex[i] = np.array(
            [(scan["vertex"][i, 0], scan["vertex"][i, 1], scan["vertex"][i, 2])],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        )
    triangles = scan["vertex_indices"]
    face = np.empty(
        len(triangles), dtype=[("vertex_indices", "i4", (3,)), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    for i in range(len(triangles)):
        face[i] = np.array(
            [([triangles[i, 0], triangles[i, 1], triangles[i, 2]], 255, 255, 255)],
            dtype=[("vertex_indices", "i4", (3,)), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
    PlyData([PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")], text=True).write(outpath)
