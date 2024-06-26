# Install openxlab using "pip install -U openxlab"
# Refer to https://opendatalab.com/OpenXDLab/RenderMe-360

import openxlab
from openxlab import dataset

YOUR_ACCESS_KEY = "TO BE INSERTED"
YOUR_SECRET_KEY = "TO BE INSERTED"

openxlab.login(ak=YOUR_ACCESS_KEY, sk=YOUR_SECRET_KEY)  # Login
dataset.info(dataset_repo="OpenXDLab/RenderMe-360")  # check dataset info
dataset.query(dataset_repo="OpenXDLab/RenderMe-360")  # check file info in the dataset
# dataset.get(dataset_repo="OpenXDLab/RenderMe-360", target_path="/path/to/local/folder/")  # download all data

dataset.download(
    dataset_repo="OpenXDLab/RenderMe-360",
    source_path="/raw/0041",
    target_path="/apdcephfs_cq10/share_1467498/cq2/datasets/RenderMe360/",
)  # Download some files in the dataset
