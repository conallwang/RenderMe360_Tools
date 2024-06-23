## A Tool to unfold RenderMe360 data

A simple tool to unfold .smc files in the [RenderMe360](https://renderme-360.github.io/inner-download.html#Download) data into a readable and usable format.

### ⚙️ &nbsp;Install

- **python base**
    - Python 3.9 or higher
- **Other Packages:**
```shell
pip install opencv-python h5py numpy tqdm pydub plyfile
```

> To download the RenderMe360 data, you need to install openxlab library using ```pip install -U openxlab```

### <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/cookie-bite.svg" width="25"> &nbsp;Usage

You can either test some frame data in some view using [test_data.py](./test_data.py) or unfold all data of one actor using [unfold_data.py](./unfold_data.py).

> Don't forget change '/path/to/RenderMe360' to your own dataset path.

**Some modifiable parameters**

- *DATA_ROOT*: the root path of your RenderMe360 data.
- *ACTOR_ID*: the actor index which you want to test or unfold.
- *UNFOLD_LIST*: identify which items you want to unfold. (You can choose from these items: "image", "mask", "uv", "scan", "lmk_2d", "lmk_3d", "audio")
- *SKIP_SEQ*: skip some expressions, speeches, or hairstyles.
