# Build-Patch-for-Sdpc
Build patches from whole slide images (WSIs).

Supported slide format: "sdpc", "svs", "ndpi", "tiff", "tif", "dcm", "svslide", "bif", "vms", "vmu", "mrxs", "scn".

Supported annotation format (for sdpc): "sdpl", "json".

Two approaches (with or without annotations) to build patches are given. 

You can use a given magnification (e.g. 20X) or resolution (e.g. 0.4um/pixel) to tile patches.

# Environment Setup
`pip install opencv-python`

`pip install openslide-python`

`pip install sdpc-for-python` (refer to [sdpc-for-python](https://github.com/WonderLandxD/sdpc-for-python))

# Parameter Description
`data_dir`: directory of slide files.

`save_dir`: directory of patch saving.

`annotation_dir`: directory of annotation files (optional). If 'annotation_dir' exists, patches with annotations will be tiled. Otherwise, leave it blank.

`csv_path`: path of csv file (optional). If 'csv_path' exists, patches within `pd.read_csv(csv_path)["slide_id"]` will be tiled. Otherwise, leave it blank.

`which2cut`: use "magnification" or "resolution" to cut patches. If "magnification" is chosen, patches will be tiled based on the given `magnification`. Otherwise, patches will be tiled based on the given `resolution`.

`magnification`: magnification to cut patch: 5x, 20x, 40x, ...

`resolution`: resolution to cut patch:  0.103795, ... (um/pixel).

`patch_w`: width of patch.

`patch_h`: height of patch.

`overlap_w`: overlap width of patch.

`overlap_h`: overlap height of patch.

`thumbnail_level`: top level to catch WSI thumbnail images (larger is higher resolution).

`use_otsu`: use the Otsu algorithm to accelerate tiling patches or not.

`blank_rate_th`: cut patches with a blank rate lower than this threshold.

`null_th`: threshold to drop null patches (larger to drop more): 5, 10, 15, 20, ...

# Let's Get Started
build patches at a magnification of 20X:
```python
python build_patch.py --data_dir DATA_DIR --save_dir SAVE_DIR --which2cut "magnification" --magnification 20 --patch_w 256 --patch_h 256
```

build patches at a resolution of 0.4 um/pixel:
```python
python build_patch.py --data_dir DATA_DIR --save_dir SAVE_DIR --which2cut "resolution" --resolution 0.4 --patch_w 256 --patch_h 256
```

build patches at a resolution of 0.4 um/pixel within annotations:
```python
python build_patch.py --data_dir DATA_DIR --save_dir SAVE_DIR --which2cut "resolution" --resolution 0.4 --patch_w 256 --patch_h 256 --annotation_dir ANNOTATION_DIR
```

build patches at a resolution of 0.4 um/pixel within annotations, where WSIs are included in `pd.read_csv(csv_path)["slide_id"]`:
```python
python build_patch.py --data_dir DATA_DIR --save_dir SAVE_DIR --which2cut "resolution" --resolution 0.4 --patch_w 256 --patch_h 256 --annotation_dir ANNOTATION_DIR --csv_path CSV_PATH
```
