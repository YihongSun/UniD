# Dataset preparation

UniD includes dataset readers, split manifests, annotation conversions, and ten training-index builders. It does not include benchmark or training datasets, generated pseudo-labels, or a downloader for them. Obtain data from its provider, review the provider's current access and license terms, and prepare it locally. The builders only read source files and atomically write Parquet indexes under the configured index root.

## Local roots and index generation

Set `UNID_DATA_ROOT` to the parent directory for the conventional dataset layout. It defaults to `data/`. Set `UNID_INDEX_ROOT` if you want generated Parquet files stored separately; it defaults to `UNID_DATA_ROOT`. Training loaders resolve indexes from this variable when they start, so export it before launching training. The index contains absolute paths to the source files; keep the source mount in place or regenerate the index after moving the data. A separate index root keeps generated files out of the source dataset tree.

Install the dependencies with `python -m pip install -r requirements.txt`. Run preparation commands from the repository root. Each command takes a dataset-specific extracted input root and can take an output root. For example:

```bash
export UNID_DATA_ROOT=/datasets/unid
export UNID_INDEX_ROOT=/datasets/unid-indexes

python -m dataloaders.prepare_dataset dms \
  --split train \
  --input-root /datasets/source/DMS_v1
```

This writes `/datasets/unid-indexes/DMS_v1/dms_train.parquet`. Pass `--output-root /another/index/root` to override the destination for a single invocation; if training will read that index, set `UNID_INDEX_ROOT` to the same destination before starting the loader. The output is written through a temporary file and atomically replaced after a successful build, so reruns do not modify source files or leave a partially written index.

The same entry point handles `coco`, `cityscapes`, `densepose`, `dms`, `hypersim`, `virtual_kitti_2`, `point_odyssey`, `tartanair`, `spring`, and `vipseg`. `--split` defaults to `train`. Virtual KITTI 2 indexes its configured training scenes only; TartanAir has no split in this loader and indexes every Hard trajectory under the root as training data. Builders sort directory listings before indexing; the COCO builder also uses a fixed random seed when selecting optional pseudo-label examples.

## Training dataset roots and required files

The paths below describe the layouts read by the in-repository builders and training loaders. Start with the upstream release layout and adapt it into these trees without modifying the upstream files in place.

| Dataset | `--input-root` and required layout | Builder output | Training consumer | Source and access notes |
| --- | --- | --- | --- | --- |
| COCO | Root containing `train2017/*.jpg`, `annotations/panoptic_train2017/*.png`, `annotations/panoptic_train2017.json`, and `annotations/person_keypoints_train2017.json`. Optional `annotations/depth-pro_pred/` and `annotations/dsine_pred/` maps are sampled into some rows. | `coco/coco_<split>.parquet` | `ImageCOCODataset` (semantic labels, boundaries, keypoints) | [COCO download and terms](https://cocodataset.org/#download). COCO annotations and images have different rights; follow the image providers' terms too. |
| DensePose | The same COCO root, with `train2014/*.jpg` and `annotations/densepose_coco_2014_<split>.json`. | `coco/densepose_<split>.parquet` | `ImageDensePoseDataset` (COCO part segmentation) | DensePose annotations accompany the [COCO dataset](https://cocodataset.org/#download); acquire any restricted annotations from their authorized source. |
| Cityscapes | Root containing `leftImg8bit/<split>/<city>/*_leftImg8bit.png` and matching `gtFine/<split>/<city>/*_gtFine_instanceIds.png`. Optional pseudo-labels can be placed in `<split>_depth-pro_pred/`, `<split>_dsine_pred/`, and `<split>_rtmo_pred/`. | `cityscapes/cityscapes_<split>.parquet` | `ImageCityscapesDataset` (semantic labels and boundaries) | [Official Cityscapes site and terms](https://www.cityscapes-dataset.com/). Registration is required; its terms limit use and redistribution. |
| DMS | Root containing `taxonomy.json`, `images/<split>/*.jpg`, and matching `labels/<split>/*.png`. The DMS labels and source RGB images have separate licensing. | `DMS_v1/dms_<split>.parquet` | `ImageDMSDataset` (material segmentation) | [Apple DMS release](https://github.com/apple-aiml-research/ml-dms-dataset). The DMS release states non-commercial terms; acquire the original Open Images separately and follow its terms. |
| Hypersim | `--input-root` is the processed root containing `<split>/filename_meta_train.csv` plus the CSV-referenced RGB, depth, albedo, and illumination files. The default training layout is `hypersim/processed/`; camera normals are expected in sibling `hypersim/normals/`. Optional OneFormer and RTMO labels are expected under sibling `oneformer-pred/` and `rtmo-pred/`. | `hypersim/processed/hypersim_<split>.parquet` | `ImageHypersimDataset` (depth, normals, albedo, illumination) | [Apple Hypersim project and preparation code](https://github.com/apple/ml-hypersim). Use the official data terms and observe the licenses of the underlying scene assets. The repository does not convert the raw scene archives into this processed layout. |
| Virtual KITTI 2 | Root containing `vkitti_2.0.3_rgb/`, `vkitti_2.0.3_depth/`, and `vkitti_DAG_normals/`, each arranged by scene, weather, frame, and camera. Optional pseudo labels are under `vkitti_2.0.3_oneformer-pred/` and `vkitti_2.0.3_rtmo-pred/`. | `virtual_kitti_2/virtual_kitti_2_<split>.parquet` | `ImageVKITTI2Dataset` (depth and normals) | [NAVER LABS Europe dataset page](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds/). Check current download and use terms. |
| PointOdyssey | Root containing `train/<scene>/anno.npz`, `rgbs/`, and `depths/`. Optional predictions use `train_oneformer-pred/<scene>/rgbs/` and `train_rtmo-pred/<scene>/rgbs/`. | `point_odyssey/point_odyssey_<split>.parquet` | `VideoPointOdysseyDataset` (temporal depth) | [Official PointOdyssey project](https://github.com/y-zheng18/point_odyssey); the project page links the dataset release and license. |
| TartanAir | Root containing `<environment>/Hard/<trajectory>/image_left/`, `depth_left/`, and `pose_left.txt`. Optional pseudo-labels use root-level `oneformer-pred/` and `rtmo-pred/` trees mirroring the source paths. | `tartanair/tartanair_<split>.parquet` | `VideoTartanAirDataset` (temporal depth) | [TartanAir source and download tools](https://theairlab.org/tartanair-dataset/). Select the image, depth, and pose modalities and follow the provider's current terms. The builder indexes all matching Hard trajectories under the root. |
| Spring | Root containing `<split>/<scene>/frame_left/`, `disp1_left/`, and `cam_data/intrinsics.txt`. Optional predictions use `<split>_oneformer-pred/` and `<split>_rtmo-pred/`. Disparity is read at training time and converted to depth using the camera intrinsics and the Spring baseline. | `spring/spring_<split>.parquet` | `VideoSpringDataset` (temporal depth) | [Spring Dataset & Benchmark](https://spring-benchmark.org/). Use the provider's download and use terms. |
| VIPSeg | Root containing `<split>.txt`, `imgs/<video>/`, and `panomasks/<video>/`. Optional predictions use `<split>_depth-pro_pred/`, `<split>_dsine_pred/`, and `<split>_rtmo_pred/`, each grouped by video. | `VIPSeg/vipseg_<split>.parquet` | `VideoVIPSegDataset` (semantic labels and boundaries) | [Official VIPSeg project](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset); the project states research-only restrictions. |

The optional pseudo-label folders are consumed only when matching files exist. They are not downloaded or generated by UniD. Most builders index source ground truth and do not perform resizing, annotation conversion, or prediction generation. For COCO, the builder requires panoptic PNG masks; the training reader separately loads the panoptic and keypoint JSON files. The DensePose reader decodes the polygon/RLE annotations while loading examples. Cityscapes and VIPSeg use their provided annotation masks. The shared loader converts panoptic IDs into semantic maps and boundaries at read time. Spring disparity is converted to metric depth at read time. These transformations are implemented in `dataloaders/datasets/` and `dataloaders/datasets/utils.py`.

## Evaluation datasets and split files

The inference evaluation configs in `Unid/config/dataset/` pair a local dataset root with a relative directory or archive and a checked-in filename list. The lists are text manifests, not the underlying data. Keep their relative paths intact when preparing local data. For the depth datasets that are stored as tar archives, the inference reader can access members without extracting them; the custom evaluation scripts generally expect the directory trees described by their command-line arguments.

| Evaluation family | Configured local tree or archive | Split manifest | Upstream source |
| --- | --- | --- | --- |
| ADE20K semantic/material labels | `images/validation/`; material evaluation also needs `dms_annotations/validation/` | `Unid/data_split/ade20k/validation.txt` | [ADE20K download and terms](https://ade20k.csail.mit.edu/) |
| Cityscapes | `leftImg8bit/val/` and matching `gtFine/val/` annotations | `Unid/data_split/cityscapes/val.txt` | [Cityscapes](https://www.cityscapes-dataset.com/) |
| COCO boundary and DensePose part segmentation | COCO 2017 validation images and panoptic annotations; COCO 2014 validation images and DensePose annotations, respectively | `Unid/data_split/coco/val2017_boundary.txt`, `Unid/data_split/coco/val2014_partseg.txt` | [COCO](https://cocodataset.org/#download) |
| DIODE depth and normals | Provider archive layout, including image, depth, validity-mask, and normal files | `Unid/data_split/diode/diode_val_all_filename_list.txt` | [DIODE toolkit and data layout](https://github.com/diode-dataset/diode-devkit) |
| ETH3D depth | Provider's ETH3D archive layout | `Unid/data_split/eth3d/eth3d_filename_list.txt` | [ETH3D datasets and terms](https://eth3d.ethz.ch/datasets) |
| Hypersim | `processed/val/` and related processed files | `Unid/data_split/hypersim/filename_list_val_filtered.txt` | [Apple Hypersim](https://github.com/apple/ml-hypersim) |
| IIW albedo | `images/` with the corresponding intrinsic-image annotations | `Unid/data_split/iiw/test.txt` | [Intrinsic Images in the Wild project](http://intrinsic.cs.cornell.edu/) |
| InteriorVerse | `mini_val/images/` and the benchmark's annotations | `Unid/data_split/intverse/mini_val.txt` | [InteriorVerse access and terms](https://interiorverse.github.io/) |
| KITTI Eigen depth | `kitti/kitti_eigen_split_test.tar` | `Unid/data_split/kitti/eigen_test_files_with_gt.txt` | [KITTI depth benchmark](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) |
| Mapillary Vistas | `validation/images/` and validation labels | `Unid/data_split/mapillary/validation.txt` | [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) |
| MINC materials | `images/` and the material labels used by the evaluation adapter | `Unid/data_split/minc/validation.txt` | [Materials in Context database](http://minc.cs.cornell.edu/) |
| NYU Depth V2 | `nyuv2/nyu_labeled_extracted.tar` | `Unid/data_split/nyu/labeled/filename_list_test.txt` | [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) |
| Pascal Part | `images/` with Pascal Person-Part annotations | `Unid/data_split/pascal/val.txt` | Obtain Pascal VOC and Person-Part annotations from their providers; check both sets of terms. |
| SAW illumination | `saw_images_512/` and the SAW annotations | `Unid/data_split/saw/test.txt` | [Original SAW release](https://github.com/kovibalu/saw_release) |
| ScanNet depth | `scannet/scannet_val_sampled_800_1.tar` | `Unid/data_split/scannet/scannet_val_sampled_list_800_1.txt` | [ScanNet access and terms](https://github.com/ScanNet/ScanNet) |

The video-depth evaluator uses a separate `video_depth_data/` tree passed with `--benchmark_path`. Its supported presets expect JSON sequence manifests at `kitti/kitti_video.json` (or `kitti_video_500.json`), `sintel/sintel_video.json`, `nyuv2/nyuv2_video_500.json`, `bonn/bonn_video.json` (or `bonn_video_500.json`), `scannet/scannet_video.json` (or `scannet_video_500.json`), and `scannet_sparse/scannet_video.json`. Each manifest points to local RGB frames, ground-truth depth, and scale factors. The required structure is a top-level object keyed by the dataset name, with a list of sequence objects; each sequence object maps its sequence name to frame objects:

```json
{
  "scannet": [
    {
      "scene0406_02": [
        {
          "image": "scene0406_02/color/000300.jpg",
          "gt_depth": "scene0406_02/depth/000300.png",
          "factor": 1000
        }
      ]
    }
  ]
}
```

The example `factor` assumes integer depth in millimeters; set it to the conversion factor for the selected benchmark files.

The manifests are not included in this repository; the user must prepare them alongside data obtained from the [KITTI](https://www.cvlibs.net/datasets/kitti/), [Sintel](http://sintel.is.tue.mpg.de/), [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), [Bonn RGB-D Dynamic](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/), or [ScanNet](https://github.com/ScanNet/ScanNet) providers. The current `quant_depth.sh` video command uses the ScanNet preset. The evaluator defaults to the supported `kitti`, `sintel`, `nyuv2_500`, `bonn`, and `scannet` presets and reports a concise error when a manifest is missing.

Some benchmark-specific prediction evaluators have additional annotation or result-directory requirements. Consult their `--help` output and the matching launcher under `experiments/eval_scripts/`. This repository does not contain dataset-specific extraction or conversion programs for every evaluation benchmark. In particular, ADE20K DMS labels, InteriorVerse annotations, Pascal Person-Part labels, Mapillary labels, and SAW annotations must be acquired or prepared from their authorized sources before the corresponding evaluator can run.

## Safe preparation and provenance

The index builders do not download files, delete data, or rewrite raw inputs. They write a new Parquet index under `UNID_INDEX_ROOT`; rerunning a builder replaces only that index after a successful write. Custom annotations and pseudo-labels should be kept separate from upstream source trees where practical. Keep provider licenses, access agreements, and citations with your local data records. Do not commit raw datasets, derived annotations, pseudo-labels, indexes tied to private locations, or evaluation predictions to this repository.
