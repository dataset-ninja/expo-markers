import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import pycocotools.mask as mask_util
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import get_file_name, get_file_name_with_ext
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    real_im_path = "/home/alex/DATASETS/TODO/Expo M arkers/EXPO_HD/real_image_dataset/images"
    synt_im_path = "/home/alex/DATASETS/TODO/Expo M arkers/EXPO_HD/synt_image_dataset/images"
    real_json = (
        "/home/alex/DATASETS/TODO/Expo M arkers/EXPO_HD/real_image_dataset/coco_annotations.json"
    )
    synt_json = (
        "/home/alex/DATASETS/TODO/Expo M arkers/EXPO_HD/synt_image_dataset/coco_annotations.json"
    )

    batch_size = 30

    ds_name_to_data = {"synthetic": (synt_im_path, synt_json), "real": (real_im_path, real_json)}

    def convert_rle_mask_to_polygon(rle_mask_data):
        if type(rle_mask_data["counts"]) is str:
            rle_mask_data["counts"] = bytes(rle_mask_data["counts"], encoding="utf-8")
            mask = mask_util.decode(rle_mask_data)
        else:
            rle_obj = mask_util.frPyObjects(
                rle_mask_data,
                rle_mask_data["size"][0],
                rle_mask_data["size"][1],
            )
            mask = mask_util.decode(rle_obj)
        mask = np.array(mask, dtype=bool)
        return sly.Bitmap(mask).to_contours()

    def create_ann(image_path):
        labels = []

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        # img_height = image_np.shape[0]
        # img_wight = image_np.shape[1]

        img_height = image_id_to_shape[get_file_name_with_ext(image_path)][0]
        img_wight = image_id_to_shape[get_file_name_with_ext(image_path)][1]

        if annotations is not None:
            ann_data = image_name_to_ann[get_file_name_with_ext(image_path)]
            for curr_data in ann_data:
                if len(curr_data) != 0:
                    obj_class = idx_to_class[curr_data[0]]

                    rle_mask_data = curr_data[1]

                    if type(rle_mask_data) is list:
                        exterior = []
                        for coords in rle_mask_data:
                            for i in range(0, len(coords), 2):
                                exterior.append([coords[i + 1], coords[i]])
                        poligon = sly.Polygon(exterior)
                        label_poly = sly.Label(poligon, obj_class)
                        labels.append(label_poly)

                    else:
                        polygons = convert_rle_mask_to_polygon(rle_mask_data)
                        for polygon in polygons:
                            label = sly.Label(polygon, obj_class)
                            labels.append(label)

                    bbox_data = list(map(int, curr_data[2]))

                    left = bbox_data[0]
                    right = bbox_data[0] + bbox_data[2]
                    top = bbox_data[1]
                    bottom = bbox_data[1] + bbox_data[3]
                    rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                    label = sly.Label(rectangle, obj_class)
                    labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    red = sly.ObjClass("marker red", sly.AnyGeometry)
    green = sly.ObjClass("marker green", sly.AnyGeometry)
    blue = sly.ObjClass("marker blue", sly.AnyGeometry)
    black = sly.ObjClass("marker black", sly.AnyGeometry)

    idx_to_class = {1: red, 2: green, 3: blue, 4: black}

    meta = sly.ProjectMeta(obj_classes=[red, green, blue, black])

    api.project.update_meta(project.id, meta.to_json())

    for ds_name, ds_data in ds_name_to_data.items():
        image_id_to_name = {}
        image_id_to_shape = {}
        image_name_to_ann = defaultdict(list)

        images_path, ann_json = ds_data

        images_names = os.listdir(images_path)

        json_ann = load_json_file(ann_json)

        images_data = json_ann["images"]
        for image_data in images_data:
            image_id_to_name[image_data["id"]] = image_data["file_name"]
            image_id_to_shape[image_data["file_name"]] = (image_data["height"], image_data["width"])

        annotations = json_ann["annotations"]
        for ann in annotations:
            image_name_to_ann[image_id_to_name[ann["image_id"]]].append(
                [ann["category_id"], ann["segmentation"], ann["bbox"]]
            )

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(images_path, im_name) for im_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
