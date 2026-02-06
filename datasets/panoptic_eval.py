# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import itertools
import json
import os
import tempfile
from collections import defaultdict

import numpy as np
import torch
from PIL import Image


class PanopticEvaluator(object):
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.predictions = []

    def update(self, predictions):
        for p in predictions:
            with io.BytesIO() as f:
                Image.fromarray(id2rgb(p.pop("image_id").cpu().numpy())).save(f, format="PNG")
                f.seek(0)
                p["png_string"] = f.read()

        self.predictions += predictions

    def synchronize_between_processes(self):
        import util.misc as utils
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        if not self.predictions:
            return None, None

        json_data = {"annotations": self.predictions}
        predictions_json = os.path.join(self.output_dir, "predictions.json")
        with open(predictions_json, "w") as f:
            f.write(json.dumps(json_data))

        from panopticapi.evaluation import pq_compute

        with open(self.gt_json, "r") as f:
            gt_json = json.load(f)

        gt_json = self.gt_json
        gt_folder = self.gt_folder
        pred_folder = self.output_dir

        with tempfile.TemporaryDirectory() as tmp_dir:
            for p in self.predictions:
                with open(os.path.join(tmp_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            pq_res = pq_compute(
                gt_json,
                predictions_json,
                gt_folder=gt_folder,
                pred_folder=tmp_dir,
            )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        results = {k: float(v) for k, v in res.items()}

        return results, pq_res


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color
