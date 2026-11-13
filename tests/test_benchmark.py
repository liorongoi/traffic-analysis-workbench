"""The benchmark script's dataset readers and scoring, on tiny hand-made inputs (no datasets or models needed)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)


def test_visdrone_classes_vans_and_ignored_regions():
    text = "10,10,20,10,1,4,0,0\n40,10,20,10,1,5,0,0\n0,0,5,5,0,0,0,0\n70,10,10,10,1,7,0,0\n1,1,3,8,1,2,0,0\n"
    anns = bm.visdrone_annotations(1, text)
    real = [a for a in anns if not a["iscrowd"]]
    assert [a["category_id"] for a in real] == [3, 3, 1]          # car, van counted as car, "people" as person
    crowd = [a for a in anns if a["iscrowd"]]
    assert len([a for a in crowd if a["bbox"] == [0.0, 0.0, 5.0, 5.0]]) == len(bm.CATS)   # hides every class
    assert len([a for a in crowd if a["bbox"] == [70.0, 10.0, 10.0, 10.0]]) == len(bm.VEHICLES)  # tricycle


def test_hituav_names_and_other_vehicles():
    assert bm.hituav_conditions("1_120_60_0_01234.jpg") == {"night": True, "altitude_m": 120, "pitch_deg": 60}
    anns = bm.hituav_annotations([
        {"image_id": 1, "bbox": [0, 0, 10, 10], "category_id": 1},
        {"image_id": 1, "bbox": [20, 0, 10, 10], "category_id": 3},
        {"image_id": 1, "bbox": [40, 0, 10, 10], "category_id": 4}])
    per_class = bm.class_view(anns)
    assert sum(1 for a in per_class if not a["iscrowd"]) == 1        # only the car is a real per-class box
    veh = bm.vehicle_view(anns)
    assert sum(1 for a in veh if not a["iscrowd"]) == 2              # car and other vehicle both count as vehicles
    assert sum(1 for a in veh if a["iscrowd"]) == 1                  # the don't-care region once, not per class


def test_operating_point_counts_crowd_as_neutral():
    images = [{"id": 1, "width": 100, "height": 100}]
    anns = [{"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "iscrowd": 0, "area": 100},
            {"image_id": 1, "category_id": 3, "bbox": [50, 50, 40, 40], "iscrowd": 1, "area": 1600}]
    dets = [{"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 3, "bbox": [60, 60, 10, 10], "score": 0.8},      # inside the crowd
            {"image_id": 1, "category_id": 3, "bbox": [20, 0, 10, 10], "score": 0.7},      # false
            {"image_id": 1, "category_id": 3, "bbox": [30, 0, 10, 10], "score": 0.1}]      # below the threshold
    op = bm.operating_point(images, anns, dets, 3, 0.3)
    assert (op["tp"], op["fp"], op["gt"]) == (1, 1, 1)
    assert op["precision"] == 0.5 and op["recall"] == 1.0


def test_coco_eval_perfect_and_empty():
    pytest.importorskip("pycocotools")
    images = [{"id": 1, "width": 100, "height": 100}]
    anns = [{"image_id": 1, "category_id": 3, "bbox": [10, 10, 30, 30], "iscrowd": 0, "area": 900}]
    good = [{"image_id": 1, "category_id": 3, "bbox": [10, 10, 30, 30], "score": 0.9}]
    r = bm.coco_eval(images, anns, good, {"car": 3}, 100)
    assert r["AP50"] == pytest.approx(1.0) and r["per_class"]["car"]["gt"] == 1
    assert bm.coco_eval(images, anns, [], {"car": 3}, 100)["AP"] == 0.0


def test_sample_frames_follow_the_video_grid():
    assert bm.sample_frames(10, 30.0, None) == list(range(1, 11))
    # 30 frames per second, one analysed frame every 0.2 s: frames 1, 7, 13, ...
    assert bm.sample_frames(30, 30.0, 0.2) == [1, 7, 13, 19, 25]
    # 14 frames per second (MOT17-05): the first frame at or after each grid point
    assert bm.sample_frames(15, 14.0, 0.2)[:4] == [1, 4, 7, 10]


def test_gt_subset_renumbers_frames(tmp_path):
    seq = tmp_path / "MOT17-99"
    (seq / "gt").mkdir(parents=True)
    rows = [[1, 1, 0, 0, 5, 5, 1, 1, 1.0], [7, 1, 1, 0, 5, 5, 1, 1, 1.0], [7, 2, 9, 9, 5, 5, 1, 1, 0.5],
            [3, 3, 0, 0, 5, 5, 1, 1, 1.0], [7, 4, 0, 0, 5, 5, 0, 8, 1.0]]
    np.savetxt(seq / "gt" / "gt.txt", rows, delimiter=",", fmt="%g")
    info = {"name": "MOT17-99", "fps": 30.0, "width": 64, "height": 48}
    got = bm.write_gt_subset(seq, info, [1, 7], tmp_path / "out")
    assert got == {"gt_people": 2, "gt_boxes": 3}                    # frame 3 dropped; a distractor not counted
    out = np.loadtxt(tmp_path / "out" / "MOT17-99" / "gt" / "gt.txt", delimiter=",")
    assert sorted(set(out[:, 0].astype(int))) == [1, 2]
    assert "seqLength=2" in (tmp_path / "out" / "MOT17-99" / "seqinfo.ini").read_text()
