import struct
from pathlib import Path

from trafficwb.inventory import inventory, read_image_header


def jpeg_bytes(w, h, exif=False, complete=True):
    b = b"\xff\xd8"
    if exif:
        payload = b"Exif\x00\x00" + b"\x00" * 8
        b += b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    sof = struct.pack(">BHHB", 8, h, w, 1) + b"\x01\x11\x00"
    b += b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof
    b += b"\x00" * 16
    if complete:
        b += b"\xff\xd9"
    return b


def test_jpeg_header(tmp_path: Path):
    p = tmp_path / "a.jpg"
    p.write_bytes(jpeg_bytes(1280, 720, exif=True))
    info = read_image_header(p)
    assert (info.format, info.width, info.height, info.has_exif, info.error) == ("jpeg", 1280, 720, True, None)


def test_partial_jpeg_flagged(tmp_path: Path):
    p = tmp_path / "a.jpg"
    p.write_bytes(jpeg_bytes(640, 480, complete=False))
    assert "partial" in read_image_header(p).error


def test_inventory_summary(tmp_path: Path):
    same = jpeg_bytes(1280, 720)
    (tmp_path / "C1_20261021T000000Z.jpg").write_bytes(same)
    (tmp_path / "C1_20261021T001000Z.jpg").write_bytes(same)  # identical bytes, different capture time
    (tmp_path / "C1_20261021T003000Z.jpg").write_bytes(jpeg_bytes(1280, 720))
    (tmp_path / "bad-name.jpg").write_bytes(jpeg_bytes(1280, 720))
    (tmp_path / "notes.txt").write_text("ignored")
    rep = inventory(tmp_path)
    s = rep["summary"]
    assert s["images"] == 4
    assert s["malformed"] == ["bad-name.jpg"]
    assert s["identical_content_groups"] == 1
    assert len([f for f in rep["files"] if f.get("capture_utc")]) == 3  # identical bytes stay separate captures
    assert s["interval_histogram_by_camera"]["C1"] == {"10 min": 1, "20 min": 1}
    assert s["dimensions"] == {"1280x720": 4}
