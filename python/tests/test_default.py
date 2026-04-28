from pathlib import Path

from extxyz import (
    read_frame_from_file,
    read_frame,
    read_frames_from_file,
    write_frame,
    write_frames,
    Frame,
    read_frames,
)
import pytest


@pytest.fixture
def default_frame() -> Frame:
    """read from stream (file like)"""
    p = Path(__file__).parent / "mgb.xyz"
    with open(p, "rb") as fh:
        frame = read_frame(fh)
        return frame


def test_read_frame_from_file():
    """test can read frame from file"""
    p = Path(__file__).parent / "mgb.xyz"
    frame = read_frame_from_file(p)
    assert frame.natoms == 4

    # check str works
    frame = read_frame_from_file(str(p))
    assert frame.natoms == 4


def test_write_default(tmp_path: Path, default_frame: Frame):
    fpath = tmp_path / "foo.xyz"
    with open(fpath, "wb") as fh:
        write_frame(fh, default_frame)

    with open(fpath, "r") as fh:
        text = fpath.read_text()
        print(default_frame)
        assert text in str(default_frame)


def test_read_frames():
    p = Path(__file__).parent / "mgb_multi_frames.xyz"
    with open(p, "rb") as fh:
        frames = read_frames(fh)
        count = 0

        for frame in frames:
            assert frame.natoms == 4
            count += 1

        assert count == 6


def test_read_frames_from_file():
    p = Path(__file__).parent / "mgb_multi_frames.xyz"
    frames = read_frames_from_file(p)

    count = 0
    for frame in frames:
        assert frame.natoms == 4
        count += 1

    assert count == 6

    frames = read_frames_from_file(str(p))

    count = 0
    for frame in frames:
        assert frame.natoms == 4
        count += 1

    assert count == 6


def test_write_frames():
    pass


def test_read_from_ase_atoms():
    pass


def test_read_frome_ccmat_structure():
    pass
