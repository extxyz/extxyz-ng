from pathlib import Path
from extxyz import read_frame_from_file, read_frame, write_frame, Frame
import pytest


@pytest.fixture
def default_frame() -> Frame:
    """read from stream (file like)"""
    with open("./mgb.xyz", "r") as fh:
        frame = read_frame(fh)
        return frame


def test_read_from_file():
    frame = read_frame_from_file("./mgb.xyz")
    assert frame.natoms == 3


def test_write_default(tmp_path: Path, frame: Frame):
    with open(tmp_path, "w") as fh:
        write_frame(fh, frame)

def test_read_frames():
    pass

def test_write_frames():
    pass

def test_read_from_ase_atoms():
    pass

def test_read_frome_ccmat_structure():
    pass
