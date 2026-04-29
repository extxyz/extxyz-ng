import io
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
    assert frame.info == {
        "key1": "a",
        "key2": "a/b",
        "key3": "a@b",
        "key4": "a@b",
        "Properties": "species:S:1:pos:R:3",
    }
    assert frame.arrs == {
        "species": ["Mg", "C", "C", "C"],
        "pos": [
            [-4.2565, 3.7918, -2.54123],
            [-1.15405, 2.86652, -1.26699],
            [-5.53758, 3.70936, 0.63504],
            [-7.2825, 4.71303, -3.82016],
        ],
    }

    # check str works
    frame = read_frame_from_file(str(p))
    assert frame.natoms == 4


def test_write_default(tmp_path: Path, default_frame: Frame) -> None:
    fpath = tmp_path / "foo.xyz"
    with open(fpath, "wb") as fh:
        write_frame(fh, default_frame)

    text = fpath.read_text()
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


def test_write_frames_round_trip(tmp_path: Path) -> None:
    p = Path(__file__).parent / "mgb_multi_frames.xyz"
    fpath = tmp_path / "foo.xyz"

    with open(p, "rb") as fh_read, open(fpath, "wb") as fh_write:
        # should not close the file handler for read when streaming.
        frames = read_frames(fh_read)
        write_frames(fh_write, frames)

    fpath = tmp_path / "foo.xyz"
    text_foo = fpath.read_text()

    inp = io.BytesIO(text_foo.encode(encoding="utf-8"))
    yet_frames = read_frames(inp)
    out = tmp_path / "baz.xyz"
    with open(out, "wb") as fh_write:
        write_frames(fh_write, yet_frames)

    text_baz = out.read_text()

    assert text_foo == text_baz


def test_read_frames_from_file_and_write_to_file(tmp_path: Path) -> None:
    p = Path(__file__).parent / "mgb_multi_frames.xyz"
    fpath = tmp_path / "foo.xyz"

    frames = read_frames_from_file(p)

    with open(fpath, "wb") as fh_write:
        write_frames(fh_write, frames)


# TODO: lazy read cause python context scope rule broken.
@pytest.mark.skip(
    reason="rework needed, this will fail, because frame didn't live long enough to go outside of the context, which it should"
)
def test_read_frames_from_and_write_using_context(tmp_path: Path) -> None:
    p = Path(__file__).parent / "mgb_multi_frames.xyz"
    fpath = tmp_path / "foo.xyz"

    with open(p, "rb") as fh_read:
        frames = read_frames(fh_read)

    with open(fpath, "wb") as fh_write:
        write_frames(fh_write, frames)
