from collections.abc import Iterator
import typing
import os

class Frame:
    """Represents a single atomic configuration or data frame."""

    def natoms(self) -> int:
        """Number of atoms in the configuration."""

    def info(self) -> dict[str, str | int | float | bool | list[float | int]]:
        """Frame-level metadata (e.g. simulation or system properties)."""

    def arrs(self) -> dict[str, list[float | int | bool | str]]:
        """Per-atom or per-site tabular data stored as columns."""

def read_frame(stream: typing.BinaryIO) -> Frame:
    """Read a single frame from a text stream.

    Parameters:
        stream: A file-like text stream ("r") containing encoded frame data.

    Returns:
        A parsed Frame object.
    """
    ...

def read_frames(stream: typing.BinaryIO) -> Iterator[Frame]:
    """Read frames from a text stream.

    Parameters:
        stream: A file-like text stream ("r") containing encoded frames.

    Yields:
        Frame objects parsed sequentially from the stream.
    """
    ...

def read_frame_from_file(inp: str | os.PathLike[str], /) -> Frame:
    """Read a frame from a file path.

    Parameters:
        inp: Path to the input file.

    Returns:
        A parsed Frame object.
    """
    ...

def read_frames_from_file(inp: str | os.PathLike[str], /) -> Iterator[Frame]:
    """Read frames from a file path.

    Parameters:
        inp: Path to the input file.

    Yields:
        Frame objects parsed from the file.
    """
    ...

def write_frame(fh: typing.BinaryIO, frame: Frame, /) -> int:
    """Serialize a frame.

    Parameters:
        fh: File handler returned by open
        frame: Frame object to serialize.
    Return: 
        number of bytes being write
    """
    ...

def write_frames(fh: typing.BinaryIO, frames: Iterator[Frame], /) -> int:
    """Serialize frames to a file handler.

    Parameters:
        fh: File handler returned by open
        frames: iterator of Frames
    Return: 
        number of bytes being write
    """
    ...
