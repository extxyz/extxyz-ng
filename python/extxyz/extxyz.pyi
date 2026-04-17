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

def read_frame_from_stream(stream: typing.TextIO) -> Frame:
    """Read a single frame from a text stream.

    Parameters:
        stream: A file-like text stream ("r") containing encoded frame data.

    Returns:
        A parsed Frame object.
    """
    ...

def read_frames_from_stream(stream: typing.TextIO) -> Iterator[Frame]:
    """Read frames from a text stream.

    Parameters:
        stream: A file-like text stream ("r") containing encoded frames.

    Yields:
        Frame objects parsed sequentially from the stream.
    """
    ...

def read_frame(inp: str | os.PathLike[str], /) -> Frame:
    """Read a frame from a file path.

    Parameters:
        inp: Path to the input file.

    Returns:
        A parsed Frame object.
    """
    ...

def read_frames(inp: str | os.PathLike[str], /) -> Iterator[Frame]:
    """Read frames from a file path.

    Parameters:
        inp: Path to the input file.

    Yields:
        Frame objects parsed from the file.
    """
    ...

def write_frame(frame: Frame, /) -> str | bytes | os.PathLike[str]:
    """Serialize a frame.

    Parameters:
        frame: Frame object to serialize.

    Returns:
        Serialized representation of the frame.
    """
    ...

def write_frames(inp: str | bytes | os.PathLike[str], /) -> Iterator[Frame]:
    """Read frames from an input source.

    Parameters:
        inp: Input file path or data source.

    Yields:
        Frame objects parsed from the input.
    """
    ...
