from .detector import MaxNumberOfPixelsPerAxis, NumberOfDetectors, NumberOfAxis
from typing import NewType

RepackChunkSizeDesc = NewType("ChunkSize", str)
InputFileName = NewType("FileNameToBeRepacked", str)
PreprocessedFileName = NewType("RepackedFileName", str)
ChunkNeeded = NewType("ChunkNeeded", bool)


def get_chunk_size_desc(
    chunk_needed: ChunkNeeded,
    number_of_pixels: MaxNumberOfPixelsPerAxis,
    number_of_detectors: NumberOfDetectors,
    number_of_axis: NumberOfAxis,
) -> RepackChunkSizeDesc:
    if chunk_needed:
        # TODO: Check if this is correct
        return RepackChunkSizeDesc(f"{number_of_pixels}x{number_of_axis*number_of_detectors}")
    else:
        return RepackChunkSizeDesc("NONE")


def preprocess_file(
        file_name: InputFileName,
        chunk_needed: ChunkNeeded,
        chunk_size: RepackChunkSizeDesc
    ) -> PreprocessedFileName:
    import subprocess
    if chunk_needed:
        suffix = "-rechunk.h5"
    else:
        suffix = "-nochunk.h5"

    processed_file_name = PreprocessedFileName(file_name.replace(".h5", suffix))

    subprocess.run(["h5repack", "-l", f"CHUNK={chunk_size}", file_name, processed_file_name])
