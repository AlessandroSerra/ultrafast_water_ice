from . import (
    c_parser,
    exyz2gpumd,
    lmp2gpumd,
    rdf,
    spectral,
    temps,
    unwrap_coords,
    wignerEXC,
    wignerEXC2B,
    writeGPUMDdump,
)
from .exyz2gpumd import read_exyz, write_gpumd
from .rdf import (
    compute_I2,
    compute_R1,
    compute_rdf,
    propagate_I2_error,
    propagate_R1_error,
)
from .spectral import calculateVACFgroup, calculateVDOS

# Expose main functions for better IntelliSense
from .temps import analyzeTEMPS
from .unwrap_coords import unwrap_coords as unwrap_coordinates
from .writeGPUMDdump import writeXYZ

__all__ = [
    # Modules
    "spectral",
    "unwrap_coords",
    "wignerEXC",
    "wignerEXC2B",
    "lmp2gpumd",
    "c_parser",
    "temps",
    "rdf",
    "exyz2gpumd",
    "writeGPUMDdump",
    # Functions
    "analyzeTEMPS",
    "compute_rdf",
    "compute_R1",
    "compute_I2",
    "propagate_R1_error",
    "propagate_I2_error",
    "calculateVACFgroup",
    "calculateVDOS",
    "unwrap_coordinates",
    "read_exyz",
    "write_gpumd",
    "writeXYZ",
]
