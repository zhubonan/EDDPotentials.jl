"""ASE wrapper used by the EDDPotentials airsspy MatterSim example."""

import os

from mattersim.forcefield import MatterSimCalculator


class EnvMatterSimCalculator(MatterSimCalculator):
    """Load MatterSim from environment variables for airsspy's ASE ML runner."""

    def __init__(self, load_path=None, **kwargs):
        load_path = load_path or os.environ["MATTERSIM_CHECKPOINT"]
        device = os.environ.get("MATTERSIM_DEVICE", "cpu")
        super().__init__(load_path=load_path, device=device, **kwargs)
