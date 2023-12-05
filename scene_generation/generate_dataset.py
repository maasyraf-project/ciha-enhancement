# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from scipy.io import wavfile

import hydra
import logging
from  omegaconf import DictConfig

logger = logging.getLogger(__name__)

@hydra.main(config_path="", config_name="config", version_base=None)
def create_speech_data(cfg: DictConfig) -> None:
    """
    The main function of this script
    """
    logger.info("Creating speech dataset \n")
    logger.info(f"Processing speech data on: {cfg.path.impulse_response_dir}")
    logger.info(f"Processing speech data from: {cfg.path.speech_dir}")
    print("Check")



if __name__ == "__main__":
    create_speech_data()