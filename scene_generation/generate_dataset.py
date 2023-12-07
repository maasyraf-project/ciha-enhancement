# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import sofa

from pathlib import Path
import hydra
import logging
from  omegaconf import DictConfig

logger = logging.getLogger(__name__)

def load_impulse_response(ir_file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    The function for load impulse response signal
    """
    # check datatype
    ir_filename = Path(ir_file_path)
    if ir_filename.suffix != ".sofa":
        logger.error("The impulse response data format is not a sofa files")

    # load impulse response
    ir = sofa.Database.open(ir_filename)

    # get values for impulser response for left and right channels
    ir_data = ir.Data.IR.get_values()
    ir_left = ir_data[:, 0, :]
    ir_right = ir_data[:, 0, :]

    return ir_left, ir_right

def filter_ir(signal: ndarray, hrtf_left: ndarray, hertf_right: ndarray) -> ndarray:
    """
    The function for applying convolutive filter to get filtered (reverberant) speech signals
    """
    # normalize the signal
    signal = signal / np.max(np.abs(signal))

    # apply fft convolution
    signal_rev_left = fftconvolve(signal, ir_left, mode="full")
    signal_rev_right = fftconvolve(signal, ir_right, mode="full")

    return signal_rev_left, signal_rev_right

def add_noise(signal: ndarray, noise: ndarray, global_snr) -> ndarray:
    """
    The function for adding diffuse noise to speech signal (still not used in this project!)
    """

    return signal_noisy_left, signal_noisy_right

def check_previous_created_data(output_dir: str | Path) -> list[str]:
    """
    The function for checking the data which have been created
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if not output_dir.exists():
        return []

    previous_created_data = []
    for i in output_dir.glob("train/"):
        previous_created_data.append(i.name)

    return previous_created_data

@hydra.main(config_path="", config_name="config", version_base=None)
def create_speech_data(cfg: DictConfig) -> None:
    """
    The main function of this script
    """
    logger.info("Creating speech dataset \n")
    logger.info(f"Processing speech data from: {cfg.path.speech_dir}")
    print("Check")

    # check previous created data
    previous_data = check_previous_created_data(cfg.path.output_dir)
    if len(previous_data) > 0:
        logger.warning(f"Found {len(previous_data)} created samples \n")

    # create speech data with defined scenes

    # steps:
    # 1 - iteration for reverberant types
    files = Path(cfg.path.impulse_response_dir).glob('**/*')
    ir_files = [x for x in files if x.is_file()]

    logger.info(f"Generating acoustic scene with impulse response (IR) on: {cfg.path.impulse_response_dir}")
    logger.info(f"There are {len(ir_files)} IR files found")

    for ir_path in ir_files:
        # define ir scenes name
        ir_scenes = str.split(str(ir_path), sep='/')[-1][:-14]

        logger.info(f"Creating speech data for {ir_scenes} scenes")

        # get impulse response
        ir_left, ir_right = load_impulse_response(ir_path)

        ir_degree = np.linspace(-90, 90,len(ir_left))
        ir_degree_names = [ str(int(i)) for i in ir_degree]
        ir_degree_create = [
            '-90',
            "-60",
            "-45",
            "-30",
            "0",
            "30",
            "45",
            "60",
            "90"
        ]

        ir_degree_idx = []
        for degree in ir_degree_create:
            index = ir_degree_names.index(degree)
            ir_degree_idx.append(index)

        # apply filtering to signal
        # for degree in ir_left:




    # 2 - iteration for speech signals processing


if __name__ == "__main__":
    create_speech_data()