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

def load_speech_signal(speech_file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    The function for loading speech signal
    """
    # check datatype
    speech_filename = Path(speech_file_path)
    if speech_file_path.suffix != ".wav":
        logger.warning("The speech data format is not a wav files")

    # load wav signal
    speech_rate, speech_signal = wavfile.read(speech_filename)

    # checking datatype
    i = np.iinfo(np.dtype(speech_signal.dtype))
    abs_max = pow(2, (i.bits - 1))
    offset = i.min + abs_max

    speech_signal = (speech_signal.astype(speech_signal.dtype) - offset) / abs_max

    return speech_rate, speech_signal

def write_speech_signal(output_filename: str, speech_rate: int, speech_data: np.ndarray) -> None:
    """
    The function to write speech data into wav file
    """
    wavfile.write(output_filename, speech_rate, speech_data.astype(np.float64))

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
    ir_right = ir_data[:, 1, :]

    return ir_left, ir_right

def filter_ir(signal: ndarray, ir_left: ndarray, ir_right: ndarray) -> np.ndarray:
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

    previous_created_data = [x for x in output_dir.glob('**/*') if x.is_file()]

    return previous_created_data

@hydra.main(config_path="", config_name="config", version_base=None)
def create_speech_data(cfg: DictConfig) -> None:
    """
    The main function of this script
    """
    logger.info("Creating speech dataset \n")
    logger.info(f"Processing speech data from: {cfg.path.speech_dir}")

    # check previous created data
    previous_data = check_previous_created_data(cfg.path.output_dir)
    if len(previous_data) > 0:
        logger.warning(f"Found {len(previous_data)} created samples \n")

    # create speech data with defined scenes
    ir_dir = Path(cfg.path.impulse_response_dir).glob('**/*')
    ir_files = [x for x in ir_dir if x.is_file()]

    logger.info(f"Generating acoustic scene with impulse response (IR) on: {cfg.path.impulse_response_dir}")
    logger.info(f"There are {len(ir_files)} IR files found")

    # check previous existed data
    previous_samples = check_previous_created_data(cfg.path.output_dir)

    for ir_path in ir_files:
        # define ir scenes name
        ir_scenes = str.split(str(ir_path), sep='/')[-1][:-14]

        # get impulse response
        ir_left, ir_right = load_impulse_response(ir_path)

        ir_degree = np.linspace(-90, 90,len(ir_left))
        ir_degree_names = [ str(int(i)) for i in ir_degree]

        # define which acoustic scenes that will be generated
        ir_degree_create = [
            '-90',
            "-45",
            "0",
            "45",
            "90"
        ]

        # 2 - itaration on degree
        ir_degree_idx = []
        for degree in ir_degree_create:
            index = ir_degree_names.index(degree)
            ir_degree_idx.append(index)

        # 3 - iteration on speakers
        speaker_dir = Path(cfg.path.speech_dir).glob('**/*')
        speaker_list = [str.split(str(x), sep='/')[-1] for x in speaker_dir if x.is_dir()]

        speaker_create = [
            "fena",
            "mmht"
        ]

        speaker_idx = []
        for speaker in speaker_create:
            index = speaker_list.index(speaker)
            speaker_idx.append(index)

        for degree in ir_degree_idx:
            for speaker in speaker_idx:
                speaker_scenes = speaker_list[speaker]
                scene_path = Path(cfg.path.output_dir) / ir_scenes / ir_degree_names[degree] / speaker_scenes
                scene_path.mkdir(parents=True, exist_ok=True)

                logger.info(f"Creating {ir_scenes} at {ir_degree_names[degree]} degree for {speaker_scenes}")

                speech_path = Path(cfg.path.speech_dir) / speaker_scenes
                speech_list = [x for x in speech_path.glob('**/*') if x.is_file()]

                for idx, speech_path in enumerate(speech_list, 1):

                    # checking output data
                    speech_name = str.split(str(speech_path), sep='/')[-1]
                    speech_number = speech_name[-8:-4]
                    output_path = scene_path / speech_name

                    if output_path in previous_samples:
                        logger.info(f"Skipping {ir_scenes} - {ir_degree_names[degree]} - {speaker_scenes} - speech number {speech_number} [{idx}/{len(speech_list)}]")
                        continue

                    logger.info(f"Processing {ir_scenes} - {ir_degree_names[degree]} - {speaker_scenes} - speech number {speech_number} [{idx}/{len(speech_list)}]")

                    # load signal data
                    rate, speech = load_speech_signal(speech_path)

                    # process speech data to create reverberant signal
                    speech_rev_left, speech_rev_right = filter_ir(speech, ir_left[degree,:], ir_right[degree,:])
                    speech_rev = np.array([speech_rev_left, speech_rev_right]).T

                    # write wav data
                    write_speech_signal(output_path, rate, speech_rev)

if __name__ == "__main__":
    create_speech_data()