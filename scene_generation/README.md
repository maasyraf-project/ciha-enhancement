# Create speech dataset for various scenes

## Datasets

In this project, there are three main database which will be processed.

1. Speech data
2. Impulse response data (specifically, we use binaural room impulse response, BRIR)

## Directory format

```text
ciha_dataset
├───clean_speech (monoaural)
|   ├───speaker_A
|   |    |  speech_1.wav
|   |    |  speech_2.wav
|   |    |  ...
|   ├───speaker_B
|   |    |  speech_1.wav
|   |    |  speech_2.wav

├───impulse_response
|   | IR_Data_A.wav
|   | IR_Data_B.wav
|   | ...

├───generated_data
|   ├───IR_A
|   |       ├───speaker_A
|   |       |   |  speech_rev_1.wav
|   |       |   |  speech_rev_2.wav
|   |       |   | ...
|   |       ├───speaker_B
|   |       |   |  speech_rev_1.wav
|   |       |   |  speech_rev_2.wav
|   |       |   | ...
|   |       ├───...

|   ├───IR_B
|   |       ├───...
|   |       |   |  ...

|   ├───...

```