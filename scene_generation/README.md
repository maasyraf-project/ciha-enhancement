# Create speech dataset for various scenes

## Datasets

In this project, there are three main database which will be processed.

1. Speech data
2. Impulse response data (specifically, we use binaural room impulse response, BRIR)

## Impulse response index

The degree-index are listed as the followings

```text
0 - min90
1 - min85
2 - min80
3 - min75
4 - min70
5 - min65
6 - min 60
7 - min55
8 - min50
9 - min 45
10 - min40
11 - min35
12 - min 30
13 - min25
14 - min20
15 - min15
16 - min10
17 - min5
18 0
19 - 5
20 - 10
21 - 15
22 - 20
23 - 25
24 - 30
25 - 35
26 - 40
27 - 45
28 - 50
29 - 55
30  - 60
31 - 65
32 - 70
33 - 75
34 - 80
35 - 85
36 - 90
```

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
|   |       ├───deg A
|   |       |   ├───speaker_A
|   |       |   |   | speech_rev_1.wav
|   |       |   |   | speech_rev_2.wav
|   |       |   |   | ...

|   ├───IR_B
|   ├───...


```