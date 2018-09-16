# Cyan

A melody generator powered by DL.

## Objectives
### Phase 1:
Input a certian chord randomly, output a short melody that corresponds to that chord.

### Phase 2:
Learn hoe to produce a music-like sequences of chords.

### Phase 3:
Learn the variations of chords and add features in Phase 1 and 2 to generate an entire music.

## Variantional autoencoder
Datasets: Melody segments that has labels as their corresponding chords.
Latent variables can be constructed as different chords.
Outputs: short melodys

## Problems:
1. Get MIDI files and turns them into our data sets.
2. Definition of variation of chords.

## What has been done:
### [CreateDataSet.py](https://github.com/functionadvanced/Cyan/blob/master/CreateDataSet.py)
1. Use the python library [mido](https://mido.readthedocs.io/en/latest/index.html) to read and phrase midi file.
2. Use the python library [pygame.mixer.music](https://www.pygame.org/docs/ref/music.html) to play the midi file.
### [beethoven_opus10_1_format0.mid](https://github.com/functionadvanced/Cyan/blob/master/beethoven_opus10_1_format0.mid)
Sample midi file.
