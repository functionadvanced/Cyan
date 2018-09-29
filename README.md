# Cyan
A melody generator powered by DL.

<img src="https://github.com/functionadvanced/Cyan/blob/master/piano.jpg?raw=true" alt="drawing" width="100%"/>

## Objectives
1. Input a certian chord randomly, output a short melody that corresponds to that chord.
2. Learn hoe to produce a music-like sequences of chords.
3. Learn the variations of chords and add features in Phase 1 and 2 to generate an entire music.

## Variantional autoencoder
Datasets: Melody segments that has labels as their corresponding chords.
Latent variables can be constructed as different chords.
Outputs: short melodys

### Neural network structures
1. Bidirectional LSTM layer for each melody segments. \(\overrightarrow{h_T}\) and \(\overleftarrow{h_T}\), then the two vectors are concatenated to from the vector \(h_t\).
2. The vector than go through two linear layer which output mean \(\mu\) and log of variance \(\log\sigma^2\)
3. Create a sample using the std and mean of from the previous layer \(s = \mu + \sigma*\epsilon, \epsilon\sim N(0,1)\)
4. The sample then become the input of two linear layer that becames the vector \(h_o\)
5. The final layer is a single layer of LSTM that finally generates the melogy.

## Problems:
1. Get MIDI files and turns them into our data sets.
2. Definition of variation of chords.

## What has been done:
### [CreateDataSet.py](https://github.com/functionadvanced/Cyan/blob/master/CreateDataSet.py)
1. Use the python library [mido](https://mido.readthedocs.io/en/latest/index.html) to read, phrase, and create midi file.
2. Use the python library [pygame.mixer.music](https://www.pygame.org/docs/ref/music.html) to play the midi file.
### [beethoven_opus10_1_format0.mid](https://github.com/functionadvanced/Cyan/blob/master/beethoven_opus10_1_format0.mid)
Sample midi file. (Downloaded from the Internet. If there is any copyright infringement, please contact us for deletion. Email: jupeizhong6@gmail.com)
