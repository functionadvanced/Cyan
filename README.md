# Cyan: A Melody Generator Powered by Deep Learning

<img src="https://github.com/functionadvanced/Cyan/blob/master/piano.jpg?raw=true" alt="drawing" width="100%" height="300px"/>

## Notice
This is our final project for the course BME 595 of Purdue University. We are still working on it now. We plan to finished it before 2019.

Some midi files are downloaded from the Internet. If there is any copyright infringement, please contact us for deletion. Email: jupeizhong6@gmail.com

## Objectives
1. Input a certian chord randomly, output a short melody that corresponds to that chord.
2. Learn how to produce a music-like sequences of chords.
3. Learn the variations of chords and add features in Phase 1 and 2 to generate an entire music.

## Team members
Peizhong Ju (github account: functionadvanced), Ziyu Gong (github account: BillyGong)

## Neural Network Structures
1. Bidirectional LSTM layer for each melody segments. <img src="https://latex.codecogs.com/gif.latex?\overrightarrow{h_T}" title="\overrightarrow{h_T}" /> and <img src="https://latex.codecogs.com/gif.latex?\overleftarrow{h_T}" title="\overleftarrow{h_T}" />, then the two vectors are concatenated to from the vector <img src="https://latex.codecogs.com/gif.latex?h_T" title="h_T" />.
2. The resulting <img src="https://latex.codecogs.com/gif.latex?h_T" title="h_T" /> vectors are then feed into a VAE model which represets the <img src="https://latex.codecogs.com/gif.latex?h_T" title="h_T" /> vector in latent space and then generate a new sample <img src="https://latex.codecogs.com/gif.latex?h_T" title="h_T" /> vector.
3. The final layer is a single layer of LSTM that took the newly generated <img src="https://latex.codecogs.com/gif.latex?h_T" title="h_T" /> vector and generates the melogy.

### LSTM models (GRU units)
1. Input are short melody segments that corresponds to a simple chord (C#, B, etc.)
2. The output of the LSTM are treated as an independent 
3. Might use GRU units in terms of the faster convergence rate and easier implementation (guess)

### VAE models
1. The <img src="https://latex.codecogs.com/gif.latex?h_T" title="h_T" /> vector then goes through two linear layer which output mean <img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /> and log of variance <img src="https://latex.codecogs.com/gif.latex?\log\sigma^2" title="\log\sigma^2" />
2. Create a sample using the std and mean of from the previous layer <img src="https://latex.codecogs.com/gif.latex?s&space;=&space;\mu&space;&plus;&space;\sigma*\epsilon,\&space;\epsilon\sim&space;N(0,1)" title="s = \mu + \sigma*\epsilon,\ \epsilon\sim N(0,1)" /> to create the new <img src="https://latex.codecogs.com/gif.latex?h_T" title="h_T" /> vector

## Problems/Challenges:
1. Get MIDI files and turns them into our data sets.
2. Definition of variation of chords.

## Tools we use:
1. An open source deep learning platform [Pytorch](https://pytorch.org/).
2. Use the python library [mido](https://mido.readthedocs.io/en/latest/index.html) to read, phrase, and create midi file.
3. Use the python library [pygame.mixer.music](https://www.pygame.org/docs/ref/music.html) to play the midi file.
