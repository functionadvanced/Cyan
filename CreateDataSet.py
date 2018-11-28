import pygame
import mido
import sys
import random

# change the following lines to your own path
currentPath = 'C:\\Users\\jupei\\Desktop\\BME595\\Project\\'
midiFileName = 'beethoven_opus10_1_format0.mid'
newMidiName = 'new_song.mid'
logFileName = 'log.txt'

mid = mido.MidiFile(currentPath + midiFileName)
sys.stdout = open(currentPath + logFileName, 'w+', encoding='utf8')
'''
To see all meta message types: https://www.recordingblogs.com/wiki/midi-meta-messages
'''
for msg in mid:
    print(msg) # print each entry of this midi file

# construct new midi
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
note_list = [60, 62, 64, 65, 67, 69, 71] # C4~B4, detailed mapping see https://newt.phys.unsw.edu.au/jw/notes.html
track.append(mido.Message('program_change', program=0, time=0))
for i in note_list:
    track.append(mido.Message('note_on', note=i, velocity=127, time=88))
track.append(mido.Message('note_off', note=0, velocity=127, time=630))
for i, _ in enumerate(note_list):
    track.append(mido.Message('note_on', note=note_list[len(note_list)-i-1], velocity=127, time=88))
track.append(mido.Message('note_off', note=0, velocity=127, time=630))
for i in range(20):
    track.append(mido.Message('note_on', note=random.randint(50, 80), velocity=127, time=330))
    # track.append(mido.Message('note_on', note=random.randint(20, 50), velocity=127, time=0))
    # track.append(mido.Message('note_on', note=random.randint(80, 108), velocity=127, time=0))
track.append(mido.Message('note_off', note=0, velocity=127, time=630))
mid.save(currentPath + newMidiName) # save as midi file

# play midi file
pygame.init()
pygame.mixer.music.load(currentPath + newMidiName)
pygame.mixer.music.play()
input("press to exit")
