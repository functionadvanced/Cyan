import pygame
import mido
import sys
import random

"""
This class is used to play music defined by arrays of note, time, and volume

Example:

a = MPlayer([60, 62, 64, 65, 67, 69, 71], [88]*7, [127, 50, 127, 50, 127, 50, 127])
a.play()

"""


class MPlayer:
    def __init__(self, note, time, volume=None):
        self.len = len(note)
        if len(note) != len(time):
            print("error: length mismatch! [Init MPlayer]")
        if volume == None:
            volume = [127] * self.len # default volumn
        if len(volume) != len(note):
            print("error: length mismatch! [Init MPlayer]")
        self.note = note        
        self.time = time
        self.volume = volume
    def play(self):
        # change the following lines to your own path
        currentPath = 'C:\\Users\\jupei\\Desktop\\BME595\\Project\\'
        newMidiName = 'new_song.mid'
        # construct new midi
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message('program_change', program=0, time=0))
        for i, _ in enumerate(self.note):
            track.append(mido.Message('note_on', note=self.note[i], velocity=self.volume[i], time=self.time[i]))
        track.append(mido.Message('note_off', note=0, velocity=127, time=630))
        mid.save(currentPath + newMidiName) # save as midi file
        # play midi file
        pygame.init()
        pygame.mixer.music.load(currentPath + newMidiName)
        pygame.mixer.music.play()
        input("press to exit")


