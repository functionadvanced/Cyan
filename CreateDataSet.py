import pygame
import mido

# change the following line to your own path of midi file
fileName = r'C:\Users\jupei\Desktop\BME595\Project\beethoven_opus10_1_format0.mid'

mid = mido.MidiFile(fileName)
for msg in mid:
    print(msg) # print each entry of this midi file

# play midi sound
pygame.init()
pygame.mixer.music.load(fileName)
pygame.mixer.music.play()
input("press to exit")