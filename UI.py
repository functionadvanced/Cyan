# Import a library of functions called 'pygame'
import pygame
import mido
import sys
import os
import time


def CreateEachNote(note):
    # construct new midi
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # note_list = [60, 62, 64, 65, 67, 69, 71] # C4~B4, detailed mapping see https://newt.phys.unsw.edu.au/jw/notes.html
    track.append(mido.Message('program_change', program=0, time=0))
    track.append(mido.Message('note_on', note=note, velocity=127, time=0))
    track.append(mido.Message('note_off', note=0, velocity=127, time=630))
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_name = "notes"
    savedModel_name = "N"+str(note)+".mid"
    model_path = os.path.join(dir_path, os.path.join(folder_name, savedModel_name))

    mid.save(model_path) # save as midi file
    # use the following website to convert .mid file to .wav file
    # https://www.conversion-tool.com/midi/ 


notes_list = []

def LoadAllNotes():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_name = "notes"
    for i in range(60):        
        savedModel_name = "N"+str(36+i)+".wav"
        model_path = os.path.join(dir_path, os.path.join(folder_name, savedModel_name))
        notes_list.append(pygame.mixer.Sound(model_path))
        

current_channel = 0
def PlayNote(note):
    global current_channel
    pygame.mixer.Channel(current_channel).play(notes_list[note-36])
    current_channel += 1
    if current_channel > 7:
        current_channel -= 8

# Initialize the game engine
pygame.mixer.init(frequency = 44100, size = -16, channels = 100, buffer = 2**12) 
pygame.init()
LoadAllNotes()

# note_list = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
# for note in note_list:
#     time.sleep(0.3)
#     PlayNote(note)
 
# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

color_overcast = "#F1F1F2"
color_warmGray = "#BCBABE"
color_ice = "#A1D6E6"
color_glacierBlue = "#1995AD"
 
# Set the height and width of the screen
size = [1800, 500]
screen = pygame.display.set_mode(size)
 
pygame.display.set_caption("Cyan")

#Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()
while not done:
 
    # This limits the while loop to a max of 10 times per second.
    # Leave this out and we will use all CPU we can.
    clock.tick(100)
     
    for event in pygame.event.get(): # User did something        
        if event.type == pygame.QUIT: # If user clicked close
            done=True # Flag that we are done so we exit this loop
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                PlayNote(77)
            if event.key == pygame.K_s:
                PlayNote(78)
            if event.key == pygame.K_d:
                PlayNote(79)
            if event.key == pygame.K_f:
                PlayNote(80)
            if event.key == pygame.K_g:
                PlayNote(81)
            if event.key == pygame.K_h:
                PlayNote(82)
            if event.key == pygame.K_j:
                PlayNote(83)
            if event.key == pygame.K_k:
                PlayNote(84)
            if event.key == pygame.K_l:
                PlayNote(85)    

    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_a]:
    #     PlayNote(44)
    # if keys[pygame.K_s]:
    #     PlayNote(45)
 
    # All drawing code happens after the for loop and but
    # inside the main while done==False loop.
     
    # Clear the screen and set the screen background
    screen.fill(pygame.Color(color_overcast))

    count = 2
    idx = 0
    for i in range(42):
        pygame.draw.rect(screen, pygame.Color(color_glacierBlue), [50+i*40, 10, 38, 200])
        pygame.draw.rect(screen, pygame.Color(color_warmGray), [52+i*40, 12, 32, 194])
        

    for i in range(42):
        if idx == count:
            count = 5 - count
            idx = 0
            continue
        idx += 1
        pygame.draw.rect(screen, pygame.Color(color_glacierBlue), [50+i*40+25, 10, 30, 125])
        pygame.draw.rect(screen, BLACK, [50+i*40+22, 10, 28, 120])

    
    
    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()
 
# Be IDLE friendly
pygame.quit()