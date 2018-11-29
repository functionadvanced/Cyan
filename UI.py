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
    global key_status
    key_status[note-36] = True
    pygame.mixer.Channel(current_channel).play(notes_list[note-36])
    current_channel += 1
    if current_channel > 7:
        current_channel -= 8

def ReleaseNote(note):
    global key_status
    key_status[note-36] = False

def GetKeyIdx(i, isWhite=True):
    a = int(i / 7)
    b = i % 7
    b *= 2
    if b > 4:
        b -= 1
    key_idx = a*12+b
    if not isWhite:
        key_idx += 1
    return key_idx

def DrawNoteNum():
    count = 2
    idx = 0
    for i in range(35):
        key_idx = GetKeyIdx(i)
        text_surface = basicfont.render(str(key_idx+36), True, pygame.Color(color_glacierBlue))
        screen.blit(text_surface, (50+i*40+10,  360))

    for i in range(35):
        if idx == count:
            count = 5 - count
            idx = 0
            continue
        idx += 1
        key_idx = GetKeyIdx(i, isWhite=False)
        text_surface = basicfont.render(str(key_idx+36), True, pygame.Color(color_overcast))
        screen.blit(text_surface, (50+i*40+29,  300))

def DrawNote():
    global key_status
    count = 2
    idx = 0
    for i in range(35):
        if not key_status[GetKeyIdx(i)]:
            pygame.draw.rect(screen, pygame.Color(color_glacierBlue), [50+i*40, 200, 38, 200])
        pygame.draw.rect(screen, pygame.Color(color_warmGray), [52+i*40, 202, 32, 194])
        

    for i in range(35):
        if idx == count:
            count = 5 - count
            idx = 0
            continue
        idx += 1
        if not key_status[GetKeyIdx(i, isWhite=False)]:
            pygame.draw.rect(screen, pygame.Color(color_glacierBlue), [50+i*40+25, 200, 30, 125])
        pygame.draw.rect(screen, BLACK, [50+i*40+22, 200, 28, 120])

def DrawInfo():
    # basicfont = pygame.font.SysFont(None, 60)
    text_surface = basicfont.render("CYAN (version 1. 0)", True, pygame.Color(color_glacierBlue))
    screen.blit(text_surface, (20,  20))
    text_surface = basicfont.render("A Melody Generator Powered by Deep Learning", True, pygame.Color(color_glacierBlue))
    screen.blit(text_surface, (20,  40))
    text_surface = basicfont.render("Made by: Peizhong Ju & Ziyu Gong", True, pygame.Color(color_glacierBlue))
    screen.blit(text_surface, (20,  60))
    text_surface = basicfont.render("Made with: Pytorch & Pygame", True, pygame.Color(color_glacierBlue))
    screen.blit(text_surface, (20,  80))
    text_surface = basicfont.render("More details on:", True, pygame.Color(color_glacierBlue))
    screen.blit(text_surface, (20,  100))
    text_surface = basicfont.render("https://github.com/functionadvanced/Cyan", True, pygame.Color(color_glacierBlue))
    screen.blit(text_surface, (20,  120))

def Mapping(event, func):
    if event.key == pygame.K_a:
        func(60)
    if event.key == pygame.K_w:
        func(61)
    if event.key == pygame.K_s:
        func(62)
    if event.key == pygame.K_e:
        func(63)
    if event.key == pygame.K_d:
        func(64)
    if event.key == pygame.K_f:
        func(65)
    if event.key == pygame.K_t:
        func(66)
    if event.key == pygame.K_g:
        func(67)
    if event.key == pygame.K_y:
        func(68)
    if event.key == pygame.K_h:
        func(69)
    if event.key == pygame.K_u:
        func(70)
    if event.key == pygame.K_j:
        func(71)
    if event.key == pygame.K_k:
        func(72)
    if event.key == pygame.K_o:
        func(73)
    if event.key == pygame.K_l:
        func(74)

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
size = [1500, 500]
screen = pygame.display.set_mode(size)
# DISPLAYSURF = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
 
pygame.display.set_caption("Cyan")

dir_path = os.path.dirname(os.path.realpath(__file__))
fontDir = os.path.join(dir_path, os.path.join("fonts", "JosefinSlab-Bold.ttf"))
basicfont = pygame.font.Font(fontDir, 15)

#Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()

key_status = []
for i in range(60):
    key_status.append(False)
while not done:
 
    # This limits the while loop to a max of 10 times per second.
    # Leave this out and we will use all CPU we can.
    clock.tick(50)
     
    for event in pygame.event.get(): # User did something        
        if event.type == pygame.QUIT: # If user clicked close
            done=True # Flag that we are done so we exit this loop
        if event.type == pygame.KEYDOWN:
            Mapping(event, PlayNote)
        if event.type == pygame.KEYUP:
            Mapping(event, ReleaseNote)
 
    # All drawing code happens after the for loop and but
    # inside the main while done==False loop.
     
    # Clear the screen and set the screen background
    screen.fill(pygame.Color(color_overcast))

    DrawNote()

    DrawNoteNum()

    DrawInfo()
    
    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()
 
# Be IDLE friendly
pygame.quit()