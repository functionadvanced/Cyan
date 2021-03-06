import mido
import os

class Point:
    def __init__(self, note, velocity, time, isLeft, delta_time = 0):
        self.note = note
        self.velocity = velocity
        self.time = time
        self.delta_time = delta_time
        self.isLeft = isLeft    
    def __str__(self):
        temp = 'left'
        if not self.isLeft:
            temp = 'right'
        return "note="+str(self.note)+" velocity="+str(self.velocity)+" time="+str(self.time)+" delta_time="+str(self.delta_time)+" "+temp

def PlayList(point_list, newMidiname="temp.mid", play=True, mode=0):
    '''
    mode=0: play both hands
    mode=1: play left hand
    mode=2: play right hand
    '''
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('program_change', program=0, time=0))
    
    for m in point_list:
        if m.isLeft:
            if mode == 2:
                continue
        else:
            if mode == 1:
                continue
        # one second is 960 ticks
        track.append(mido.Message('note_on', note=m.note, velocity=m.velocity, time=int(m.delta_time*960)))
        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filePath = os.path.join(dir_path, 'midi_files', newMidiname)
    mid.save(filePath) # save as midi file
    if play:
        import pygame
        pygame.init()
        pygame.mixer.music.load(filePath)
        pygame.mixer.music.play()
        input("press to exit")

class PointList:
    def __init__(self, l_file_name, r_file_name=None):
        self.list = []
        self.add(l_file_name, True)
        if r_file_name != None:
            self.add(r_file_name, False)
        self.list.sort(key=lambda x: x.time)
        self.calDeltaTime()

    def calDeltaTime(self):
        for i in range(len(self.list)-1):
            self.list[i+1].delta_time = self.list[i+1].time - self.list[i].time

    def add(self, file_name, isLeft):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filePath = os.path.join(dir_path, 'midi_files', file_name)
        mid = mido.MidiFile(filePath)
        current_time = 0
        for msg in mid:
            if msg.time != 0 or msg.type == 'note_on':
                current_time += msg.time
                if msg.type == 'note_on':
                    self.list.append(Point(msg.note, msg.velocity, current_time, isLeft, msg.time))

    def saveAsMidi(self, newMidiname, play=False, mode=0):
        if play:
            PlayList(self.list, newMidiname=newMidiname, play=play, mode=mode)
        
    
# pl = PointList('1-l.mid', '1-r.mid')
# pl.saveAsMidi('temp.mid', play=True)