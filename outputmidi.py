from __future__ import division
import os
import numpy as np
from pretty_midi import PrettyMIDI, Instrument
import fmeasure
from midiio import MidiIO 

from scoreevent import Note

#Creates a midi file given a valid binary file and file path
def binary_to_midi(Y,output_path,numpy=False):
    # Y needs to by a n by 51 matrix
    # output_path isa string that needs to end with a filename.mid
    # ex Y = [[0,0,...0],[0,0,...1].....], output_path='test.mid'
    Notes = []
    midio = MidiIO(output_path)
    prev_i = Y[1]
    t =0
    note_times = [[] for i in range(51)]
    if numpy:
        Y = Y.tolist()
    for i in Y:
        
        for j in range(1,51):
            Yj = i[j]
            prev_j = prev_i[j]
            
            if Yj == 1 and prev_j==0:
                note_times[j].append([t,t])
            if Yj == 0 and prev_j ==1:
                note_times[j][-1][-1] = t


        prev_i = i
        t+=0.120
    for i in range(0,51):
        for j in note_times[i]:
            onset=j[0]
            offset=j[1]
            octave = int(i/12)+1
            p = i%12
            pname = Note.pitch_classes[p]
            note = Note(
                        pname, octave,
                        onset_ts=(onset),
                        offset_ts=(offset),
                    )
            Notes.append(note)
    
    midio.write_midi(Notes,24)
    return Notes

#Just used for testing, it's not perfect by any means. Don't use it. 
def midi_to_binary(y):
    timeconvert = 1/0.12;
    length_song = y[-1].offset_ts*timeconvert
    yp = [[0 for a in range(51)] for b in range( int(length_song))]

    for i in y:
        #Notes
        pname = i.pname
        octave = i.oct-1
        pindex= 0
        for j in i.pitch_classes:
            if j==pname:
                break
            pindex+=1
        pindex += octave*12 #get the index in the binary vector
        on = int(i.onset_ts *timeconvert)
        off = int(i.offset_ts * timeconvert)
        for j in range (on, off):
            yp[j][pindex] = 1
        
    return yp

#design for windows
if __name__ == '__main__':
    opath = ''
    if os.name=='nt':
        opath = os.getcwd()+"\\test.mid"
    else:
        opath = os.getcwd()+"/test.mid"
    label_path = 'beatles_herecomesthesun.mid'
    m = MidiIO(label_path)
    y = m.parse_midi()
    y = midi_to_binary(y)
    binary_to_midi(y,opath)
    Y = MidiIO(opath).parse_midi()
    Y = midi_to_binary(Y)
    Y.append([0]*51)
    print(fmeasure.fmeasure(y,Y)) #about 89% accurate since the midi-to-bin isn't perfect and doesn't need to be used at all.
    
