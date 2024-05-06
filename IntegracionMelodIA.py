import torch
import torch.nn as nn
import math
import music21 
import matplotlib.pyplot as plt
import numpy as np
import PositionalEncoding
from HarmonIA import HarmonIA
from MelodIA import MelodIA

class Integracion:
    def __init__(self):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.bach_works = music21.corpus.search('bach')
        self.note_dictionary = []
        self.create_note_dictionary()
        self.vocab_size = len(self.note_dictionary)
        self.works_list = []
        self.all_Works_To_List()
        self.pairs = self.all_works_to_parts_pairs()

        self.combination = []

        self.harmonIA =HarmonIA(self.vocab_size,1024,16,6,2048,self.device, self.note_dictionary.index("<PAD>"))
        sd = torch.load('./MelodIA-Pyra-90-Bass-Compass.pth')
        self.harmonIA.load_state_dict(sd)
        self.melodIA =MelodIA(self.vocab_size,1024,8,4096,self.device, self.note_dictionary.index("<PAD>"))
        sd = torch.load('./MelodIA2-Har2Mel-25-Bass-4_0.pth')
        self.melodIA.load_state_dict(sd)
        print("Modelos creados con éxito")

    def create_note_dictionary(self):
        self.note_dictionary = [] 
        for work in self.bach_works:    
            work = work.parse()
            for workparts in work.parts:
                part_thing = len(workparts.recurse().getElementsByClass('Clef'))
                if part_thing > 0:
                    if workparts.recurse().getElementsByClass('Clef')[0].sign == 'F' or workparts.recurse().getElementsByClass('Clef')[0].sign == 'G':
                        for elmc in workparts.recurse():
                            if(isinstance(elmc,music21.note.Note) and (f'{elmc.pitch}|{elmc.duration.type}') not in self.note_dictionary):
                                self.note_dictionary.append(f'{elmc.pitch}|{elmc.duration.type}')
                            if(isinstance(elmc,music21.note.Rest) and (f'rest|{elmc.duration.type}' not in self.note_dictionary)):
                                self.note_dictionary.append(f'rest|{elmc.duration.type}')
                            elif(isinstance(elmc,music21.chord.Chord) and (f'{elmc.normalOrderString}|{elmc.duration.type}') not in self.note_dictionary):
                                self.note_dictionary.append(f'{elmc.normalOrderString}|{elmc.duration.type}')
                            elif(isinstance(elmc,music21.stream.Measure) and ( '|' not in self.note_dictionary)):
                                self.note_dictionary.append(f'|')
        self.note_dictionary.append("<BOS>")
        self.note_dictionary.append("<EOS>")
        self.note_dictionary.append("<PAD>")
        self.vocab_size = len(self.note_dictionary)
        return self.note_dictionary   
         
    def part_to_array(self, part):
        part_vector = [self.note_dictionary.index('<BOS>')]
        for elmc in part.recurse():
                if(isinstance(elmc,music21.note.Note)):
                    part_vector.append(self.note_dictionary.index(f'{elmc.pitch}|{elmc.duration.type}'))#note_dictionary[elmc.pitch.ps])
                if(isinstance(elmc,music21.chord.Chord)):
                    part_vector.append(self.note_dictionary.index(f'{elmc.normalOrderString}|{elmc.duration.type}'))#note_dictionary[elmc.normalOrderString])
                if(isinstance(elmc,music21.stream.Measure)):
                    part_vector.append(self.note_dictionary.index(f'|'))#note_dictionary[elmc.normalOrderString])
                if(isinstance(elmc,music21.note.Rest)):
                    part_vector.append(self.note_dictionary.index(f'rest|{elmc.duration.type}'))#note_dictionary[elmc.normalOrderString])

        part_vector.append(self.note_dictionary.index('<BOS>'))
        return part_vector
    
    def all_Works_To_List(self):
        self.works_list = []
        for work in self.bach_works:
            work = work.parse()
            for parts in work.parts:
                part_thing = len(parts.recurse().getElementsByClass('Clef'))
                if part_thing > 0:
                        if parts.recurse().getElementsByClass('Clef')[0].sign == 'F':
                            work_vector = self.part_to_array(parts)
                            self.works_list.append(work_vector)
        return self.works_list
    
    def mix_parts(self,parts_separated):
        self.combination = []
        for bass_part in parts_separated[0]:
            for trebble_part in parts_separated[1]:
                self.combination.append([bass_part,trebble_part])
        return self.combination
    
    def get_work_parts_pairs(self,work):
        try:
            Bass_parts = [part for part in work.parse().parts if part.recurse().getElementsByClass('Clef')[0].sign == 'F']
    
            Trebble_parts = [part for part in work.parse().parts if part.recurse().getElementsByClass('Clef')[0].sign == 'G']
        except:
            return [None,None]
        bass_token_parts = []
        trebble_token_parts = []
        if (len(Bass_parts)>0 and len(Trebble_parts) > 0):
            for bass_part in Bass_parts:
                bass_token_parts.append(self.part_to_array(bass_part))
            for trebble_part in Trebble_parts:
                trebble_token_parts.append(self.part_to_array(trebble_part))
        else:
            return [None,None]
        bass_trebble_array = [bass_token_parts,trebble_token_parts]
        # part_arr = part_to_array(Bass_parts[0],note_dictionary)
        return bass_trebble_array
    
    def all_works_to_parts_pairs(self):
        work_pairs = []
        for work in self.bach_works:
            parts_separated = self.get_work_parts_pairs(work)
            if parts_separated[0] is None : continue
            for pair in self.mix_parts(self.get_work_parts_pairs(work)):
                work_pairs.append(pair)
        return work_pairs
   
        
    def gen_harmony(self, notes_to_generate):
        print("Generando Armonias")
        self.harmonIA.eval()
        src = self.pairs[0][0][:5]
        tgt = torch.tensor([src],device= self.device)
        notes_generated = 0
        while notes_generated < notes_to_generate:
            logits = self.harmonIA(tgt)
            out = logits[0]
            # print(out.topk(1))
            tokens = out.topk(2)[-1][-1]
            randint = torch.randint(0,2,(1,)).item()
            next_token = tokens[0].item()
            tgt = torch.cat((tgt,torch.tensor([[next_token]],device = self.device)), dim=1)
            if next_token != 0:
                notes_generated = notes_generated + 1 
        return tgt
    def tokens_to_notes(self, tensor_tokens):
        note_list = []
        for note in tensor_tokens[0]:
            note_list.append(self.note_dictionary[note.item()])
        note_list = note_list[1:]
        return note_list
    def extract_measures(self, harmony):
        harmony_wo_measures =[note for note in harmony if note != "|"]
        return harmony_wo_measures
    def hex_to_notes(self, hex_string):
        """
        Convert a hexadecimal-like string, where digits represent musical notes,
        into an array of note names.
    
        0-9 are the same, A corresponds to 10, B to 11.
    
        :param hex_string: A string representing musical notes as hexadecimal digits.
        :return: An array of note names.
        """
        # Mapping from hexadecimal digit to note name
        note_mapping = {
            '0': 'C', '1': 'C#',
            '2': 'D', '3': 'D#',
            '4': 'E', '5': 'F',
            '6': 'F#', '7': 'G',
            '8': 'G#', '9': 'A',
            'A': 'A#', 'B': 'B'
        }
        hex_string = hex_string[1:-1]
        # Convert each digit in the hex string to its corresponding note
        notes = [note_mapping[digit] for digit in hex_string.upper()]
    
        return notes
    def get_elm_to_append(self, elm):
        note_name,note_len = elm.split('|')
        note_to_append = None
        # print(note_len)
        if note_name=="rest":
            note_to_append = music21.note.Rest(type = note_len)
        elif '<' in note_name:
            note_to_append = music21.chord.Chord(self.hex_to_notes(note_name), type = note_len)
        else:
            # print(note_name)
            note_to_append = music21.note.Note(note_name, type = note_len)
        
        return note_to_append
        
    def gen_melody(self,notes_to_generate, lyrics):
        self.melodIA.eval()
        src = self.gen_harmony(notes_to_generate)
        print("Generando Melodias")
        tgt_arr =  self.pairs[0][1][:5]
        tgt = torch.tensor([tgt_arr], device= self.device)
        notes_generated = 0
        while notes_generated < notes_to_generate:
            logits = self.melodIA(src, tgt)
            out = logits[-1][-1]
            tokens = out.topk(1)[-1][-1]
            next_token = tokens.item()
            tgt = torch.cat((tgt,torch.tensor([[next_token]],device = self.device)), dim=1)
            if next_token != 0:    
                notes_generated = notes_generated + 1 
        print(tgt)
        harmony_notes_string = self.tokens_to_notes(src)
        melody_note_string = self.tokens_to_notes(tgt)
        harmony_notes_string = self.extract_measures(harmony_notes_string)
        melody_note_string = self.extract_measures(melody_note_string)


        myScore = music21.stream.Score()
        part = music21.stream.Part()
        voice = music21.stream.Voice()
        bass_vocie = music21.stream.Voice()
        bass_line = music21.stream.Part()
        for lyric,pair in zip(lyrics, melody_note_string):
            # print(pair)
            note_to_append = self.get_elm_to_append(pair)
            note_to_append.lyric=lyric if lyric != None else ""
            voice.append(note_to_append)
        for pair in harmony_notes_string:
            note_to_append = self.get_elm_to_append(pair)
            bass_vocie.append(note_to_append)
        part.append(voice)
        bass_line.append(bass_vocie)
        myScore.insert(0, part) 
        myScore.insert(0, bass_line) 
        print("Creando Archivo")
        myScore.write('xml',"../PartiruraGenerada.xml")
        
        return open('../PartiruraGenerada.xml', 'rb')
    