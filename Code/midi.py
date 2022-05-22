import os
import pretty_midi
import numpy as np
from helpers import *


# Process a midi file into necessary format
class midi:

    def __init__(self):

        self.pitch_hist = []
        self.ioi_hist = []
        self.duration_hist = []

        self.pitch_diff_hist = []
        self.ioi_diff_hist = []
        self.duration_diff_hist = []

        self.pitch_fft = []
        self.ioi_fft = []
        self.duration_fft = []

        self.pitch_diff_fft = []
        self.ioi_diff_fft = []
        self.duration_diff_fft = []

        self.pitch_avg = 0
        self.ioi_avg = 0
        self.duration_avg = 0
        self.track_length = 0
        self.num_tracks = 0
        self.num_notes = []
        self.track_lengths = []
        self.corpus_size = 0
        self.note_mean = 0
        self.note_sd = 0
        self.note_max = 0
        self.track_mean = 0
        self.track_sd = 0
        self.track_max = 0

        self.score_lst = []

    def midi_to_list(self, midi):

        if isinstance(midi, str):
            midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)
        elif isinstance(midi, pretty_midi.pretty_midi.PrettyMIDI):
            midi_data = midi
        else:
            raise RuntimeError(
                'midi must be a path to a midi file or pretty_midi.PrettyMIDI')
        score = []
        track_length = 0
        ioi_len = 0
        pitch_len = 0
        duration_len = 0
        total_count = 0
        for instrument in midi_data.instruments:
            count = 0
            for note in instrument.notes:
                start = note.start
                duration = note.end - start
                pitch = note.pitch
                velocity = note.velocity
                if count == 0:
                    ioi = start
                else:
                    ioi = start - score[count-1][0]
                score.append(
                    [start, duration, pitch, velocity, instrument.name, ioi])
                ioi_len += ioi
                pitch_len += pitch
                duration_len += duration
                count += 1
                total_count += 1
            track_length += score[-1][0]

        return score, track_length, ioi_len/total_count, pitch_len/total_count, duration_len/total_count

    def corpus_descriptors(self):

        # scan corpus of music for all midi files
        with os.scandir('corpus/') as it:
            for entry in it:
                if entry.name.endswith(".mid") and entry.is_file():
                    self.num_tracks += 1
                    # print(entry.name, entry.path)
                    fn = entry.path
                    midi_data = pretty_midi.PrettyMIDI(fn)
                    midi, self.track_length, self.ioi_avg, self.pitch_avg, self.duration_avg = self.midi_to_list(
                        midi_data)
                    self.score_lst.append(midi)
                    self.track_lengths.append(self.track_length)
                    self.num_notes.append(len(midi))

        self.corpus_size = len(self.score_lst)

        self.note_mean = np.mean(self.num_notes)
        self.note_sd = np.std(self.num_notes)
        self.track_mean = np.mean(self.track_lengths)
        self.track_sd = np.std(self.track_lengths)

        # calculate maximum value so that you can get normalized value
        self.max_note = max([normal_dist(i, self.note_mean, self.note_sd)
                             for i in range(int(self.note_mean-self.note_sd), int(self.note_mean+self.note_sd))])
        self.max_track = max([normal_dist(i, self.track_mean, self.track_sd)
                              for i in range(int(self.track_mean-self.track_sd), int(self.track_mean+self.track_sd))])

        bin = 128

        for i in self.score_lst:

            self.pitch_hist.append(np.histogram(
                [j[2] for j in i], bins=bin)[0])
            self.ioi_hist.append(np.histogram([j[5] for j in i], bins=bin)[0])
            self.duration_hist.append(
                np.histogram([j[1] for j in i], bins=bin)[0])

            self.pitch_diff_hist.append(np.histogram(
                np.diff([j[2] for j in i]), bins=bin)[0])
            self.ioi_diff_hist.append(np.histogram(
                np.diff([j[5] for j in i]), bins=bin)[0])
            self.duration_diff_hist.append(np.histogram(
                np.diff([j[1] for j in i]), bins=bin)[0])

            self.pitch_fft.append(np.abs(np.fft.fft([j[2] for j in i], n=bin)))
            self.ioi_fft.append(np.abs(np.fft.fft([j[5] for j in i], n=bin)))
            self.duration_fft.append(
                np.abs(np.fft.fft([j[1]for j in i], n=bin)))

            self.pitch_diff_fft.append(
                np.abs(np.fft.fft(np.diff([j[2] for j in i]), n=bin)))
            self.ioi_diff_fft.append(
                np.abs(np.fft.fft(np.diff([j[5] for j in i]), n=bin)))
            self.duration_diff_fft.append(
                np.abs(np.fft.fft(np.diff([j[1] for j in i]), n=bin)))
                