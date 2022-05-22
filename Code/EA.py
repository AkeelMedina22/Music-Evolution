import pretty_midi
import music21 as m21
import numpy as np
import random
import textdistance
import matplotlib.pyplot as plt
from midi import midi
from helpers import *

# Evolutionary Algorithm with variable parameters
class EA:

    def __init__(self, generations, p_size, num_offspring, mutation_rate):

        self.generations = generations
        self.p_size = p_size
        self.num_offspring = num_offspring
        self.mutation_rate = mutation_rate
        self.p_count = 0

        self.avg_fitness = []
        self.best_fitness = []
        self.maximization = True
        self.max_fitness_instance = {}
        self.max_fitness_pop = {}

        self.age = {}
        self.midi_instance = midi()
        self.midi_instance.corpus_descriptors()

    def run(self, fitness_func):

        population = self.initialize_population()
        for _ in range(self.generations):

            print(_)

            if fitness_func == 1:
                fitness_lst = self.fitness1(population)
            elif fitness_func == 2:
                fitness_lst = self.fitness2(population)
            elif fitness_func == 3:
                fitness_lst = self.fitness3(population)

            parents = self.binary_tournament(
                self.num_offspring, fitness_lst, self.maximization)
            offspring = self.crossover(parents, population)

            for i, j in offspring.items():
                if random.random() < self.mutation_rate:
                    self.mutation(i, j)

            if fitness_func == 1:
                new_fitness = self.fitness1(offspring)
            elif fitness_func == 2:
                new_fitness = self.fitness2(offspring)
            elif fitness_func == 3:
                new_fitness = self.fitness3(offspring)

            survivors = self.truncation(
                new_fitness, self.p_size, self.maximization)
            population = {i[0]: offspring[i[0]] for i in survivors}
            self.age = {i[0]: self.age[i[0]]+1 for i in survivors}

            self.best_fitness.append(max(new_fitness.values()))
            self.avg_fitness.append(
                sum(new_fitness.values())/len(new_fitness.values()))

        if fitness_func == 1:
            new_fitness = self.fitness1(population)
        elif fitness_func == 2:
            new_fitness = self.fitness2(population)
        elif fitness_func == 3:
            new_fitness = self.fitness3(population)
        self.max_fitness_pop = population[max(new_fitness, key=new_fitness.get)]


    def plot(self):

        print("Best Fitness:", max(self.best_fitness))
        plt.figure(figsize=(7, 4))
        plt.title('BT and Truncation')
        plt.plot(list(range(self.generations)),
                 self.best_fitness, color='salmon')
        plt.plot(list(range(self.generations)),
                 self.avg_fitness, color='darkslategrey')
        plt.legend(['Best Fitness', 'Average Fitness'])
        plt.xlabel('Generation Number')
        plt.ylabel('Fitness')
        plt.show()

    def gene_to_midi(self, path):

        score = []
        count = 0
        for i in self.max_fitness_pop:
            pitch = int(map(bin2float(i[0:64]), [0, 2**64], [30, 100]))
            ioi = np.float64(map(bin2float(i[64:128]), [0, 2**64], [0.5, 2]))
            duration = np.float64(
                map(bin2float(i[128:192]), [0, 2**64], [1, 4]))
            if count == 0:
                score.append([ioi, duration, pitch])
            else:
                score.append([ioi+score[count-1][0], duration, pitch])
            count += 1

        piano_c_chord = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program(
            'Acoustic Grand Piano')

        piano = pretty_midi.Instrument(program=piano_program)

        for i in score:
            piano.notes.append(pretty_midi.Note(
                velocity=int(127), pitch=i[2], start=i[0], end=i[0]+i[1]))
        piano_c_chord.instruments.append(piano)
        piano_c_chord.write(path)
        # self.create_score(path)

    def create_score(self, path):
        # https://stackoverflow.com/questions/63257434/preview-score-from-midi-file-in-python
        parsed = m21.converter.parse(path)
        parsed.show('musicxml.png')


    def initialize_population(self) -> dict({int: [list, ...]}):
        # Generates 'p_size' chromosomes, in the form of a random path with the input nodes.

        population = {}

        for _ in range(self.p_size):
            bins = []
            num = np.random.randint(np.round(self.midi_instance.note_mean-self.midi_instance.note_sd),
                                    np.round(self.midi_instance.note_mean+self.midi_instance.note_sd))
      
            for j in range(num):
                bin1 = float2bin(random.randint(0, 2**64))
                bin2 = float2bin(random.randint(0, 2**64))
                bin3 = float2bin(random.randint(0, 2**64))

                bins.append("{}{}{}".format(bin1, bin2, bin3))

            self.age.update({_:0})
            population.update({_: bins})
            self.p_count += 1
        return population

    def crossover(self, parents, population):

        for i in range(0, len(parents)-1, 2):

            parent1 = "".join(population[parents[i][0]])
            parent2 = "".join(population[parents[i+1][0]])

            path_length = min(len(parent1), len(parent2))
            crossover_length = path_length//2

            rand_point1 = random.randint(0, path_length-crossover_length)
            rand_point2 = random.randint(0, path_length-crossover_length)

            mid1 = parent1[rand_point1:rand_point1+crossover_length]
            mid2 = parent2[rand_point2:rand_point2+crossover_length]

            end1 = parent2[rand_point1+crossover_length:]
            start1 = parent2[:rand_point1]
            end2 = parent1[rand_point2+crossover_length:]
            start2 = parent1[:rand_point2]

            offspring1 = []
            offspring2 = []

            for _ in start1:
                offspring1.append(_)

            for _ in mid1:
                offspring1.append(_)

            for _ in end1:
                offspring1.append(_)

            for _ in start2:
                offspring2.append(_)

            for _ in mid2:
                offspring2.append(_)

            for _ in end2:
                offspring2.append(_)

            population.update({self.p_count: ["".join(
                offspring1[j:j+192]) for j in range(0, len("".join(offspring1)), 192)]})
            self.age.update({self.p_count: 0})
            self.p_count += 1
            population.update({self.p_count: ["".join(
                offspring2[j:j+192]) for j in range(0, len("".join(offspring2)), 192)]})
            self.age.update({self.p_count: 0})
            self.p_count += 1

        return population

    def mutation(self, count: int, path: list(list([float, float]))):

        # https://arxiv.org/ftp/arxiv/papers/1203/1203.3099.pdf

        path = "".join(path)

        i = random.randint(0, len(path)-(len(path)//2)-1)
        j = i + len(path)//2

        tmp = path[i:j]
        tmp = tmp[::-1]
        path = path[:i] + tmp + path[j:]

        return ["".join(path[j:j+192]) for j in range(0, len("".join(path)), 192)]

    def truncation(self, fitness_lst: dict({int: [float]}), size: int, maximization: bool) -> dict({int: [float]}):

        return [[k, v] for k, v in sorted(fitness_lst.items(), key=lambda item: item[1], reverse=maximization) if self.age[k]<=3][0:size]

    def binary_tournament(self, size: int, fitness_lst: dict({int: [float]}), maxi: bool) -> dict({int: [float]}):

        result = []

        for i in range(size):

            temp = [random.choice(list(fitness_lst.keys())),
                    random.choice(list(fitness_lst.keys()))]

            if not maxi:
                if fitness_lst[temp[0]] < fitness_lst[temp[1]]:
                    result.append([temp[0], fitness_lst[temp[0]]])
                else:
                    result.append([temp[1], fitness_lst[temp[1]]])
            else:
                if fitness_lst[temp[0]] > fitness_lst[temp[1]]:
                    result.append([temp[0], fitness_lst[temp[0]]])
                else:
                    result.append([temp[1], fitness_lst[temp[1]]])

        return result

    def fitness1(self, population: dict({int: [[float, float], ...]})) -> dict({int: float}):

        fitness_lst = {}

        for _ in population.keys():

            fitness = 0
            track_duration = 0
            score = []

            for i in population[_]:

                pitch = int(map(bin2float(i[0:64]), [0, 2**64], [30, 100]))
                ioi = float(map(bin2float(i[64:128]), [0, 2**64], [0.5, 2]))
                duration = float(
                    map(bin2float(i[128:192]), [0, 2**64], [1, 4]))
                x = [pitch, ioi, duration]

                score.append(x)
                track_duration += duration
            
            bin = 128

            indiv_pitch_hist = np.histogram([j[0] for j in score], bins=bin)[0]
            indiv_ioi_hist = np.histogram([j[1] for j in score], bins=bin)[0]
            indiv_duration_hist = np.histogram(
                [j[2] for j in score], bins=bin)[0]

            indiv_pitch_diff_hist = np.histogram(
                np.diff([j[0] for j in score]), bins=bin)[0]
            indiv_ioi_diff_hist = np.histogram(
                np.diff([j[1] for j in score]), bins=bin)[0]
            indiv_duration_diff_hist = np.histogram(
                np.diff([j[2] for j in score]), bins=bin)[0]

            indiv_pitch_fft = np.abs(np.fft.fft([j[0] for j in score], n=bin))
            indiv_ioi_fft = np.abs(np.fft.fft([j[1] for j in score], n=bin))
            indiv_duration_fft = np.abs(
                np.fft.fft([j[2] for j in score], n=bin))

            indiv_pitch_diff_fft = np.abs(np.fft.fft(
                np.diff([j[0] for j in score]), n=bin))
            indiv_ioi_diff_fft = np.abs(np.fft.fft(
                np.diff([j[1] for j in score]), n=bin))
            indiv_duration_diff_fft = np.abs(np.fft.fft(
                np.diff([j[2] for j in score]), n=bin))

            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = [
            ], [], [], [], [], [], [], [], [], [], [], []

            for i in range(self.midi_instance.corpus_size):

                c1.append(correlation_coefficient(
                    np.array(self.midi_instance.pitch_hist[i]), np.array(indiv_pitch_hist)))
                c2.append(correlation_coefficient(
                    np.array(self.midi_instance.ioi_hist[i]), np.array(indiv_ioi_hist)))
                c3.append(correlation_coefficient(
                    np.array(self.midi_instance.duration_hist[i]), np.array(indiv_duration_hist)))

                c4.append(correlation_coefficient(
                    np.array(self.midi_instance.pitch_diff_hist[i]), np.array(indiv_pitch_diff_hist)))
                c5.append(correlation_coefficient(
                    np.array(self.midi_instance.ioi_diff_hist[i]), np.array(indiv_ioi_diff_hist)))
                c6.append(correlation_coefficient(
                    np.array(self.midi_instance.duration_diff_hist[i]), np.array(indiv_duration_diff_hist)))

                c7.append(correlation_coefficient(
                    np.array(self.midi_instance.pitch_fft[i]), np.array(indiv_pitch_fft)))
                c8.append(correlation_coefficient(
                    np.array(self.midi_instance.ioi_fft[i]), np.array(indiv_ioi_fft)))
                c9.append(correlation_coefficient(
                    np.array(self.midi_instance.duration_fft[i]), np.array(indiv_duration_fft)))

                c10.append(correlation_coefficient(
                    np.array(self.midi_instance.pitch_diff_fft[i]), np.array(indiv_pitch_diff_fft)))
                c11.append(correlation_coefficient(
                    np.array(self.midi_instance.ioi_diff_fft[i]), np.array(indiv_ioi_diff_fft)))
                c12.append(correlation_coefficient(
                    np.array(self.midi_instance.duration_diff_fft[i]), np.array(indiv_duration_diff_fft)))

            x = len(population[_])
            y = track_duration

            correlation1 = np.mean([np.mean(c1), np.mean(c2), np.mean(c3), np.mean(c4), np.mean(c5), np.mean(c6), np.mean(c7), np.mean(c8), np.mean(c9), np.mean(
                c10), np.mean(c11), np.mean(c12)])

            correlation2 = np.mean([normalized_normal_dist(
                x, self.midi_instance.note_mean, self.midi_instance.note_sd, self.midi_instance.max_note), normalized_normal_dist(y, self.midi_instance. track_mean, self.midi_instance.track_sd, self.midi_instance.max_track)])

            fitness = max([correlation1, correlation2])

            fitness_lst.update({_: fitness})

        return fitness_lst

    def fitness2(self, population: dict({int: [[float, float], ...]})) -> dict({int: float}):

        fitness_lst = {}

        for _ in population.keys():

            track_duration = 0
            score = []

            for i in population[_]:

                pitch = int(map(bin2float(i[0:64]), [0, 2**64], [0, 127]))
                ioi = float(map(bin2float(i[64:128]), [0, 2**64], [0, 2]))
                duration = float(
                    map(bin2float(i[128:192]), [0,
                     2**64], [0, 4]))
                x = [pitch, ioi, duration]

                score.append(x)
                track_duration += score[-1][2]

            x = ""
            for j in score:
                x += "{} {} {} ".format(j[0], j[1], j[2])
            y = ""
            for j in self.midi_instance.score_lst[0]:
                y += "{} {} {} ".format(j[2], j[5], j[1])
            ncd_fitness = textdistance.entropy_ncd(x, y)

            fitness_lst.update({_:ncd_fitness})

        return fitness_lst

    def fitness3(self, population: dict({int: [[float, float], ...]})) -> dict({int: float}):
            
        fitness_lst = {}

        for _ in population.keys():

            track_duration = 0
            score = []

            for i in population[_]:

                pitch = int(map(bin2float(i[0:64]), [0, 2**64], [0, 127]))
                ioi = float(map(bin2float(i[64:128]), [0, 2**64], [0, 2]))
                duration = float(
                    map(bin2float(i[128:192]), [0, 2**64], [0, 4]))
                x = [pitch, ioi, duration]

                score.append(x)
                track_duration += score[-1][2]

            ratings = self.sub_rater(score, len(score))
            fitness_lst.update({_: np.mean(ratings)})

        return fitness_lst

    def sub_rater(self, score, num_notes):
        """
        sr_select: Integer to select sub rater
        1. Neighbouring pitch range
        2. Direction of melody
        3. Direction stability of melody
        4. Variety of note density - NOT DONE
        5. Syncopation Notes In Melody - NOT DONE
        6. Pitch Range in Melody
        7. Variety of Rest Note Density - NOT DONE
        8. Continuous Silence
        9. Unique Note Pitches
        10. Equal Consecutive Notes
        11. Unique Rhythm Values
        """
        sr_score = []

        cnc = 0  # crazy note count
        cnc1 = 0
        cnc2 = 0  # pitch change
        high = 0
        low = 9999
        longest_silence_interval = 0
        total_silence_interval = 0
        num_consecutive_pitch = 0
        unique_pitch = []
        unique_duration = []
        for i in range(1, len(self.midi_instance.score_lst)):
            if score[i][0] >= (score[i-1][0] + 16) or score[i][0] <= (score[i-1][0] - 16):
                cnc += 1
            if score[i][0] >= score[i-1][0]:
                cnc1 += 1
            if abs(score[i][0] - score[i - 1][0]) > 0:
                cnc2 += 1
            if score[i][0] > high:
                high = score[i][0]
            elif (score[i][0] < low) and (score[i][0] != 0):
                low = score[i][0]
            silence = abs(score[i][1] - score[i-1][2])
            if silence > longest_silence_interval:
                longest_silence_interval = silence
            total_silence_interval += silence
            if score[i][0] not in unique_pitch:
                unique_pitch.append(score[i][0])
            if score[i - 1][0] == score[i][0]:
                num_consecutive_pitch += 1
            if score[i][2] not in unique_duration:
                unique_duration.append(score[i][2])

        sr_score.append(cnc/num_notes)
        sr_score.append(cnc1/num_notes)
        sr_score.append(cnc2/num_notes)
        sr_score.append((high / low)/100)    
        sr_score.append(longest_silence_interval / total_silence_interval)      
        sr_score.append(len(unique_pitch) / num_notes)            
        sr_score.append(num_consecutive_pitch / num_notes)       
        sr_score.append(len(unique_duration) / num_notes)

        return sr_score