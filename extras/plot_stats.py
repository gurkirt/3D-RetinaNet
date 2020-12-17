
# import multiprocessing as mp
import os
import json
import numpy as np
import math
def run_exp(cmd):
    return os.system(cmd)

def filter_counts(counts_, labels):
    new_counts = np.zeros(len(labels))
    for l in counts_:
        if l in labels:
            new_counts[labels.index(l)] = counts_[l]
    return new_counts

if __name__ == '__main__':
    count_file = 'instance_counts.json'
    with open(count_file,'r') as fff:
        counts = json.load(fff)
    
    anno_file  =  'annots_12fps_full_v1.0.json'

    with open(anno_file,'r') as fff:
        final_annots = json.load(fff)

    logger = open('classwise_counts.tex','w')
    result_modes = ['box','tube']
    label_types = ['av_actions','agent','action','loc'] #'loc', 'duplex','triplet']
    new_labels = {'Ped':'Pedestrian','Tl':'TrafficLight','Cyc': 'Cyclist',
                   'Medveh':'MediumVehicle','Othtl':'OtherTrafficLight',
                   'Larveh':'LargeVehicle','Mobike':'MotorBike','Emveh':'EmergencyVehicle',
                   'Movtow':'MovTowards','Wait2x':'Waiting2Cross','Mov':'Move',
                   'Turrht':'TurnRight','Turlft':'TurnLeft','Incatrht':'IndicateRinght',
                   'Incatlft':'IndicateLeft','Pushobj':'PushObject','Xing':'ToCross',
                   'Hazlit':'HazardLightsOn','Ovtak':'OverTake',
                   'Lftpav':'LeftPavement','Vehlane':'VehicaleLane',
                   'Rhtpav':'RightPavement','Jun':'Junction','Pav':'Pavement',
                   'Outgolane':'OutgoLane','Incomlane':'Incominglane',
                   'xing':'AtCrossing','Outgocyclane':'OutgoCycLane',
                   'Incomcyclane':'IncomingCycLane','Busstop':'BusStop',
                   }
    table = ''
    all_labels = []
    video_counts = []
    frame_counts = []
    total_count = 0
    total_count_used = 0
    for label_type in label_types:
        temp_label_type  = label_type
        if label_type == 'av_actions':
            temp_label_type = label_type[:-1]
            
        labels = final_annots[temp_label_type+'_labels']

        table += '\midrule\n \\multicolumn{2}{c}{' + '{:s} results \\\\ \n\\midrule\n'.format(label_type.capitalize())
            
        vfcounts = {}

        vfcounts['all'] = {}
        for rm in result_modes:
            vfcounts['all'][rm]= filter_counts(counts[rm]['all'][temp_label_type], labels)
            
        box_counts = counts['box']['all'][temp_label_type]

        sorted_counts = sorted(box_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_counts_0 = sorted_counts[0]
        sorted_counts.append(sorted_counts_0)
        
        del sorted_counts[0]
        
        for l in range(len(sorted_counts)):
            label = sorted_counts[l][0]
            if label == 'ztotal':
                print(np.sum(vfcounts[subset]['box']))
            if label not in labels: # and label != 'ztotal':
                continue
            if label != 'ztotal':
                ind = labels.index(label)
            table += '{:s} '.format(label.capitalize().ljust(25))
            for subset in ['all',]: # 'val_1','val_2','val_3','test']:
                if label == 'ztotal':
                    numa = np.sum(vfcounts[subset]['box'])
                    numb = np.sum(vfcounts[subset]['tube'])
                else:
                    numa = vfcounts[subset]['box'][ind]
                    numb = vfcounts[subset]['tube'][ind]
                frame_counts.append(float(numa))
                if numb<6:
                    numb = 6
                video_counts.append(float(numb))
                new_label = label
                if label != 'xing':
                    new_label = label.capitalize()
                if new_label in new_labels:
                    new_label = new_labels[new_label]
                all_labels.append(new_label)
                mystr = '& {:d}/{:d} '.format(int(numa), int(numb))
                print('{} {:d}/{:d} '.format(new_label, int(numa), int(numb)))
                
                table += mystr.ljust(15)

    logger.write(table)
    
    import matplotlib.pyplot as plt
    # plt.bars(all_labels, frame_counts)
    ind = np.arange(len(all_labels)) 
    width = 0.32
    fig, ax = plt.subplots()
    # p1 = ax.bar(ind, frame_counts, width)
    # p2 = ax.bar(ind, video_counts, width, bottom=frame_counts)
    ax.margins(0.005, 0.005)
    # ax.axis('off')
    ax.grid(axis='y',color='gray', linestyle='-.')
    p1 = ax.bar(ind-width/2, video_counts, width, bottom=0)
    p2 = ax.bar(ind+width/2, frame_counts, width, bottom=0)

    plt.ylabel('Number of instances')
    plt.title('Scores by group and gender')
    plt.xticks(ind, all_labels, rotation='vertical')
    
    plt.legend((p1[0], p2[0]), ('Video-level', 'Frame-level'))
    ax.set_yscale('log')
    ax.set_ylim(1,10**5.5)
    
    
    plt.box(on=None)
    # plt.yticks([0.01]+[10**n for n in np.arange(0, 5)])
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.29,top=0.99)
    plt.show()