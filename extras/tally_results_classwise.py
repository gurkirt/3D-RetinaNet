
# import multiprocessing as mp
import os
import json
import numpy as np

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

    logger = open('results_classwise_road.tex','w')
    result_modes = ['box','tube']
    label_types = ['av_actions','agent','action','loc', 'duplex','triplet']
    base = '/mnt/mercury-alpha/road/cache/resnet50'
    
    subsets = ['train_1', 'train_2','train_3']
    
    seq, bs, tseq = (8,4,32)
    com_text = {'joint':'','marginals':'-j4m'}
    for combination_type in ['joint', 'marginals']:
        for net in ['I3D',]:
            logger.write('\n\nRESULTS FOR '+net+ ' ' + combination_type +'\n\n')
            table = 'Train-subset & #instances '
            for l in subsets:
                table += ' & \\multicolumn{2}{c}{' + l.replace('_','-').capitalize() +'} '
            table += '\\\\ \n\\midrule \n Eval subset & #Boxes/#Tubes '
            cc = 1
            for i, l in enumerate((subsets+['avg'])*2):
                if i % 2 == 1:
                    table += ' & Test '
                else:
                    table += ' & Val-{:d}'.format(cc) 
                    cc += 1   
            table += '\\\\ \n\\midrule\n'  
            for label_type in label_types:
                temp_label_type  = label_type
                if label_type == 'av_actions':
                    temp_label_type = label_type[:-1]
                
                labels = final_annots[temp_label_type+'_labels']

                table += '\midrule\n \\multicolumn{2}{c}{' + '{:s} results \\\\ \n\\midrule\n'.format(label_type.capitalize())
                
                vfaps = {}
                vfcounts = {}
                
                for train_subset in subsets:
                    for result_mode in result_modes:
                        
                        splitn = train_subset[-1]
                        
                        if result_mode == 'box':
                            result_file = '{:s}{:s}512-Pkinetics-b{:d}s{:d}x1x1-roadt{:s}-h3x3x3/frame-ap-results-30-{:02d}-50{}.json'.format(base, net, bs, seq, splitn, tseq, com_text[combination_type])
                        else:
                            result_file = '{:s}{:s}512-Pkinetics-b{:d}s{:d}x1x1-roadt{:s}-h3x3x3/tubes-30-{:02d}-50-10-score-50-4/video-ap-results-none-0-20-stiou{}.json'.format(base, net, bs, seq, splitn, tseq,com_text[combination_type])

                        if os.path.isfile(result_file):
                            with open(result_file, 'r') as f:
                                results = json.load(f)
                        else:
                            results = None
                            
                        for subset, pp in [('val_'+splitn,' & '), ('test','/')]: #,
                            tag  = subset + ' & ' + label_type
                            if results is not None and tag in results:
                                aps = np.asarray(results[tag]['APs'])
                                # if subset.startswith('val'):
                                if train_subset+subset not in vfaps:
                                    vfaps[train_subset+subset] = {}
                                if subset not in vfcounts:
                                    vfcounts[subset] = {}
                                vfaps[train_subset+subset][result_mode] = aps
                            else:
                                # logger.write(result_file)
                                vfaps[train_subset+subset][result_mode] = aps*0.0
                            
                            temp_counts = filter_counts(counts[result_mode][subset][temp_label_type], labels)
                            vfcounts[subset][result_mode] = temp_counts
                
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
                    if label not in labels and label != 'ztotal':
                        continue
                    if label != 'ztotal':
                        ind = labels.index(label)
                    table += '{:s} '.format(label.capitalize().ljust(25))
                    for subset in ['all','val_1','val_2','val_3','test']:
                        if label == 'ztotal':
                            numa = np.sum(vfcounts[subset]['box'])
                            numb = np.sum(vfcounts[subset]['tube'])
                        else:
                            numa = vfcounts[subset]['box'][ind]
                            numb = vfcounts[subset]['tube'][ind]
                        mystr = '& {:d}/{:d} '.format(int(numa), int(numb))
                        table += mystr.ljust(15)
                    
                    nums = {'vala':0, 'valb':0,'tesa':0,'tesb':0}
                    for train_subset in subsets:
                        for subset in ['val_'+train_subset[-1], 'test']:
                            if label == 'ztotal':
                                numa = np.mean(vfaps[train_subset+subset]['box'])
                                numb = np.mean(vfaps[train_subset+subset]['tube'])
                            else:
                                numa = vfaps[train_subset+subset]['box'][ind]
                                numb = vfaps[train_subset+subset]['tube'][ind]
                            nums[subset[:3]+'a'] += numa
                            nums[subset[:3]+'b'] += numb
                            mystr = '& {:0.01f}/{:0.01f} '.format(numa, numb)
                            table += mystr.ljust(15)
                    mystr = '& {:0.01f}/{:0.01f} '.format(nums['vala']/3, nums['valb']/3)
                    table += mystr.ljust(15)
                    mystr = '& {:0.01f}/{:0.01f} '.format(nums['tesa']/3, nums['tesb']/3)
                    table += mystr.ljust(15)
                    table += '\\\\ \n'
                    
            logger.write(table)