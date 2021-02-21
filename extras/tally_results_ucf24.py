

"""
This script load and tally the result of UCF24 dataset in latex format.
"""

import os
import json

def run_exp(cmd):
    return os.system(cmd)

if __name__ == '__main__':
    base = '/mnt/mercury-alpha/ucf24/cache/resnet50'
    modes = [['frames',['frame_actions', 'action_ness', 'action']],
            ['video',['action',]]]
    logger = open('results_ucf24.tex','w')
    
    for result_mode, label_types in modes:
        logger.write('\n\nRESULTS FOR '+result_mode+'\n\n')
    
        atable = 'Model'
        for l in label_types: #['0.2', '0.5', '075','Avg-mAP']:
            atable += ' & ' + l.replace('_','-').capitalize() 
        atable += '\\\\ \n\\midrule\n'
        
        subsets = ['train']
        
        for net,d in [('C2D',1), ('I3D',1),('RCN',1), ('RCLSTM',1)]:
            for seq, bs, tseqs in [(8,4,[8,32])]:
                for tseq in tseqs: 
                    if result_mode == 'video':
                        trims = ['none','indiv']
                        eval_ths_all = [[20], [50], [75], [a for a in range(50,95,5)]]
                    else:
                        trims = ['none'] 
                        eval_ths_all = [[50]]

                    for trim in trims:
                        if result_mode != 'video':
                            atable += '{:s}-{:02d} '.format(net, tseq).ljust(15)
                        else:
                            atable += '{:s}-{:02d}-{:s} '.format(net, tseq, trim).ljust(20)

                        for train_subset in subsets:
                            
                            splitn = train_subset[-1]
                            for eval_ths in eval_ths_all:
                                # logger.write(eval_ths)
                                anums = [[0,0] for _ in label_types]
                                for eval_th in eval_ths:
                                    if result_mode == 'frames':
                                        result_file = '{:s}{:s}512-Pkinetics-b{:d}s{:d}x1x1-ucf24t{:s}-h3x3x3/frame-ap-results-10-{:02d}-50.json'.format(base, net, bs, seq, splitn, tseq)
                                    else:
                                        result_file = '{:s}{:s}512-Pkinetics-b{:d}s{:d}x1x1-ucf24t{:s}-h3x3x3/tubes-10-{:02d}-80-20-score-25-4/video-ap-results-{:s}-0-{:d}-stiou.json'.format(base, net, bs, seq, splitn, tseq, trim, int(eval_th))

                                    if os.path.isfile(result_file):
                                        with open(result_file, 'r') as f:
                                            results = json.load(f)
                                    else:
                                        results = None
                                    for nlt, label_type in enumerate(label_types):
                                        cc = 0
                                        for subset, pp in [('test','&')]: #,
                                            tag  = subset + ' & ' + label_type
                                            if results is not None and tag in results:
                                                num = results[tag]['mAP']
                                                anums[nlt][cc] += num
                                                cc += 1

                                for nlt, label_type in enumerate(label_types):
                                    cc = 0
                                    for subset, pp in [('test','&')]: #,
                                        num = anums[nlt][cc]/len(eval_ths)
                                        atable +=  '{:s} {:0.01f} '.format(pp, num)
                                        cc += 1
                        atable += '\\\\ \n'
        logger.write(atable)            
                        