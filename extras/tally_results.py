
"""
This script load and tally the result of ROAD dataset in latex format.
Results are avergaged accross all three splits.
"""

import os
import json

def run_exp(cmd):
    return os.system(cmd)

if __name__ == '__main__':
    modes = [['frames',['av_actions', 'agent_ness', 'agent', 'action', 'loc', 'duplex', 'triplet']],
             ['video',['agent', 'action', 'loc', 'duplex', 'triplet']]]
    base = '/mnt/mercury-alpha/road/cache/resnet50'
    thesholds = {'video':['20','50'],'frames':['50']}
    com_text = {'joint':'','marginals':'-j4m'}
    logger = open('results_road.tex','w')
    for combination_type in ['joint', 'marginals']:
        for result_mode, label_types  in modes:
            for th in thesholds[result_mode]:
                logger.write('\n\nRESULT MODE '+result_mode + 'Thrshold '+ th + 'combination type '+combination_type+' \n\n')
                atable = 'Model'
                table = 'Train-Set & \\multicolumn{5}{c}{Train-1} & \\multicolumn{5}{c}{Train-2}  \\multicolumn{5}{c}{Train-3} \\\\ \n\\midrule\nModel \\textbackslash Label '
                
                for l in label_types*3:
                    table += ' & ' + l.replace('_','-').capitalize() 
                for l in label_types:
                    atable += ' & ' + l.replace('_','-').capitalize() 
                table += '\\\\ \n\\midrule\n'
                atable += '\\\\ \n\\midrule\n'

                subsets = ['train_1', 'train_2','train_3']

                for net,d in [('C2D',1), ('I3D',1),('RCN',1), ('RCLSTM',1)]:
                    for seq, bs, tseqs in [(8,4,[8,32])]:
                        for tseq in tseqs: 
                            if net == 'RCLSTM' and seq == 16:
                                continue
                            table += '{:s}-{:02d}-{:02d} '.format(net.rjust(6), seq, tseq)
                            atable += '{:s}-{:02d}-{:02d} '.format(net.rjust(6), seq, tseq)
                            anums = [[0,0] for label_type in label_types]
                            
                            for train_subset in subsets:
                                splitn = train_subset[-1]
                                if result_mode == 'frames':
                                    result_file = '{:s}{:s}512-Pkinetics-b{:d}s{:d}x1x1-roadt{:s}-h3x3x3/frame-ap-results-30-{:02d}-{:s}{:s}.json'.format(base, net, bs, seq, splitn, tseq, th, com_text[combination_type])
                                else:
                                    result_file = '{:s}{:s}512-Pkinetics-b{:d}s{:d}x1x1-roadt{:s}-h3x3x3/tubes-30-{:02d}-50-10-score-50-4/video-ap-results-none-0-{:s}-stiou{:s}.json'.format(base, net, bs, seq, splitn, tseq, th, com_text[combination_type])

                                if os.path.isfile(result_file):
                                    with open(result_file, 'r') as f:
                                        results = json.load(f)
                                else:
                                    results = None
                                    
                                for nlt, label_type in enumerate(label_types):
                                    cc = 0
                                    mstr = ''
                                    for subset, pp in [('val_'+splitn,' & '), ('test','/')]: #,
                                        tag  = subset + ' & ' + label_type
                                        if results is None or tag not in results:
                                            mstr +=  '{:s} -- '.format(pp)
                                        else:
                                            num = results[tag]['mAP']
                                            mstr +=  '{:s}{:0.01f}'.format(pp, num)
                                            anums[nlt][cc] += num
                                            cc += 1
                                    table += mstr.ljust(12)
                            for nlt, label_type in enumerate(label_types):
                                cc = 0
                                mstr = ''
                                for subset, pp in [('val_'+splitn,' & '), ('test','/')]: #,
                                    num = anums[nlt][cc]/len(subsets)
                                    mstr +=  '{:s}{:0.01f}'.format(pp, num)
                                    cc += 1
                                atable += mstr.ljust(12)
                            table += '\\\\ \n'
                            atable += '\\\\ \n'
                logger.write(table)
                logger.write(atable)            
                        