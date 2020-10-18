
# import multiprocessing as mp
import os
import json

def run_exp(cmd):
    return os.system(cmd)

if __name__ == '__main__':
    mode = 'eval_frames'
    import os
    count = 0
    cmds = []
    base = '/mnt/mercury-alpha/aarav/cache/resnet50'
    table = 'model & train-1 & tain 2 & train 3\\\\ \nlabel-type '
    label_types = ['agent', 'action', 'loc', 'duplex', 'triplet']
    for l in label_types*3:
        table += ' & ' + l.replace('_','-')  
    table += '\\\\ \n'
    for net,d in [('C2D',1), ('I3D',1),('RCN',1)]:
        for seq, bs, tseqs in [(16,8, [16,32]), (8,4,[8,32])]:
            for tseq in tseqs: 
                table += '{:s}-{:02d}-{:02d} '.format(net, seq, tseq)
                for train_subset in ['train_1', 'train_2','train_3']:
                    splitn = train_subset[-1]
                    result_file = '{:s}{:s}512-Pkinetics-b{:d}s{:d}x1x1-aaravt{:s}-h3x3x3/frame-ap-results-30-{:02d}.json'.format(base, net, bs, seq, splitn,tseq)
                    # cmd = 'CUDA_VISIBLE_DEVICES={:d} python main.py --MODE={:s} --MODEL_TYPE={:s} --TEST_SEQ_LEN={:d} --TRAIN_SUBSETS={:s} --SEQ_LEN={:d} --BATCH_SIZE={:d}'.format(d, mode, net, tseq, train_subset, seq, bs)
                    if os.path.isfile(result_file):
                        count += 1
                        print(count, result_file)
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                    else:
                        results = None
                        print('NON', result_file)
                    # print(results)
                    for label_type in label_types:
                        for subset, pp in [('test','  & ')]: # ('val_'+splitn,'/'),
                            tag  = subset + ' & ' + label_type
                            if results is None:
                                table +=  '{:s} -- '.format(pp)
                            else:
                                num = results[tag]['mAP']
                                table +=  '{:s}{:0.01f}'.format(pp, num)
                table += '\\\\ \n'
    print(table)
                            
                        
    # print('Number of commands', len(cmds))