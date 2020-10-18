
import multiprocessing as mp
import os

def run_exp(cmd):
    return os.system(cmd)

if __name__ == '__main__':
    mode = 'eval_frames'
    import os
    count = 0
    cmds = []
    for net,d in [('I3D',1),('RCN',1),('C2D',1)]:
        for train_subset in ['train_3', 'train_2','train_1']:
            for seq, bs, tseqs in [(16,8, [16,32]), (8,4,[8,32])]:
                for tseq in tseqs: 
                        # if not (train_subset == 'train_1' and seq == 8):
                        if count<34:
                            count += 1
                            cmd = 'python main.py --MODE={:s} --MODEL_TYPE={:s} --TEST_SEQ_LEN={:d} --TRAIN_SUBSETS={:s} --SEQ_LEN={:d} --BATCH_SIZE={:d}'.format(mode, net, tseq, train_subset, seq, bs)
                            # cmd = 'CUDA_VISIBLE_DEVICES={:d} python main.py --MODE={:s} --MODEL_TYPE={:s} --TEST_SEQ_LEN={:d} --TRAIN_SUBSETS={:s} --SEQ_LEN={:d} --BATCH_SIZE={:d}'.format(d, mode, net, tseq, train_subset, seq, bs)
                            print(count, cmd)
                            cmds.append(cmd)
                            # os.system(cmd )
    print('Number of cpus', mp.cpu_count())
    print('Number of commands', len(cmds))

    pool = mp.Pool(10)
    results = pool.map(run_exp, [row for row in cmds])

    pool.close()

    print('Results::', results)
