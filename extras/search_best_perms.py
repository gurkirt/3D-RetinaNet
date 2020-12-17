
"""
Search/run for result for searching best alpha values in temporal trimming.
used before 'serach_road_results.py'.

"""

import multiprocessing as mp
import os
import json

def run_exp(cmd):
    return os.system(cmd)


def is_done(base_dir, count, tseq):
    filename50 = '{:s}video-ap-results-10-{:02d}_{:d}_50_stiou.json'.format(base_dir, tseq, count)
    filename75 = '{:s}video-ap-results-10-{:02d}_{:d}_20_stiou.json'.format(base_dir, tseq, count)
    # print(filename50)
    if os.path.isfile(filename50) and os.path.isfile(filename75):
        return True
    return False

if __name__ == '__main__':
    mode = 'eval_tubes'
    import os
    count = 5000
    cmds = []
    setups = []
    for net,d in [('RCN',1), ('I3D',1), ]:
        for train_subset in ['train_3']:
            for seq, bs, tseqs in [(8,4,[32])]:
                for tseq in tseqs: 
                    cmd = 'python main.py --COMPUTE_PATHS=False --COMPUTE_TUBES=False --EVAL_EPOCHS=30 --MODE={:s} --MODEL_TYPE={:s} --TEST_SEQ_LEN={:d} --TRAIN_SUBSETS={:s} --SEQ_LEN={:d} --BATCH_SIZE={:d} '.format(mode, net, tseq, train_subset, seq, bs)
                    # cmd = 'python main.py --COMPUTE_PATHS=False --COMPUTE_TUBES=True --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE={:s} --MODEL_TYPE={:s} --TEST_SEQ_LEN={:d} --TRAIN_SUBSETS={:s} --SEQ_LEN={:d} --BATCH_SIZE={:d} '.format(mode, net, tseq, train_subset, seq, bs)
                    # cmd = 'CUDA_VISIBLE_DEVICES={:d} python main.py --MODE={:s} --MODEL_TYPE={:s} --TEST_SEQ_LEN={:d} --TRAIN_SUBSETS={:s} --SEQ_LEN={:d} --BATCH_SIZE={:d}'.format(d, mode, net, tseq, train_subset, seq, bs)
                    for nms in [0.5, 0.6, 0.8]:
                        for topk in [10,20,30,40]:
                            for jp in [4]:
                                for iouth in [0.1,0.2,0.25,0.35,0.4,0.45,0.5,0.6]:
                                    for cst in ['score','scoreiou']:
                                        for trim in ['none','indiv']:
                                            count += 1
                                            if trim == 'indiv':
                                                alphas = [0,1,2,3,5,8,16]
                                            else:
                                                alphas = [0]
                                            for alpha in alphas:
                                                cmd1 = cmd + '--GEN_NMS={:0.02f} --PATHS_IOUTH={:0.02f} --PATHS_COST_TYPE={:s} --PATHS_JUMP_GAP={:d} --TOPK={:d} --PATHS_MIN_LEN=4 --TRIM_METHOD={:s} --TUBES_ALPHA={:f}'.format(nms,iouth,cst,jp, topk, trim, alpha)
                                                setups.append([net,nms,topk,jp,iouth,cst,tseq,trim,alpha,count])
                                                # base_dir ='/mnt/mercury-alpha/ucf24/cache/resnet50{}512-Pkinetics-b4s8x1x1-ucf24tn-h3x3x3/'.format(net)
                                                print(cmd1)
                                                cmds.append(cmd1)

    
    print('Number of cpus', mp.cpu_count())
    print('Number of commands', len(cmds))
    with open('setups_alphas_road.json', 'w') as f:
        json.dump([setups,cmds], f, indent=2)
    
    pool = mp.Pool(16)
    results = pool.map(run_exp, [row for row in cmds])
    pool.close()

    print('Results::', results)
