
"""
This script load the result for hypermeter search and tool to filter them
Run after 'search_best_perms_all.py'
"""

import multiprocessing as mp
import os
import json
import numpy as np
def run_exp(cmd):
    return os.system(cmd)

def get_maps(base_dir, trim, alpha, count, load_name='test & action'):
    # filename50 = '{:s}video-ap-results-{:s}-{:02d}_{:d}_50_stiou.json'.format(base_dir, trim, alpha)
    # filename75 = '{:s}video-ap-results-{:s}-{:02d}_{:d}_75_stiou.json'.format(base_dir, trim, alpha)
    filename50 = "{pt:s}/video-ap-results-{tm:s}-{a:d}-{th:d}-{m:s}.json".format(tm=trim, a=int(alpha*10), pt=base_dir,  th=int(0.5*100), m='stiou')
    filename75 = "{pt:s}/video-ap-results-{tm:s}-{a:d}-{th:d}-{m:s}.json".format(tm=trim, a=int(alpha*10), pt=base_dir,  th=int(0.75*100), m='stiou')
                    
    # print(filename50)
    if os.path.isfile(filename50) and os.path.isfile(filename75):
        with open(filename50, 'r') as f:
            results = json.load(f)
        mAP50 = results[load_name]['mAP']
        aps = results[load_name]['APs']
        with open(filename75, 'r') as f:
            results = json.load(f)
        mAP75 = results[load_name]['mAP']
        return mAP50, mAP75, np.asarray(aps), np.asarray(results[load_name]['APs']), results[load_name]['APs']
    
    print(filename50, 'not found!!!')

    return 0,0, [], [], []

if __name__ == '__main__':
    mode = 'eval_tubes'
    import os
    count = 0
    with open('setups_alphas_ucf24.json', 'r') as f:
        [setups,cmds] = json.load(f)
    
    strs = ['|' for _ in range(26)]
    maxs = [[0,-1] for _ in range(24)]
    # for net in ['C2D', 'RCLSTM', 'RCN', 'I3D']:
    max_map_50 = 0
    max_map_20 = 0
    imax_map_50 = 0
    imax_map_20 = 0
    imax_m = 0
    max_m = 0
    s25 = 0
    s50 = 0
    
    cc = 0
    for ind in range(len(setups)):
        [net,nms,topk,jp,iouth,cst,tseq,trim,alpha,count] = setups[ind]

        if trim == 'none' and net == 'RCN' and nms==0.8 and iouth==0.25 and jp==4 and topk==20: # and trim == 'none' and nms==0.8: # and topk==25: # and jp==2 and alpha==0 and :
            print(setups[ind])
            for net in ['RCLSTM','RCN','I3D','C2D']:
            
                # print(net, count)
                base_dir ='/mnt/mercury-alpha/ucf24/cache/resnet50{}512-Pkinetics-b4s8x1x1-ucf24tn-h3x3x3/'.format(net)
                # base_dir ='/mnt/mercury-alpha/road/cache/resnet50{}512-Pkinetics-b4s8x1x1-roadt3-h3x3x3/'.format(net)
                save_dir = "{pt:s}/tubes-{it:02d}-{sq:02d}-{n:d}-{tk:d}-{s:s}-{io:d}-{jp:d}/".format(pt=base_dir, it=10,  
                                    sq=tseq,  n=int(100*nms), tk=topk, s=cst, io=int(iouth*100), jp=jp)
                #APs75iouth>=0.025 and trim == 'none' and nms==0.8: # and topk==25: # and jp==2 and alpha==0 and :
                amAP50, amAP20, ameanv = 0,0,0
                iamAP50, iamAP20, iameanv = 0,0,0
                results = []
                
                for load_name in ['test & action']:
                    mAP50, mAP20, APs50, APs20, apstrs = get_maps(save_dir, 'none', 0, load_name)
                    meanv = (mAP50+mAP20)/2
                    amAP50 += mAP50
                    amAP20 += mAP20
                    ameanv += meanv
                    results.append([mAP50, mAP20, APs50, APs20, apstrs])
                
                amAP50 /= 1
                amAP20 /= 1
                ameanv /= 1

                # if ameanv>12.0:
                all_found = True
                for load_name in ['test & action']:
                    aps5 = []
                    aps2 = []
                    numc = 0
                    alphas = []
                    for alpha in [1,2,3,5,8,16]:
                        mAP50, mAP20, APs50, APs20, apstrs = get_maps(save_dir, 'indiv', alpha, load_name)
                        if mAP50 == 0:
                            all_found = False
                            break
                        alphas.append(alpha)
                        numc = APs20.shape[0]
                        aps5.append(APs50)
                        aps2.append(APs20)
                    if not all_found:
                        continue
                    aps2 = np.asarray(aps2)
                    aps5 = np.asarray(aps5)
                    alphas = np.asarray(alphas)
                    aps5_inds = np.argmax(aps5, axis=0)
                    # print(aps5_inds, aps5)
                    # print(aps5[aps5_inds])
                    alphas = alphas[aps5_inds]
                    aps5_ = []
                    aps2_ = []
                    for c in range(aps5.shape[1]):
                        aps5_.append(aps5[aps5_inds[c],c])
                        aps2_.append(aps2[aps5_inds[c],c])
                    aps2 = np.asarray(aps2_)
                    aps5 = np.asarray(aps5_)
                    assert numc == aps2.shape[0], aps2.shape
                    aps2 = np.mean(aps2)
                    aps5 = np.mean(aps5)
                    meanv = (aps2+aps5)/2
                    iamAP50 += aps5
                    iamAP20 += aps2
                    iameanv += meanv
                        
                    

                    iamAP50 /= 1
                    iamAP20 /= 1
                    iameanv /= 1

                    max_map_50 = max(max_map_50, amAP50)
                    max_map_20 = max(max_map_20, amAP20)
                    max_m = max(max_m, ameanv) 

                    imax_map_50 = max(imax_map_50, iamAP50)
                    imax_map_20 = max(imax_map_20, iamAP20)
                    imax_m = max(imax_m, iameanv)    
                    
                    print('{:s} {:0.2f} {:02d} {:0.02f} {:s} {:s} {:0.02f} {:0.02f} {:0.02f} {:0.02f} {:0.02f} {:0.02f} '.format(net, nms, topk, iouth, cst[-5:], trim, amAP20,amAP50, ameanv, iamAP20,iamAP50, iameanv), ['{:0.01f}'.format(m[0]) for m in results])
                    print(alphas)
                # print(cmds[ind])
                # strs[0] += '{:d}|'.format(int(alpha))
                # for c in range(1,25):
                #     strs[c] += '{:0.01f}/{:0.01f}|'.format(APs50[c-1], APs75[c-1])
                
                # strs[25] += '{:0.01f}/{:0.01f}|'.format(mAP50, mAP75)
                
                # for c in range(24):
                #     mm = APs50[c] + APs75[c]
                #     if maxs[c][0]<mm:
                #         maxs[c][0] = mm
                #         maxs[c][1] = alpha

                # cc += 1
                # if cc % 6 == 0:
                #     print([m[1] for m in maxs])
                #     # for str in strs:
                #     #     print(str)
                #     strs = ['|' for _ in range(26)]
                #     maxs = [[0,-1] for _ in range(24)]
                #     cc = 0
    # for str in strs:
    #     print(str)
    
    print('amx map:: {:0.02f} {:0.02f} {:0.02f} {:0.02f} {:0.02f} {:0.02f} '.format(max_map_20, max_map_50, max_m, imax_map_20, imax_map_50, imax_m))


        # 
