
# import multiprocessing as mp
import os
import json
import numpy as np
import pdb

if __name__ == '__main__':

    anno_file  =  'annots_12fps_full_v1.0.json'
    version = '1.2'
    with open(anno_file,'r') as fff:
        final_annots = json.load(fff)
    
    # pdb.set_trace()
    trainval = {}
    test = {}
    for key in final_annots:
        if key != 'db':
            trainval[key] = final_annots[key]
            test[key] = final_annots[key]
        else:
            db = final_annots[key]
            trainval_db = {}
            test_db = {}
            for vidname in db:
                if 'test' in db[vidname]['split_ids']:
                    test_db[vidname] = db[vidname]
                else:
                    trainval_db[vidname] = db[vidname]
            test[key] = test_db
            trainval[key] = trainval_db
    
    anno_file  =  'road_trainval_v1.0.json'
    with open(anno_file,'w') as fff:
        json.dump(trainval, fff)
    
    anno_file  =  'road_test_v1.0.json'
    with open(anno_file,'w') as fff:
        json.dump(test, fff)


    
