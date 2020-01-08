import os
import numpy as np

PAD = 'PAD' #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK = 'UNK' #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP = 'KEEP' # This has a vocab id, which is used for copying from the source [2]
DEL = 'DEL' # This has a vocab id, which is used for deleting the corresponding word [3]
START = 'START' # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP = 'STOP' # This has a vocab id, which is used to stop decoding [5]

PAD_ID = 0 #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK_ID = 1 #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP_ID = 2 # This has a vocab id, which is used for copying from the source [2]
DEL_ID = 3 # This has a vocab id, which is used for deleting the corresponding word [3]
START_ID = 4 # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP_ID = 5 # This has a vocab id, which is used to stop decoding [5]

def replace_lrb(sent_string):
    sent_string = sent_string.lower()
    # new_sent= sent_string.replace('-lrb-','(').replace('-rrb-',')')
    new_sent = sent_string.replace('-lrb-', '(').replace('-rrb-', ')')
    return new_sent


def human_eval_create(folder, sample_size=30, dataset_size=100):
    idxs = np.random.choice(range(dataset_size), size=sample_size, replace=False, p=None)
    idx_list = [i in idxs for i in range(dataset_size)]
    listdir=os.listdir(folder)
    for file in listdir:
        path = folder + file
        if os.path.isfile(path):
            with open(path,'r') as f:
                doc=f.readlines()
                doc=[replace_lrb(line) for line in doc]
                sample = [line for index,line in zip(idx_list, doc) if index]
            tgt_path=folder+'/sample/'
            if not os.path.exists(tgt_path):
                os.mkdir(tgt_path)
            with open(tgt_path+file+'.%d'%sample_size,'w') as writer:
                for line in sample:
                    writer.write(line)

def replace_parenthesis(folder):
    listdir = os.listdir(folder)
    for file in listdir:
        path = folder + file
        if os.path.isfile(path):
            with open(path, 'r') as f:
                doc = f.readlines()
                doc = [replace_lrb(line) for line in doc]
            tgt_path = folder + '/sample3/'
            if not os.path.exists(tgt_path):
                os.mkdir(tgt_path)
            with open(tgt_path + file , 'w') as writer:
                for line in doc:
                    writer.write(line)

def copy_ratio(sys_list, src_list):
    sys_list=sys_list.split(' ')
    src_list=src_list.split(' ')
    denominator = len(src_list)
    numerator = len([i for i in sys_list if i in src_list])
    return numerator*1.0/denominator


def novel_ratio(sys_list, src_list):
    sys_list=sys_list.split(' ')
    src_list=src_list.split(' ')

    denominator = len(sys_list)
    numerator = len([i for i in sys_list if i not in src_list ])
    return numerator*1.0/denominator

def correct_novel_ratio(sys_list, src_list,tar_list):
    sys_list=sys_list.split(' ')
    src_list=src_list.split(' ')
    tar_list=tar_list.split(' ')
    denominator = len(sys_list)
    numerator = len([i for i in sys_list if i not in src_list and i in tar_list])
    return numerator*1.0/denominator

def ablation_stats(file,source_path,tar_path):
    all_lines=[]
    STAR_list=[]
    with open(tar_path,'r') as tar:
        tar_lines=tar.readlines()
        # print(len(tar_lins))
    with open(source_path,'r') as src:
        src_lines=src.readlines()
        # print(len(src_lines))
    with open(file,'r') as f:
        sys_lines = f.readlines()
        # print(len(sys_lines))

    ## sentence length
    avg_len=np.mean([len(l.split(' ')) for l in sys_lines] )
    print('sentence avg len:', avg_len)

    # avg_copy
    avg_copy=np.mean([copy_ratio(sys_lines[idx],src_lines[idx]) for idx in range(len(sys_lines))])
    print('avg_copy',avg_copy)

    # avg_novel
    avg_novel = np.mean([novel_ratio(sys_lines[idx], src_lines[idx]) for idx in range(len(sys_lines))])
    print('avg_novel', avg_novel)

    avg_novel = np.mean([correct_novel_ratio(sys_lines[idx], src_lines[idx], tar_lines[idx]) for idx in range(len(sys_lines))])
    print('avg_correct_novel', avg_novel)

    return avg_len,avg_copy,avg_novel


def unchanged_line_stat(src_path, file_path):
    with open(src_path, 'r') as src:
        src_lines = src.readlines()
        src_lines = [line.lower() for line in src_lines]
    with open(file_path, 'r') as f:
        sys_lines = f.readlines()
        sys_lines = [line.lower() for line in sys_lines]

    unchanged_lines = [l for l in sys_lines if l in src_lines]
    # print(len(unchanged_lines))
    print(len(unchanged_lines)*1.0/len(sys_lines))
    return len(unchanged_lines)*1.0/len(sys_lines)




