import os
import SARI
import numpy as np

def corpus_sari(path_tar=None, path_comp=None, path_ref=None, num_refs=None):
    """
    :param path_tar: the data path for the model output to be evaluated
    :param path_comp: the file path for the original test text
    :param path_ref: the file path for the reference text
    :param num_refs: the number of references, 8 in wikilarge, and 1 in other datasets
    :return: none, print the evaluation score
    """
    a = get_result_sari(path_tar, path_comp, path_ref, num_refs)
    print(a)


def get_result_sari(path_tar, path_comp, path_simp, num_refs):
    output_stats= []
    with open(path_comp) as comp_f:
        comp = comp_f.readlines()

    with open(path_tar) as tar_f:
        tar = tar_f.readlines()

    if num_refs ==1:
        with open (path_simp) as ref_f:
            ref = ref_f.readlines()
    else:
        ref_list=[]
        for i in range(num_refs):
            with open(path_simp+str(i)) as ref_f:
                ref = ref_f.readlines()
            ref_list.append(ref)

    for i in range(len(tar)):
        if num_refs==1:
            s_stat = SARI.SARIsent(comp[i], tar[i], [ref[i]])
        else:
            s_stat=SARI.SARIsent(comp[i], tar[i], [ref[i] for ref in ref_list])
        output_stats.append(s_stat)

    output_stats = np.array(output_stats)
    result = output_stats.mean(axis=0)
    return result #final score, add,keep, del
#
def lower(raw_data_dir,tgt_dir):
    for file in os.listdir(raw_data_dir):
        print(file)
        if os.path.isfile(raw_data_dir+file):
            print(file)
            tgt = open(tgt_dir + file, 'w')

            with open(raw_data_dir + file, 'r') as f:
                lines = f.readlines()
                print(file, 'has %d lines' % len(lines))
                for line in lines:
                    line = line.lower()
                    tgt.write(line)
            tgt.close()
            print(tgt_dir + file, 'saved')


# wikilarge_path_comp='/home/yue/vocab_data/TS/arxiv/xu/turkcorpus/test/test.8turkers.tok.norm'
# wikilarge_path_ref='/home/yue/vocab_data/TS/arxiv/xu/turkcorpus/test/test.8turkers.tok.turk.'
# wikilarge_num_refs=8

# newsela_path_comp='/home/yue/vocab_data/TS/arxiv/newsela_xingxing/test.comp'
# newsela_path_ref='/home/yue/vocab_data/TS/arxiv/newsela_xingxing/test.simp'

# wikismall_path_comp = '/home/yue/vocab_data/TS/arxiv/wikismall_eval/test.comp'
# wikismall_path_ref='/home/yue/vocab_data/TS/arxiv/wikismall_eval/test.simp'

