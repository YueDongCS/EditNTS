import subprocess
import re

joshua_script = '/home/ml/ydong26/phd/editNTS/editNTS/script/ppdb-simplification-release-joshua5.0/joshua/bin/bleu'
joshua_class ='/home/ml/ydong26/phd/editNTS/editNTS/script/ppdb-simplification-release-joshua5.0/joshua/class'

def corpus_bleu(path_tar=None, path_comp=None, path_ref=None, num_refs=None):
    """
    :param path_tar: the data path for the model output to be evaluated
    :param path_comp: the file path for the original test text
    :param path_ref: the file path for the reference text
    :param num_refs: the number of references, 8 in wikilarge, and 1 in other datasets
    :return: none, print the evaluation score
    """
    a = get_result_joshua(path_ref, path_tar,num_refs)
    print(a)

def get_result_joshua(path_ref, path_tar, num_refs):
    args = ' '.join([joshua_script, path_tar, path_ref,
                     str(num_refs), joshua_class])

    pipe = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    mteval_result = pipe.communicate()

    m = re.search(b'BLEU = ([\d+\.]+)', mteval_result[0])

    try:
        result = float(m.group(1))
    except AttributeError:
        result = 0
    return result

# wikilarge_path_ref='/home/yue/vocab_data/TS/arxiv/xu/turkcorpus/test/test.8turkers.tok.turk.'
# newsela_path_ref="/home/ml/ydong26/phd/editNTS/editNTS/script/refs/newsela_xingxing/test.simp"
# wikismall_path_ref='/home/ml/ydong26/phd/editNTS/editNTS/script/refs/wikismall_eval/test.simp'
