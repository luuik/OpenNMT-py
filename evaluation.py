from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import subprocess
import argparse
import pyter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser(description='evaluate.py')
parser.add_argument('-ref_file', required=True,
                help='Path to reference file')
parser.add_argument('-hyp_file', required=True,
                    help='Path to hypothesis file')

def bleu_score(ref_file, hyp_file):
    """Computes corpus level BLEU score with Moses' multi-bleu.pl script

    Arguments:
        ref_file (str): Path to the reference file
        hyp_file (str): Path to the hypothesis file

    Returns:
        tuple: Tuple (BLEU, details) containing the bleu score and the detailed output of the perl script

    Raises:
        ValueError: Raises error if the perl script fails for some reason
    """
    command = 'perl scripts/multi-bleu.pl ' + ref_file + ' < '+hyp_file
    c = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    details, error = c.communicate()
    if not details.startswith('BLEU ='):
        raise ValueError('Error in BLEU score computation:\n' + error)
    else:
        BLEU_str = details.split(' ')[2][:-1]
        BLEU = float(BLEU_str)
        return BLEU, details

def ter_score(ref_file, hyp_file):
    with open(ref_file) as f:
      references = f.readlines()
      references = [ref.strip().split() for ref in references]

    av_ref_len = np.mean([len(ref) for ref in references])

    with open(hyp_file) as f:
      hypothesis = f.readlines()
      hypothesis = [hyp.strip().split() for hyp in hypothesis]

    av_hyp_len = np.mean([len(hyp) for hyp in hypothesis])

    ter_list = [pyter.ter(hyp, ref) for hyp, ref in zip(hypothesis, references)]

    return sum(ter_list)/len(ter_list), av_ref_len, av_hyp_len

def plot_heatmap(att_weights, idx, srcSent, tgtSent):

    plt.figure(figsize=(8, 6), dpi=80)
    att_weights = att_weights[0][0].cpu().numpy()
    #print("Att_weights", att_weights)
    plt.imshow(att_weights, cmap='gray', interpolation='nearest')
    srcSent = [str(s) for s in srcSent]
    tgtSent = [str(s) for s in tgtSent]
    
    plt.xticks(range(0, len(tgtSent)),tgtSent)
    plt.yticks(range(0, len(srcSent)),srcSent)
    plt.savefig("softmax_att_matrix_"+str(idx), bbox_inches='tight')
    plt.close()


def main():
    opt = parser.parse_args()
    
    bleu, details = bleu_score(opt.ref_file, opt.hyp_file)
    av_ter , av_ref_len, av_hyp_len = ter_score(opt.ref_file, opt.hyp_file)
    print("BLEU Score: %f , %s" % (float(bleu), details))
    print("TER Score: %f" % av_ter)
    print("Average Reference Length: %f" % av_ref_len)
    print("Average Hypothesis Length: %f" % av_hyp_len) 

if __name__ == "__main__":

    main()
