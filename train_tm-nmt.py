from __future__ import division

import os

import onmt
import onmt.Models
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
import opts

parser = argparse.ArgumentParser(description='train.py')

# Data and loading options
parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

def check_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def main():
    print("Loading data from '%s'" % opt.data)

    train = torch.load(opt.data + '.train.pt')
    fields = onmt.IO.ONMTDataset.load_fields(
        torch.load(opt.data + '.vocab.pt'))
    valid = torch.load(opt.data + '.valid.pt')
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields
    src_features = [fields["src_feat_"+str(j)]
                    for j in range(train.nfeatures)]

    checkpoint = None
    dict_checkpoint = opt.train_from

    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint,
                                map_location=lambda storage, loc: storage)
        #fields = onmt.IO.load_fields(checkpoint['vocab'])

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' %
              (j, len(feat.vocab)))

    print(' * number of training sentences. %d' %
          len(train))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building Base NMT model...')
    cuda = (len(opt.gpuid) >= 1)
    BaseNMTModel = onmt.Models.make_base_model(opt, opt, fields, cuda, checkpoint)
    print(BaseNMTModel)


    if len(opt.gpuid) > 1:
        print('Multi gpu training ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    #     generator = nn.DataParallel(generator, device_ids=opt.gpuid, dim=0)

    TMNMTModel = onmt.Models.TMNMTModel(BaseNMTModel, opt, len(opt.gpuid) > 1)

    if opt.param_init != 0.0:
        print('Intializing params')
        for p in TMNMTModel.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)
            
    optim = onmt.Optim(
        opt.optim, opt.learning_rate, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        opt=opt
    )
    optim.set_parameters(TMNMTModel.parameters())

    # TODO: count and display number of parameters
    
    check_model_path()

    # TODO: train TM-NMTModel


if __name__ == "__main__":
    main()
