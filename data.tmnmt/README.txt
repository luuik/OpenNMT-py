# preprocess the data

python preprocess_tmnmt.py -train_src ./data.tmnmt/train.fr.toy.bpe -train_tgt ./data.tmnmt/train.en.toy.bpe -valid_src ./data.tmnmt/train.fr.toy.bpe -valid_tgt ./data.tmnmt/train.en.toy.bpe -valid_tm ./data.tmnmt/train.toy.bpe.tms -save_data ./mydata -train_tm ./data.tmnmt/train.toy.bpe.tms

# train the baseline NMTModel

python train.py -data ./mydata -save_model ./mynmtmodel

# train the TM_NMTModel

python train_tm-nmt.py -data ./mydata -save_model ./mymodel -train_from ./mynmtmodel


