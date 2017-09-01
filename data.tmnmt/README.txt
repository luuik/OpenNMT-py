# preprocess the data

python preprocess_tmnmt.py -train_src ./train.fr.toy.bpe -train_tgt ./train.en.toy.bpe -valid_src ./train.fr.toy.bpe -valid_tgt ./train.en.toy.bpe -valid_tm ./train.toy.bpe.tms -save_data ./mydata -train_tm ./train.toy.bpe.tms

# train

python train_tm-nmt.py -data ./mydata -save_model ./mymodel
