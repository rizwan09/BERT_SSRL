import os, sys

config_path = 'training_config/'
data_path = '/home/rizwan/SBCR/data/conll12/conll-formatted-ontonotes-5.0/data/'

output_path = '../.allennlp/cache/'
dev_file = data_path + 'test'
dev_file += '/data/english/'

remove = True
# remove = False


option = 'train'
# option = 'evaluate'
config = config_path + 'srl_sbert_config.jsonnet'
model_file = '../.allennlp/cache/bert-base-srl-2019.06.17.tar.gz'
output_path += 'SRL_BERT_baseline'
# model_file = output_path + '/model.tar.gz'
#
devices = '1,3,4,5'

run_command = 'CUDA_VISIBLE_DEVICES=' + devices + ' python -m allennlp.run ' + option + ' '

if option == 'train':
    if remove: os.system('rm -r ' + output_path)
    run_command += config + ' -s ' + ' ' + output_path  # + ' --recover '
# if option == 'evaluate':
else:
    run_command += model_file + ' '
    # run_command += ' --evaluation-data-file '
    run_command += dev_file

print(run_command)
os.system(run_command)


