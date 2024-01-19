

import os



os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'




#emb_dim_list=['128','256','512','768','1024']
emb_dim_list=['64']

#d_ff_list=['128','256','512','1024','2048','4096']
d_ff_list=['64']

d_kv_list=['32']
d_model_list=['32']
num_heads_list=['4']
num_layers_list=['4']
wd_list=['0.01']
lr_list=['5e-5']




msa_num_list=['31']
neighbor_max_distance_list=[99999] # set a constraint for the max distance from original seq to be a msa seq in version V3
gradient_accumulation_list=[1] # use accumulated gradient to enlarge the batchsize equivalently
batchsize_list=[16]


encoder_num_layers_list=['8']
encoder_num_heads_list=['8']


# ----------------------------------


#mode='data'
#mode='train'
mode='pred'


# ----------------------------------

for emb_dim in emb_dim_list:
  for d_ff in d_ff_list:
    for d_kv in d_kv_list:
      for d_model in d_model_list:
        for num_heads in num_heads_list:
          for num_layers in num_layers_list:
            for wd in wd_list:
              for lr in lr_list:
                for msa_num in msa_num_list:
                  for neighbor_max_distance in neighbor_max_distance_list:
                    for gradient_accumulation in gradient_accumulation_list:
                      for encoder_num_layers in encoder_num_layers_list:
                        for encoder_num_heads in encoder_num_heads_list:
                          for batchsize in batchsize_list:
                            os.system("python anomaly.py %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s" % (emb_dim, d_ff, d_kv, d_model, num_heads, num_layers, wd, mode, lr, msa_num, neighbor_max_distance,gradient_accumulation, encoder_num_layers, encoder_num_heads, batchsize))



























