

import sys
import os

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, Adafactor
from transformers import BertModel, BertConfig 
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler # NOTE: the BERT (or other modules apart from GPT) was used during the development process and partially used as placeholders in this script, and will not be actually loaded/trained

import transformers
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput,BaseModelOutputWithPastAndCrossAttentions,Seq2SeqLMOutput,Seq2SeqModelOutput
import random

from transformers.models.t5.modeling_t5 import T5Stack
import copy
import csv
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import pickle
seed=2
random.seed(seed)
torch.manual_seed(3)
np.random.seed(seed)

import torch.nn.functional as F
import torch.nn as nn


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score





# ----------------------------  configs



max_seq_len=18 # the max EHR seq length we accept



our_config={'med':{
              'emb_size':64,
              'emb_dim':int(sys.argv[1]),
              'pad_token_id':0,
              'eos_token_id':1,
              'missing_token_id':2, # added, to mask the future meds in X (not D, D meds are masked with mask token. This is similar to what we did for labs)
              'mask_token_id':3,
              'mask2_token_id':4,
              'mask3_token_id':5,
              'mask4_token_id':6,
              'mask5_token_id':7
              },

        'vitals':{'emb_size':1000,
              'emb_dim':int(sys.argv[1]),
              #'sep_token_id':2,
              'pad_token_id':0,
              'eos_token_id':1,
              'missing_token_id':2,
              'mask1_token_id':3,
              'mask2_token_id':4,
              'mask3_token_id':5,
              'mask4_token_id':6,
              'mask5_token_id':7
              },

        'age':{'emb_size':300,
              'emb_dim':int(sys.argv[1]),
              'pad_token_id':0,
              'eos_token_id':1,
              'missing_token_id':2,
              'mask_token_id':3,
              'mask2_token_id':4,
              'mask3_token_id':5,
              'mask4_token_id':6,
              'mask5_token_id':7
              },
      
        'others':{'emb_size':64,
              'emb_dim':int(sys.argv[1]),
              'pad_token_id':0,
              'eos_token_id':1,
              'missing_token_id':2,
              'mask_token_id':3,
              'mask2_token_id':4,
              'mask3_token_id':5,
              'mask4_token_id':6,
              'mask5_token_id':7
              },

        'labs':{'emb_size':128,
              'emb_dim':int(sys.argv[1]),
              'pad_token_id':0,
              'eos_token_id':1,
              'missing_token_id':2,
              'mask_token_id':3,
              'mask2_token_id':4,
              'mask3_token_id':5,
              'mask4_token_id':6,
              'mask5_token_id':7
              },

        'future_to_predict':3,

        'd_ff': int(sys.argv[2]),
        'd_kv': int(sys.argv[3]),
        'd_model': int(sys.argv[4]),
        'num_heads': int(sys.argv[5]),
        'num_layers': int(sys.argv[6]),
        
        
        'mode': sys.argv[8], # data mode / training mode / prescription recommendation mode
        'msa_num':int(sys.argv[10]), # how many msas (apart from original seq) shall be used at most
        'neighbor_max_distance':float(sys.argv[11]), # set a constraint for the max distance from original seq to be a msa seq in version V3
        
        
        'mlm':False,
        'mlm_dup_factor':9,
        'eval_epochs':5, # first don't do eval to save time
        'save_epochs':5,#100,
        'training_epoch':250, # joint MLM is slower
        'batchsize':128,#int(sys.argv[15]),#256,
        'eval_batchsize':512,#int(sys.argv[15]),
        #'wd':0.01,
        'wd':float(sys.argv[7]),
        'lr':float(sys.argv[9]),
        'gradient_accumulation':int(sys.argv[12]),
        
        'encoder_num_layers':int(sys.argv[13]),
        'encoder_num_heads':int(sys.argv[14]),
        
        
        'config_path':'/data2/brhao/anomaly_project/our_bert_config/',
        'ckpt':None,

        }



config=BertConfig.from_pretrained(our_config['config_path']) # NOTE: the BERT (or other modules apart from GPT) was used during the development process and partially used as placeholders in this script, and will not be actually loaded/trained

print(config)



config.intermediate_size=our_config['d_ff']
config.hidden_size=our_config['d_model']
config.num_attention_heads=our_config['num_heads']
config.num_hidden_layers=our_config['num_layers']
config.vocab_size=500

print(config)



mission_name='anomaly_BMC'



writer_built=False

if our_config['mode']=='train': # only generating csv during training

  our_config['out_folder']='ckpt/'+mission_name+'/'
  #our_config['out_folder']='/data2/brhao/anomaly_project/ckpt'


  #out = open('/mnt/2080ti/data/brhao/hyp_project/ckpt/'+mission_name+'.csv', 'a', newline='',encoding='utf-8') # performance recording file
  out = open(mission_name+'.csv', 'a', newline='',encoding='utf-8') # performance recording file
  csv_write = csv.writer(out, dialect='excel')
  writer_built=True








all_features={'age':['age'],
              
              

              'vitals':['Temp',
              'SBP',
              'DBP',
              'Pulse',
              'SpO2',
              'Resp',
              
              'icu_records',
              'intu_records',
              'death_records',
              
              'age',
              'gender_female',
              'hispanic',
              'race_black',
              'race_white',
              'race_other',
              'language_eng',
              
              'diabetes',
              'htn',
              'ckd',
              'chd',
              'vd_deficiency',
              'obesity',
              'exam_with_abnormal',
              'medical_facilities',
              'reflux',
              'anemia',
              'other_specified_health_status',
              'other_specified_counseling',
              'personal_risk_factors',


              
              ],
           
              'categ':['icu_records',
              'intu_records',
              'death_records',
              
              
              'gender_female',
              'hispanic',
              'race_black',
              'race_white',
              'race_other',
              'language_eng',

              
              'diabetes',
              'htn',
              'ckd',
              'chd',
              'vd_deficiency',
              'obesity',
              'exam_with_abnormal',
              'medical_facilities',
              'reflux',
              'anemia',
              'other_specified_health_status',
              'other_specified_counseling',
              'personal_risk_factors',

              ],


              'informational':['age',
                              'gender_female',
                              'hispanic',
                              'race_black',
                              'race_white',
                              'race_other',
                              'language_eng',

                              'diabetes',
                              'htn',
                              'ckd',
                              'chd',
                              'vd_deficiency',
                              'obesity',
                              'exam_with_abnormal',
                              'medical_facilities',
                              'reflux',
                              'anemia',
                              'other_specified_health_status',
                              'other_specified_counseling',
                              'personal_risk_factors',

                              
                                ], # these features only provide information, they will not be masked, but not to be predicted


              'outcome':['icu_records', # these features will be predicted, and will be always masked in input so they provide no information 
              'intu_records',
              'death_records',
              
              ],


              
              'med':['prescription_CCB',
              'prescription_Thiazides',
              'prescription_ARB',
              'prescription_ACEI',
              'prescription_Beta-Blockers',
              'prescription_Loops',
              'prescription_Mineralocorticoid_receptor_antagonists'],

              'others':['MSDRG:470',
              'MSDRG:378',
              'MSDRG:189',
              'MSDRG:190',
              'MSDRG:247',
              'MSDRG:251',
              'MSDRG:280',
              'MSDRG:291',
              'MSDRG:689',
              'MSDRG:690',
              'MSDRG:698',
              'SVC:CARDIOTHORACIC SURGERY',
              'SVC:NEUROSURGERY',
              'SVC:NEUROLOGY',
              'CPT:87086',
              'CPT:99214',
              'ICD10-CM:I10',
              'ICD9-CM:V04.81',
              'ICD9-PCS:13.41',
              'ICD9-CM:963.1',
              'ICD9-CM:366.16',
              'ICD10-PCS:0392',
              'ICD9-CM:E000.8',
              'ICD9-PCS:00.40',
              'ICD9-CM:V58.69',
              'ICD9-CM:745.5',
              'ICD9-PCS:93.90',
              'ICD9-CM:288.60',
              'ICD9-CM:070.54',
              'ICD9-CM:250.00',
              'ICD10-CM:F79',
              'ICD10-CM:R50.9',
              'ICD10-CM:A63.0',
              'ICD10-CM:I87.2',
              'ICD10-CM:Z79.01',
              'ICD10-CM:F20.0',
              'ICD10-CM:W19.XXXA',
              'ICD10-CM:T82.898A',
              'ICD9-CM:V17.3',
              'ICD10-CM:Z82.49',
              'ICD10-CM:I25.10',
              'ICD10-CM:I73.9',
              'ICD10-CM:I24.8',
              'ICD10-CM:Z45.2',
              'ICD10-CM:I13.0',
              'ICD10-CM:I50.22',
              'ICD10-CM:I11.0',
              'ICD10-CM:Z95.2',
              'ICD10-CM:O76',
              'RXCUI:824194',
              'RXCUI:1298368',
              'RXCUI:199220',
              'sex_M',
              'race_White',
              'race_Black',
              'past_2y_record_CCB',
              'past_2y_record_Thiazides',
              'past_2y_record_ARB',
              'past_2y_record_ACEI',
              'past_2y_record_Beta-Blockers',
              'past_2y_record_Loops',
              'past_2y_record_Mineralocorticoid_receptor_antagonists',
              'hist_prescription_CCB',
              'hist_prescription_Thiazides',
              'hist_prescription_ARB',
              'hist_prescription_ACEI',
              'hist_prescription_Beta-Blockers',
              'hist_prescription_Loops',
              'hist_prescription_Mineralocorticoid_receptor_antagonists'],
              
              'labs':['SOMD:78564009',
              'SOMD:271650006',
              'SOMD:60621009',
              'SOMD:103228002|385641008=128975004',
              'SOMD:86290005',
              'SOMD:276885007',
              'SOMD:78564009|246273001=102538003',
              'SOMD:78564009|246273001=33586001',
              'SOMD:78564009|246273001=10904000',
              'LOINC:26474-7',
              'LOINC:26484-6',
              'HDID:18',
              'HDID:35',
              'HDID:15',
              'HDID:13',
              'HDID:11',
              'HDID:234909',
              'HDID:9',
              'HDID:8',
              'HDID:159',
              'HDID:68990',
              'HDID:3133',
              'HDID:64',
              'HDID:65',
              'HDID:1030',
              'HDID:68',
              'HDID:315',
              'HDID:15567',
              'HDID:1031',
              'HDID:2647',
              'HDID:66',
              'HDID:44',
              'HDID:28',
              'HDID:14',
              'HDID:26',
              'HDID:30',
              'HDID:43',
              'HDID:2',
              'HDID:36',
              'HDID:317',
              'HDID:39',
              'HDID:40',
              'HDID:1019',
              'HDID:2427',
              'HDID:2426',
              'HDID:2422',
              'HDID:2420',
              'HDID:2421',
              'HDID:316',
              'HDID:285',
              'HDID:3',
              'HDID:2418',
              'HDID:3889',
              'HDID:325',
              'HDID:324'],
              
              
              'others_need_to_mask':['MSDRG:470', # those features in 'others' that may not be suitable for future input
              'MSDRG:378',
              'MSDRG:189',
              'MSDRG:190',
              'MSDRG:247',
              'MSDRG:251',
              'MSDRG:280',
              'MSDRG:291',
              'MSDRG:689',
              'MSDRG:690',
              'MSDRG:698',
              'SVC:CARDIOTHORACIC SURGERY',
              'SVC:NEUROSURGERY',
              'SVC:NEUROLOGY',
              'CPT:87086',
              'CPT:99214',
              'ICD10-CM:I10',
              'ICD9-CM:V04.81',
              'ICD9-PCS:13.41',
              'ICD9-CM:963.1',
              'ICD9-CM:366.16',
              'ICD10-PCS:0392',
              'ICD9-CM:E000.8',
              'ICD9-PCS:00.40',
              'ICD9-CM:V58.69',
              'ICD9-CM:745.5',
              'ICD9-PCS:93.90',
              'ICD9-CM:288.60',
              'ICD9-CM:070.54',
              'ICD9-CM:250.00',
              'ICD10-CM:F79',
              'ICD10-CM:R50.9',
              'ICD10-CM:A63.0',
              'ICD10-CM:I87.2',
              'ICD10-CM:Z79.01',
              'ICD10-CM:F20.0',
              'ICD10-CM:W19.XXXA',
              'ICD10-CM:T82.898A',
              'ICD9-CM:V17.3',
              'ICD10-CM:Z82.49',
              'ICD10-CM:I25.10',
              'ICD10-CM:I73.9',
              'ICD10-CM:I24.8',
              'ICD10-CM:Z45.2',
              'ICD10-CM:I13.0',
              'ICD10-CM:I50.22',
              'ICD10-CM:I11.0',
              'ICD10-CM:Z95.2',
              'ICD10-CM:O76',
              'RXCUI:824194',
              'RXCUI:1298368',
              'RXCUI:199220']

              }




no_dot_all_features={k:[f.replace('.','_') for f in all_features[k]] for k in all_features}
print(no_dot_all_features)






# --------------------    data loading and preprocessing




df_all = pd.read_csv('../all_vitals_seq.csv')
pkl_save_fn='BMC.pkl'


print(df_all)




# now we are using the toy data, skip some preprocess


all_sample_ind_list=sorted(df_all['sample_ind'].unique().tolist())
#print(all_pid_list)




normalization_factor_dic=pickle.load(open('../mean_std_dic.pkl','rb')) # in v6, use standardization




def extract_dataset_from_df_for_mlm(df, sample_ind_list, mlm=True, max_seq_len=6, interval=None):
  
  
  data=[]
  
  #for pid in pid_list[:1000]:
  for sample_ind in sample_ind_list:
    
    print(sample_ind)
    
    df_sample_ind_ori=df[df['sample_ind']==sample_ind]
    df_sample_ind_ori_normalized=copy.deepcopy(df_sample_ind_ori)
    
    for k in normalization_factor_dic:
      #df_sample_ind_ori_normalized[k]=df_sample_ind_ori_normalized[k]/normalization_factor_dic[k]
      df_sample_ind_ori_normalized[k]=(df_sample_ind_ori_normalized[k]-normalization_factor_dic[k]['mean'])/normalization_factor_dic[k]['std'] # in v6, use standardization
    
    
    
    df_sample_ind=copy.deepcopy(df_sample_ind_ori)
    
    sample_dic={}
    
    sample_dic['PID']=df_sample_ind.loc[df_sample_ind.index[0],'PID']
    sample_dic['Time']=df_sample_ind.loc[df_sample_ind.index[0],'Time'] # use the first time stamp in seq
    sample_dic['for_training']=df_sample_ind.loc[df_sample_ind.index[0],'for_training']
    sample_dic['df_sample_ind_ori']=df_sample_ind_ori
    sample_dic['df_sample_ind_ori_normalized']=df_sample_ind_ori_normalized
    
    sample_dic['seq_len']=df_sample_ind.shape[0]
    

    
    #------------- preprocess for features
    
    
    
    #sample_dic['df_sample_ind_tokenid']=df_sample_ind
    
    sample_dic['df_sample_ind_tokenid']=copy.deepcopy(df_sample_ind_ori_normalized)
    
    sample_dic['eventually_die_after_66h_IncludingHome']=df_sample_ind.loc[df_sample_ind.index[0],'eventually_die_after_66h_IncludingHome']
    sample_dic['eventually_die_after_66h_StrictDischarge']=df_sample_ind.loc[df_sample_ind.index[0],'eventually_die_after_66h_StrictDischarge']
    sample_dic['eventually_die_all_IncludingHome']=df_sample_ind.loc[df_sample_ind.index[0],'eventually_die_all_IncludingHome']
    sample_dic['eventually_die_all_StrictDischarge']=df_sample_ind.loc[df_sample_ind.index[0],'eventually_die_all_StrictDischarge']
    sample_dic['eventually_die_within3weeks_IncludingHome']=df_sample_ind.loc[df_sample_ind.index[0],'eventually_die_within3weeks_IncludingHome']
 

    sample_dic['is_covid_adm']=df_sample_ind.loc[df_sample_ind.index[0],'is_covid_adm'] # add covid adm indicator in v6.2

    sample_dic['adm_index']=df_sample_ind.loc[df_sample_ind.index[0],'adm_index'] # record the adm_index in v7, for later case study

    #-------------
    
    
    data.append(sample_dic)
  
  return data






def pad_seq(seq,max_seq_len):
  
  if seq.shape[0]>max_seq_len:
    print('wrong data, please check')
    aa
  
  elif seq.shape[0]==max_seq_len:
    return seq
  
  else:
  
    while 1:
      seq.loc[seq.index[-1]+1]=seq.loc[seq.index[-1]]
      
      if seq.shape[0]==max_seq_len:
        break
    
    return seq
  
  




def mask_one_sample_causal(sample_dic, position=None):

  # predict the next-period features using a GPT style!!
  
  ori_normalized=copy.deepcopy(sample_dic['df_sample_ind_ori_normalized'])
  ori_normalized=pad_seq(seq=ori_normalized,max_seq_len=max_seq_len)
  
  ori_normalized_left_shifted=copy.deepcopy(ori_normalized)
  
  ori_normalized_left_shifted.loc[ori_normalized_left_shifted.index[-1]+1,:]=ori_normalized.loc[ori_normalized.index[-1],:]
  ori_normalized_left_shifted=ori_normalized_left_shifted.loc[ori_normalized_left_shifted.index[1:],:]
  #ori_normalized_left_shifted.loc[ori_normalized_left_shifted.index[1:],:].values=ori_normalized.loc[ori_normalized.index[:-1],:].values # left shift the groundtruth!
  
  masked_tokenid=copy.deepcopy(sample_dic['df_sample_ind_tokenid'])
  masked_tokenid=pad_seq(seq=masked_tokenid,max_seq_len=max_seq_len)
  
  #masked_tokenid[all_features['outcome']]=our_config['vitals']['mask1_token_id'] # those outcomes shall not be used as input or provide any information
  masked_tokenid[all_features['outcome']]=0 # now we mask outcomes as 0 in input so they are not informative at all
  
  mask=masked_tokenid*0
  
  #mask.loc[mask.index[:-1],:]=1 # will not predict for the last hidden state, since we don't know the future groundtruth
  mask.loc[mask.index[:sample_dic['seq_len']-1],:]=1 # in v5.5, include short seqs. if seq_len=7, we only care about the first 6 prediction, since we have no groundtruth for 7th
  mask[all_features['informational']]=0 # these informational features like demo will not contribute to the loss
  

  
  
  return (masked_tokenid[all_features['vitals']].values, 
          mask[all_features['vitals']].values, 
          ori_normalized_left_shifted[all_features['vitals']].values,
          sample_dic['eventually_die_within3weeks_IncludingHome'],
          sample_dic['seq_len'],
          )









if our_config['mode']=='data':

  data=extract_dataset_from_df_for_mlm(df=df_all, sample_ind_list=all_sample_ind_list, mlm=True, max_seq_len=max_seq_len, interval=None)
  pickle.dump(data,open(pkl_save_fn,'wb'))

  aa



# ---------------------------------- data separate


#drg_standard_dic={}


def assert_drg(sample):
  
  drg_exclude_dic={'APR-DRGV36': [10,20,55,56,57,135,384,770,772,773,774,775,776,841,842,843,844,910,911,912,930], 
                    'MS-DRG V37 (FY 2020)': [82,83,84,85,86,87,88,89,90,183,184,185,604,605,894,895,896,897,901,902,903,904,905,906,907,908,909,913,914,927,928,929,933,934,935,955,956,957,958,959,963,964,965], 
                    'None': [-99], # "None" and -99 indicate a missing drg in our dataset. 61904 cases have non-missing drg
                    'HCFA': [], 
                    'APR-DRG V30': [10,20,55,56,57,135,384,770,772,773,774,775,776,841,842,843,844,910,911,912,930], 
                    'APR-DRG V34': [10,20,55,56,57,135,384,770,772,773,774,775,776,841,842,843,844,910,911,912,930], 
                    'MS-DRG V36 (FY 2019)': [82,83,84,85,86,87,88,89,90,183,184,185,604,605,894,895,896,897,901,902,903,904,905,906,907,908,909,913,914,927,928,929,933,934,935,955,956,957,958,959,963,964,965], 
                    'MS-DRG V38 (FY 2021)': [82,83,84,85,86,87,88,89,90,183,184,185,604,605,894,895,896,897,901,902,903,904,905,906,907,908,909,913,914,927,928,929,933,934,935,955,956,957,958,959,963,964,965], 
                    'MS-DRG V35 (FY 2018)': [82,83,84,85,86,87,88,89,90,183,184,185,604,605,894,895,896,897,901,902,903,904,905,906,907,908,909,913,914,927,928,929,933,934,935,955,956,957,958,959,963,964,965], 
                    'APR-DRG V38': [10,20,55,56,57,135,384,770,772,773,774,775,776,841,842,843,844,910,911,912,930], 
                    'MS-DRG V40 (FY 2023)': [82,83,84,85,86,87,88,89,90,183,184,185,604,605,894,895,896,897,901,902,903,904,905,906,907,908,909,913,914,927,928,929,933,934,935,955,956,957,958,959,963,964,965], 
                    'APR-DRG V26': [10,20,55,56,57,135,384,770,772,773,774,775,776,841,842,843,844,910,911,912,930], 
                    'MS-DRG V39 (FY 2022)': [82,83,84,85,86,87,88,89,90,183,184,185,604,605,894,895,896,897,901,902,903,904,905,906,907,908,909,913,914,927,928,929,933,934,935,955,956,957,958,959,963,964,965],
                    
                    
                    } # for each drg standard, we list the code to exclude, such as trauma


 
  drg_code_list=eval(sample['df_sample_ind_ori'].iloc[0]['drg_code'])
  is_pandemic_related_drg=False
  
  for (drg_standard,drg) in drg_code_list:
    if drg not in drg_exclude_dic[drg_standard]:
      
      is_pandemic_related_drg=True # even if there is only one drg that is pandemic-related in the corresponding drg standard, the whole adm is related to pandemic
      break

  
  return is_pandemic_related_drg





data=pickle.load(open(pkl_save_fn,'rb'))
print('data loaded')




train=[i for i in data if i['for_training']==1]
val=[i for i in data if i['for_training']==0]
test=[i for i in data if i['for_training']==0] # data is really limited, we don't set val set
#test=[i for i in data if i['day']>4000]

old_people_data=[i for i in data if i['df_sample_ind_ori'].iloc[0]['age']>=60]

drg_data=[i for i in data if assert_drg(sample=i)==True]


print('train sample num:', len(train))
print('val sample num:', len(val))
print('test sample num:', len(test))
print('old people sample num:', len(old_people_data))
print('pandemic-related drg sample num:', len(drg_data))





# --------------------------------- build model





class our_bert(BertModel): # NOTE: the BERT here was used during the development process and partially used as placeholders in this script, which will not be actually loaded/trained
    def __init__(self, config):
        super().__init__(config)
    
        causal_encoder_layer=torch.nn.TransformerEncoderLayer(d_model=config.hidden_size, 
                                                                      nhead=config.num_attention_heads, 
                                                                      batch_first=True,
                                                                      dim_feedforward=config.intermediate_size,
                                                                      dropout=0.1,#0, # add dropout in smaller model
                                                                      )
        
        
        encoder_norm = torch.nn.LayerNorm(config.hidden_size, 
                                              eps=1e-5,
                                              )
        
        self.causal_encoder=torch.nn.TransformerEncoder(encoder_layer=causal_encoder_layer, 
                                                        num_layers=config.num_hidden_layers, 
                                                        norm=encoder_norm,
                                                        )
        
        
        
        self.pooler_dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = torch.nn.Tanh()
        
        self.vitals_emb_dic=torch.nn.ModuleDict({name:torch.nn.Embedding(our_config['vitals']['emb_size'], our_config['vitals']['emb_dim']) for name in no_dot_all_features['vitals']})
        
         
        #self.embed_map = torch.nn.Linear(our_config['vitals']['emb_dim']*len(all_features['vitals']), config.hidden_size) # we also share this layer for decoder
        self.embed_map = torch.nn.Linear(len(all_features['vitals']), config.hidden_size) # now we don't use tokenization!
        
        #self.decoder_embed_map = torch.nn.Linear(our_config['bp']['emb_dim']+our_config['med']['emb_dim']*len(all_features['med']), config.d_model)
        
        print(self.embed_map)
        #self.regression_head = torch.nn.Linear(config.hidden_size, 1, bias=True) # emb16.2: add this regression head



        final_layer={}
        for name in no_dot_all_features['vitals']:
          if name not in all_features['categ']:
            final_layer[name]=torch.nn.Linear(config.hidden_size, 1, bias=True)
          else:
            final_layer[name]=torch.nn.Linear(config.hidden_size, 2, bias=True)
        
        self.regression_head_dic=torch.nn.ModuleDict(final_layer)
        
        self.seq_death_classification_head=torch.nn.Linear(config.hidden_size, 2, bias=True) # use the last hidden state to do seq classification
        #self.regression_head_dic=torch.nn.ModuleDict({name:torch.nn.Linear(config.hidden_size, 1, bias=True) for name in no_dot_all_features['vitals']})
        
        print(self.regression_head_dic)

        self.init_weights()  # use the way huggingface initialization to initialize our layers!!!!
        



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask=None,
        ori_normalized=None,
        eventually_die=None,
        seq_len=None,
        ignore_index=-100,
        
        
    ):

        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        # ------------------------------------------------ use our embedding layers
            
        
        inputs_embeds=self.embed_map(inputs_embeds) # now we directly use input vectors

        embedding_output = self.embeddings(
            input_ids=None, # we force the model to use inputs_embeds
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # ----------------------------------------------------------------------------- end
        
        
        sequence_output=self.causal_encoder(src=embedding_output,
                                            is_causal=True, # is_causal: make sure the encoder is unidirectional, so that we only use past clinical features to predict future. This guarantees the GPT training style
                                            mask=torch.nn.Transformer.generate_square_subsequent_mask(max_seq_len).cuda(),
                                            )
        
        #seq_repr_vector=sequence_output[:,-1,:] # use the last hidden state to do seq death classification
        seq_repr_vector=torch.stack([sequence_output[i,seq_len[i]-2,:] for i in range(seq_len.shape[0])], 0) # in v5.5, use the last valid hidden state
        seq_death_classification_pred=self.seq_death_classification_head(seq_repr_vector)
        
        death_prob=torch.nn.functional.softmax(seq_death_classification_pred,dim=1)[:,1]
        
        regression_pred=[self.regression_head_dic[no_dot_all_features['vitals'][num]](sequence_output) for num in range(len(all_features['vitals']))]
        #regression_pred=torch.cat(tuple(regression_pred), -1)
        
        #pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        
        
        loss = None

        if ori_normalized is not None:

            mixed_loss=[]
            
            
            if self.training:
            
              #seq_classification_ce_loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1,5]).cuda(),reduce=False)
              seq_classification_ce_loss = torch.nn.CrossEntropyLoss(reduce=False)
              mae_loss=torch.nn.L1Loss(reduce=False)
              #ce_loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1,5]).cuda(),reduce=False)
              ce_loss = torch.nn.CrossEntropyLoss(reduce=False)
              
              
              seq_death_classification_loss = seq_classification_ce_loss(seq_death_classification_pred.view(-1, 2), eventually_die.view(-1))
              
              for i, name in enumerate(all_features['vitals']):
                if name not in all_features['categ']:
                  mixed_loss.append(mae_loss(regression_pred[i][:,:,0],ori_normalized[:,:,i]))

                else:
                  mixed_loss.append(ce_loss(regression_pred[i].view(-1,2),ori_normalized[:,:,i].view(-1).type(torch.LongTensor).cuda()).view(ori_normalized.shape[0],ori_normalized.shape[1]))
              
              
              mixed_loss=torch.stack(tuple(mixed_loss), -1)
              
            else:
              seq_classification_ce_loss = torch.nn.CrossEntropyLoss(reduce=False,ignore_index=ignore_index)
              seq_death_classification_loss = seq_classification_ce_loss(seq_death_classification_pred.view(-1, 2), eventually_die.view(-1))

              mae_loss=torch.nn.L1Loss(reduce=False)
              ce_loss = torch.nn.CrossEntropyLoss(reduce=False,ignore_index=ignore_index)
              

              for i, name in enumerate(all_features['vitals']):
                if name not in all_features['categ']:
                  error=regression_pred[i][:,:,0]-ori_normalized[:,:,i]
                  mixed_loss.append(1*abs(error))
                    
                else:
                  mixed_loss.append(1*ce_loss(regression_pred[i].view(-1,2),ori_normalized[:,:,i].view(-1).type(torch.LongTensor).cuda()).view(ori_normalized.shape[0],ori_normalized.shape[1]))
              
              mixed_loss=torch.stack(tuple(mixed_loss), -1)

            
            mixed_loss=mixed_loss*mask
            
            all_period_loss=mixed_loss # add this in v5.5
            
            #all_loss=torch.sum(mixed_loss,dim=[-2,-1])/torch.sum(mask,dim=[-2,-1]) # this is for outputing the loss (anomaly score) for each sample
            all_loss=torch.sum(mixed_loss,dim=[-2])/torch.sum(mask,dim=[-2]) # only average across periods, so we save loss for each feature!

            all_loss=torch.cat([all_loss,seq_death_classification_loss.reshape(-1,1)],dim=1) # IMPORTANT:   the last dim of loss is the seq classification loss!!!
            
            #print(all_loss.shape)
            #aa
            
            ################ IMPORTANT ###################
            
            # for informative features, since mask value for each period are always 0, 0/0 results in nan in all_loss. 
            
            ##############################################
            
            loss=torch.sum(mixed_loss)/torch.sum(mask)+torch.mean(seq_death_classification_loss) # this is for optimization, later we may think about if we should direcetly average over samples rather than tokens, in case that there is mask imbalance in future


        return loss, all_loss, all_period_loss, death_prob,#, regression_output_pred, repr_out








def pred_one_dataset_batch_causal(model,dataset,batchsize=our_config['eval_batchsize'],ignore_index=-100):

  model.eval()
  
  ALL_LOSS=[]
  ALL_PERIOD_LOSS=[]
  ALL_death_GT=[]
  ALL_death_PRED=[]


  eval_index=[i for i in range(len(dataset))]
  
  for r in range(int(len(dataset)/batchsize)+1): # +1: fix the previous bug
    
    print(r)
    
    #if r<760:
      #continue


    ind_slice=eval_index[r*batchsize:(r+1)*batchsize]
    
    if ind_slice==[]:
      continue

    sample_batch=[dataset[i] for i in ind_slice]
    
    masked=[mask_one_sample_causal(sample_dic=i) for i in sample_batch] # for causal training, we don't use [mask] token, but directly predict the next period features!


    #masked_tokenid=torch.LongTensor([i[0] for i in masked]).cuda()
    masked_tokenid=torch.Tensor([i[0] for i in masked]).cuda() # now they are real values
    mask=torch.LongTensor([i[1] for i in masked]).cuda() # do not mask the last period since we are predicting nothing
    ori_normalized_left_shifted=torch.Tensor([i[2] for i in masked]).cuda() # left-shift the original features, so the current-period hidden state is actually predicting the next period features!
    
    eventually_die=torch.LongTensor([i[3] for i in masked]).cuda() # death label for seq classification
    seq_len=torch.LongTensor([i[4] for i in masked]).cuda() # record the seq len
    
    loss, all_loss, all_period_loss, death_prob=model(inputs_embeds=masked_tokenid, 
                                      mask=mask, 
                                      ori_normalized=ori_normalized_left_shifted, 
                                      eventually_die=eventually_die,
                                      seq_len=seq_len,
                                      ignore_index=ignore_index,
                                      )
    
    
    
    ALL_LOSS.extend(all_loss.cpu().detach().numpy().tolist())
    ALL_PERIOD_LOSS.extend(all_period_loss.cpu().detach().numpy().tolist())
    ALL_death_GT.extend(eventually_die.cpu().detach().numpy().tolist())
    ALL_death_PRED.extend(death_prob.cpu().detach().numpy().tolist())

  
  
  ALL_death_PRED_LABEL=[]
  for p in ALL_death_PRED:
    if p>0.5:
      ALL_death_PRED_LABEL.append(1)
    else:
      ALL_death_PRED_LABEL.append(0)
      
  print(np.mean(ALL_death_PRED_LABEL))
  print(np.mean(ALL_death_GT))
  print('AUC:',roc_auc_score(ALL_death_GT,ALL_death_PRED))
  
  
  
  TIME=[sample['Time'] for sample in dataset]
  
  SEQ_LEN=[sample['seq_len'] for sample in dataset]
  
  IS_COVID_ADM=[sample['is_covid_adm'] for sample in dataset]
  
  return ALL_LOSS, ALL_PERIOD_LOSS, TIME, SEQ_LEN, IS_COVID_ADM




# load/initialize models -----------------------------------------


if our_config['ckpt']:
  model = our_bert.from_pretrained(our_config['ckpt']).cuda() # NOTE: the BERT (or other modules apart from GPT) was used during the development process and partially used as placeholders in this script, and will not be actually loaded/trained
else:
  model = our_bert(config).cuda()  # initialize from config, without using pretrained weights


model.train()


if our_config['mode']=='train':
  Training=True
else:
  Training=False




# training ----------------------------------------



optimizer = transformers.AdamW(model.parameters(), lr=our_config['lr'], weight_decay=our_config['wd'])



batchsize=our_config['batchsize']
all_index=[i for i in range(len(train))]




for e in range(1,our_config['training_epoch']+1): 
    
    if Training==False:
        break
    
    # training for each epoch -----------------------------------
    
    model.train()

    random.shuffle(all_index)
    #for r in range(2):
    for r in range(int(len(train)/batchsize)): # no +1: in training, make sure the batchsize is stable
        
        ind_slice=all_index[r*batchsize:(r+1)*batchsize]
        
        
        #print(ind_slice)
        
        sample_batch=[train[i] for i in ind_slice]
        
        masked=[mask_one_sample_causal(sample_dic=i) for i in sample_batch] # for causal training, we don't use [mask] token, but directly predict the next period features!

        #masked_tokenid=torch.LongTensor([i[0] for i in masked]).cuda()
        masked_tokenid=torch.Tensor([i[0] for i in masked]).cuda() # now they are real values
        mask=torch.LongTensor([i[1] for i in masked]).cuda() # do not mask the last period since we are predicting nothing
        ori_normalized_left_shifted=torch.Tensor([i[2] for i in masked]).cuda() # left-shift the original features, so the current-period hidden state is actually predicting the next period features!
        
        eventually_die=torch.LongTensor([i[3] for i in masked]).cuda() # death label for seq classification
        seq_len=torch.LongTensor([i[4] for i in masked]).cuda()

        optimizer.zero_grad()

        
        #loss, all_loss=model(inputs_embeds=masked_tokenid, mask=mask, ori_normalized=ori_normalized)
        loss, all_loss, all_period_loss, death_prob=model(inputs_embeds=masked_tokenid, 
                                          mask=mask, 
                                          ori_normalized=ori_normalized_left_shifted, 
                                          eventually_die=eventually_die,
                                          seq_len=seq_len,
                                          ignore_index=-100,
                                          
                                          ) # recovery the shifted features to predict future
        
        #loss, output, repr_ = model(inputs_embeds=X_slice, labels=y_slice, labels_realvalue=y_realvalue_slice)#[:2]
        #loss, output = model(inputs_embeds=check_emb, labels=check_y)[:2]
        l_numerical = loss.item()
        
        loss.backward()
        optimizer.step()

        print(f"Epoch: {e}, Loss: {l_numerical}")

    print(f"Epoch: {e}, Loss: {l_numerical}")

    
    # eval for certain epochs -----------------------------------
    
    if e%our_config['eval_epochs']==0:# and our_config['mlm']==False:
        #continue
      
      out_line=[e]
      

      
      print('Now eval train set')

      #ALL_LOSS=pred_one_dataset_batch(model,dataset=val)
      ALL_LOSS, ALL_PERIOD_LOSS, TIME, SEQ_LEN, IS_COVID_ADM=pred_one_dataset_batch_causal(model,dataset=train,ignore_index=-100)
      print('Now eval test set')
      ALL_LOSS, ALL_PERIOD_LOSS, TIME, SEQ_LEN, IS_COVID_ADM=pred_one_dataset_batch_causal(model,dataset=test,ignore_index=-100)
      

        

    # save model and the output for anomaly detection ----------------------------------------------

    tar_folder=our_config['out_folder']
    
    try:
        os.makedirs(tar_folder)
    except FileExistsError:
        print('Folder already there')
    model.save_pretrained(tar_folder+"anomaly_ep"+str(e))


    if e%our_config['save_epochs']==0:
      
      for ignore_index in [-100]:
        
        ALL_LOSS, ALL_PERIOD_LOSS, TIME, SEQ_LEN, IS_COVID_ADM=pred_one_dataset_batch_causal(model,dataset=data,ignore_index=ignore_index)
        
        pickle.dump([ALL_LOSS, ALL_PERIOD_LOSS, TIME, SEQ_LEN, IS_COVID_ADM],open('all_loss_ep'+str(e)+'_IgnoreIndex_'+str(ignore_index)+'.pkl','wb'))
        




if our_config['mode']=='train': # there is no 'out' if not training
  out.close()







# ------------------------------- check the anomaly detection



from datetime import datetime, timedelta

def difference_time_A_B_seconds(timeA,timeB,format_pattern="%m/%d/%Y %H:%M"):
  
  
  #print(timeA)
  #print(timeB)
  
  gap_seconds=(datetime.strptime(timeA,format_pattern) - datetime.strptime(timeB,format_pattern)).total_seconds()
  
  return gap_seconds




test_500_days=[i for i in data if difference_time_A_B_seconds(timeA=i['Time'],timeB='01/01/2019 00:00',format_pattern="%m/%d/%Y %H:%M")>0 and difference_time_A_B_seconds(timeA=i['Time'],timeB='01/01/2019 00:00',format_pattern="%m/%d/%Y %H:%M")<500*24*3600]


print(len(test_500_days))


if our_config['mode']=='pred':
  
  
  #for model_ep in [55,50,45,40,35,30,25,20,15,10,5]:
  for model_ep in [100]:
  
    model = our_bert.from_pretrained('ckpt/anomaly_BMC/anomaly_ep'+str(model_ep)+'/').cuda()
    #model = our_bert.from_pretrained('/data2/brhao/anomaly_project/BMC_anomaly_v4/anomaly_model/ckpt/anomaly_BMC/anomaly_ep6/').cuda()
    model.eval()
  
    print('Now eval test set')
    
    
    for ignore_index in [-100]:
    
    
      #ALL_LOSS, ALL_PERIOD_LOSS, TIME, SEQ_LEN, IS_COVID_ADM=pred_one_dataset_batch_causal(model,dataset=data,ignore_index=ignore_index)
      ALL_LOSS, ALL_PERIOD_LOSS, TIME, SEQ_LEN, IS_COVID_ADM=pred_one_dataset_batch_causal(model,dataset=test_500_days,ignore_index=ignore_index)
      
      print('test LOSS: ',np.mean(ALL_LOSS))
      
      assert (len(TIME)==len(ALL_LOSS))
      











