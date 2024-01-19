



import csv
import os
import numpy as np
import pandas as pd
import copy
import pickle
import random
from datetime import datetime, timedelta
import re


random.seed(2)




feature_dic_empty={'SBP':[],
                    'DBP':[],
                    'Pulse':[],
                    'Temp':[],
                    'SpO2':[],
                    'Resp':[],



                    'adm_records':[],
                    'icu_records':[],
                    'intu_records':[],
                    'death_records':[],
                    
                    'adm_start':[],
                    'adm_records_time':[],
                    'icu_records_time':[],
                    'intu_records_time':[],
                    'death_records_time':[],


                    'age':[],
                    'gender_female':[],
                    'hispanic':[],
                    'race_black':[],
                    'race_white':[],
                    'race_other':[],
                    'language_eng':[],
                    
                    
                    'diabetes':[],
                    'htn':[],
                    'ckd':[],
                    'chd':[],
                    'vd_deficiency':[],
                    'obesity':[],
                    'exam_with_abnormal':[],
                    'medical_facilities':[],
                    'reflux':[],
                    'anemia':[],
                    'other_specified_health_status':[],
                    'other_specified_counseling':[],
                    'personal_risk_factors':[],
                    

                    'adm_index':[],
                    'drg_code':[],                    

                    }


feature_type={'continuous':{'SBP':0,
                    'DBP':0,
                    'Pulse':0,
                    'Temp':0,
                    'SpO2':0,
                    'Resp':0,
                    'age':0,
                    
                    'adm_index':0, # will be averaged
                    },
              
              
              
              'binary':{'adm_records':0,
                        'icu_records':0,
                        'intu_records':0,
                        'death_records':0,
                        
                        'adm_start':0,
                        
                        'gender_female':0,
                        'hispanic':0,
                        'race_black':0,
                        'race_white':0,
                        'race_other':0,
                        'language_eng':0,
                        
                        'diabetes':0,
                        'htn':0,
                        'ckd':0,
                        'chd':0,
                        'vd_deficiency':0,
                        'obesity':0,
                        'exam_with_abnormal':0,
                        'medical_facilities':0,
                        'reflux':0,
                        'anemia':0,
                        'other_specified_health_status':0,
                        'other_specified_counseling':0,
                        'personal_risk_factors':0,
                        
                        
                        },
              
              
              }



demo_feature_list=['age',
                  'gender_female',
                  'hispanic',
                  'race_black',
                  'race_white',
                  'race_other',
                  'language_eng',
                  ]



pmh_feature_list=['diabetes',
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

                  ]




outcome_gaps={'adm_records':[],
              'icu_records':[],
              'intu_records':[],
              'death_records':[],
                        

                        }







data_dic=pickle.load(open('all_vitals.pkl','rb'))
all_PID=sorted(list(data_dic.keys()))

demo_data_dic=pickle.load(open('all_demo.pkl','rb'))

pmh_data_dic=pickle.load(open('all_pmh.pkl','rb'))

covid_data_dic=pickle.load(open('all_covid.pkl','rb'))




tau=4 # use 6h as one period in time series # v6_4h: use 4h
time_relaxation_hr=1*tau # use the adm time - ktau to include pre-adm features!




print(len(all_PID))


def difference_time_A_B_seconds(timeA,timeB,format_pattern="%m/%d/%Y %H:%M"):

  gap_seconds=(datetime.strptime(timeA,format_pattern) - datetime.strptime(timeB,format_pattern)).total_seconds()
  
  return gap_seconds



def A_plus_hrs(timeA='05/12/2020 14:34', hr=2,format_pattern="%m/%d/%Y %H:%M"):

	r=(datetime.strptime(timeA,format_pattern)+timedelta(hours=hr)).strftime(format_pattern)

	return str(r)




averaged_data_dic={}

adm_index=0 # this is a unique number for each adm gap, no matter from a different or same patients, in order to avoid the adm record overlapping caused by 6hr relax pre adm

die_eventually_66h_StrictDischarge_dic={} 
die_eventually_66h_IncludingHome_dic={} # for each unique adm_index, record if the patient die eventually after 72h since relaxed adm

die_eventually_all_StrictDischarge_dic={} 
die_eventually_all_IncludingHome_dic={}
die_eventually_within3weeks_IncludingHome_dic={}

is_covid_adm_dic={}


#for i,PID in enumerate(all_PID[:100]):
for i,PID in enumerate(all_PID):
  
  print(i)
  print(PID)


  
  
  
  #print(data_dic[PID])
  
  all_time=[] # collect all vitals time, so later use the earliest one as the initial time of the seq

  
  all_adm_time=[(t['time'],t['time_end']) for t in data_dic[PID]['adm_records']]
  all_adm_time=list(set(all_adm_time))
  
  
  adm_time_drg_dic={g:[] for g in all_adm_time} # since one adm can have multiple drg codes, we use a list to record all of them!!!!
  
  for t in data_dic[PID]['adm_records']:
    adm_time_drg_dic[(t['time'],t['time_end'])].append(t['drg_code'])
    

  
  all_adm_time.sort(key=lambda date: datetime.strptime(date[0], "%m/%d/%Y %H:%M")) # sort the (adm_start,adm_end) pairs by the adm_start time
  
  #print(all_adm_time)
  
  
  #if all_time==[]:
  if all_adm_time==[]: # now we only care about the adm records
    print(str(PID)+' has no valid features!!!')
    continue

  
  
  features_wrt_gaps={}
  
  for (time,time_end) in all_adm_time:
    
    adm_index+=1
    drg_code=adm_time_drg_dic[(time,time_end)]
    
    
    # ------------------ assert if it is covid admission
    
    
    if PID in covid_data_dic:
      
      for covid_pos_time in covid_data_dic[PID]:
        
        if abs(difference_time_A_B_seconds(timeA=time,timeB=covid_pos_time,format_pattern="%m/%d/%Y %H:%M"))<=7*24*3600: # as long as there is one positive covid test in adm_time+=7 days, regarding this adm as covid adm
          
          is_covid_adm_dic[adm_index]=1
          break
    
    
    
    
    # -------------------------------
    
    
    relaxed_adm_time=A_plus_hrs(timeA=time, hr=-time_relaxation_hr,format_pattern="%m/%d/%Y %H:%M")
    first_recorded_time=relaxed_adm_time # now use the adm_time-6hr as the corresponding reference  # in later versions we didn't use relaxation, though keeping an extra row in csv doesn't hurt
  
    for feature in feature_dic_empty:
  
      if feature in ['adm_start','adm_records_time','icu_records_time','intu_records_time','death_records_time','adm_index','drg_code']+demo_feature_list+pmh_feature_list: # these are not in database pkl but will be recorded later
        continue
  
      
      if feature not in ['adm_records','icu_records','intu_records','death_records']:
  
  
        for value_dic in data_dic[PID][feature]:
          
          if difference_time_A_B_seconds(timeA=value_dic['time'],timeB=first_recorded_time,format_pattern="%m/%d/%Y %H:%M")>=0 and difference_time_A_B_seconds(timeA=value_dic['time'],timeB=time_end,format_pattern="%m/%d/%Y %H:%M")<=0: # make sure that the feature is within relaxed adm duration
          
            k=int(difference_time_A_B_seconds(timeA=value_dic['time'],timeB=first_recorded_time,format_pattern="%m/%d/%Y %H:%M")//(tau*3600))
            
            #print(k)
            our_period_stamp=A_plus_hrs(timeA=first_recorded_time, hr=k*tau,format_pattern="%m/%d/%Y %H:%M")
            #print(our_period_stamp)
            
            if our_period_stamp not in features_wrt_gaps:
              features_wrt_gaps[our_period_stamp]=copy.deepcopy(feature_dic_empty)
            
            features_wrt_gaps[our_period_stamp][feature].append(value_dic['value'])
            
            
            #--------------- add demo for each record!
            
            for demo_feature in demo_feature_list:
            
              if demo_feature=='age':
                age=difference_time_A_B_seconds(timeA=our_period_stamp,timeB=demo_data_dic[PID]['birthday'],format_pattern="%m/%d/%Y %H:%M")/(365.25*24*3600)
                features_wrt_gaps[our_period_stamp][demo_feature].append(age)
              else:
                features_wrt_gaps[our_period_stamp][demo_feature].append(demo_data_dic[PID][demo_feature])
              
            #-----------------------------------------
            
            #--------------- add pmh for each record!
            
            for pmh_feature in pmh_feature_list:
              
              if difference_time_A_B_seconds(timeA=pmh_data_dic[PID][pmh_feature],timeB=our_period_stamp,format_pattern="%m/%d/%Y %H:%M")<0: # pmh before the current time stamp
                
                features_wrt_gaps[our_period_stamp][pmh_feature].append(1)
            
            
            #-----------------------------------------
            
            
            features_wrt_gaps[our_period_stamp]['adm_index'].append(adm_index) # record the unique adm index for each adm span
            features_wrt_gaps[our_period_stamp]['drg_code'].append(drg_code)
            
      else:
        
        all_outcome_start_time=[] # later use this to compute mean/median gaps!
        
        #print(feature)
        for value_dic in data_dic[PID][feature]:
          
          if difference_time_A_B_seconds(timeA=value_dic['time'],timeB=first_recorded_time,format_pattern="%m/%d/%Y %H:%M")>=0 and difference_time_A_B_seconds(timeA=value_dic['time_end'],timeB=time_end,format_pattern="%m/%d/%Y %H:%M")<=24*3600: # make sure that the feature is within relaxed adm duration. for icu and intu, maybe the time is before adm, so we also add them into relaxation  # allow a delay of end records
            
            
            all_outcome_start_time.append(value_dic['time'])
            
          
            k=int(difference_time_A_B_seconds(timeA=value_dic['time'],timeB=first_recorded_time,format_pattern="%m/%d/%Y %H:%M")//(tau*3600))
            k_end=int(difference_time_A_B_seconds(timeA=value_dic['time_end'],timeB=first_recorded_time,format_pattern="%m/%d/%Y %H:%M")//(tau*3600))
            
            for ii,num_tau in enumerate(range(k,k_end+1)): # for death, end+1 gives only the beginning period. give every period in a begin/discharge duration the positive label
              
              our_period_stamp=A_plus_hrs(timeA=first_recorded_time, hr=num_tau*tau,format_pattern="%m/%d/%Y %H:%M")
              
              if our_period_stamp not in features_wrt_gaps:
                features_wrt_gaps[our_period_stamp]=copy.deepcopy(feature_dic_empty)
              
              features_wrt_gaps[our_period_stamp][feature].append(value_dic['value'])
              features_wrt_gaps[our_period_stamp][feature+'_time'].append(value_dic['time']) # record the adm/icu/intu/death start time
              
              if ii==0 and feature=='adm_records':
                features_wrt_gaps[our_period_stamp]['adm_start'].append(1) # an indicator that there is at least one adm start point in this period
        
              #--------------- add demo for each record!
              
              for demo_feature in demo_feature_list:
              
                if demo_feature=='age':
                  age=difference_time_A_B_seconds(timeA=our_period_stamp,timeB=demo_data_dic[PID]['birthday'],format_pattern="%m/%d/%Y %H:%M")/(365.25*24*3600)
                  features_wrt_gaps[our_period_stamp][demo_feature].append(age)
                else:
                  features_wrt_gaps[our_period_stamp][demo_feature].append(demo_data_dic[PID][demo_feature])
                
              #-----------------------------------------

              #--------------- add pmh for each record!
              
              for pmh_feature in pmh_feature_list:
                
                if difference_time_A_B_seconds(timeA=pmh_data_dic[PID][pmh_feature],timeB=our_period_stamp,format_pattern="%m/%d/%Y %H:%M")<0: # pmh before the current time stamp
                  
                  features_wrt_gaps[our_period_stamp][pmh_feature].append(1)
              
              
              #-----------------------------------------


              features_wrt_gaps[our_period_stamp]['adm_index'].append(adm_index) # record the unique adm index for each adm span
              features_wrt_gaps[our_period_stamp]['drg_code'].append(drg_code)



         
        if all_outcome_start_time!=[]:
          all_outcome_start_time.sort(key=lambda date: datetime.strptime(date, "%m/%d/%Y %H:%M"))
          first_outcome_start_time=all_outcome_start_time[0]
          outcome_gap=difference_time_A_B_seconds(timeA=first_outcome_start_time,timeB=time,format_pattern="%m/%d/%Y %H:%M")/3600
          print(PID,(time,time_end),first_outcome_start_time,feature,outcome_gap)
          
          outcome_gaps[feature].append([PID,(time,time_end),first_outcome_start_time,outcome_gap])
          
          #if feature=='death_records' and outcome_gap>66: # 66h: 12 periods starting from adm-6h
          if feature=='death_records' and outcome_gap>68: # using gap=4
            die_eventually_66h_IncludingHome_dic[adm_index]=1
            
            if difference_time_A_B_seconds(timeA=value_dic['time_end'],timeB=time_end,format_pattern="%m/%d/%Y %H:%M")<=0:
              die_eventually_66h_StrictDischarge_dic[adm_index]=1


          if feature=='death_records': # 66h: 12 periods starting from adm-6h
            die_eventually_all_IncludingHome_dic[adm_index]=1
            
            if outcome_gap<=504: # die with in 3 weeks
              die_eventually_within3weeks_IncludingHome_dic[adm_index]=1
            
            
            if difference_time_A_B_seconds(timeA=value_dic['time_end'],timeB=time_end,format_pattern="%m/%d/%Y %H:%M")<=0:
              die_eventually_all_StrictDischarge_dic[adm_index]=1


    #features_wrt_gaps[our_period_stamp][feature+'_time'].sort(key=lambda date: datetime.strptime(date, "%m/%d/%Y %H:%M"))



      
  #print(features_wrt_gaps)
  
  for our_period_stamp in features_wrt_gaps:
    for feature in features_wrt_gaps[our_period_stamp]:
      if feature not in ['adm_records_time','icu_records_time','intu_records_time','death_records_time','drg_code']:
        if features_wrt_gaps[our_period_stamp][feature]==[]:
          if feature in feature_type['continuous']:
            features_wrt_gaps[our_period_stamp][feature]=''#-99.0 # if no certain records in a period, record it as missing using -99
          else:
            features_wrt_gaps[our_period_stamp][feature]=0
        else:
          if feature in feature_type['continuous']:
            features_wrt_gaps[our_period_stamp][feature]=np.mean(features_wrt_gaps[our_period_stamp][feature]) # else, using the average of those values
          else:
            features_wrt_gaps[our_period_stamp][feature]=np.max(features_wrt_gaps[our_period_stamp][feature])
      
      else:
        if features_wrt_gaps[our_period_stamp][feature]==[]:
          features_wrt_gaps[our_period_stamp][feature]='N'
        else:
          features_wrt_gaps[our_period_stamp][feature]=features_wrt_gaps[our_period_stamp][feature][0] # just use the first recorded time
  
  
  #print(features_wrt_gaps)
  
  averaged_data_dic[PID]=features_wrt_gaps
  



print('averaged_data_dic complete')



out = open('all_vitals_period.csv', 'a', newline='',encoding='utf-8')
csv_write = csv.writer(out, dialect='excel')

feature_in_head=['SBP','DBP','Pulse','Temp','SpO2','Resp','adm_index','drg_code','adm_records','adm_records_time','adm_start','icu_records','icu_records_time','intu_records','intu_records_time','death_records','death_records_time']+demo_feature_list+pmh_feature_list
csv_write.writerow(['PID','Time']+feature_in_head+['eventually_die_after_66h_IncludingHome','eventually_die_after_66h_StrictDischarge','eventually_die_all_IncludingHome','eventually_die_all_StrictDischarge','eventually_die_within3weeks_IncludingHome','is_covid_adm'])



for PID in sorted(list(averaged_data_dic.keys())):
  
  all_our_period_stamp=list(averaged_data_dic[PID].keys())
  all_our_period_stamp.sort(key=lambda date: datetime.strptime(date, "%m/%d/%Y %H:%M"))
  
  for our_period_stamp in all_our_period_stamp:
    
    if int(averaged_data_dic[PID][our_period_stamp]['adm_index']) in die_eventually_66h_IncludingHome_dic:
      eventually_die_after_66h_IncludingHome=1
    else:
      eventually_die_after_66h_IncludingHome=0


    if int(averaged_data_dic[PID][our_period_stamp]['adm_index']) in die_eventually_66h_StrictDischarge_dic:
      eventually_die_after_66h_StrictDischarge=1
    else:
      eventually_die_after_66h_StrictDischarge=0


    if int(averaged_data_dic[PID][our_period_stamp]['adm_index']) in die_eventually_all_IncludingHome_dic:
      eventually_die_all_IncludingHome=1
    else:
      eventually_die_all_IncludingHome=0


    if int(averaged_data_dic[PID][our_period_stamp]['adm_index']) in die_eventually_all_StrictDischarge_dic:
      eventually_die_all_StrictDischarge=1
    else:
      eventually_die_all_StrictDischarge=0


    if int(averaged_data_dic[PID][our_period_stamp]['adm_index']) in die_eventually_within3weeks_IncludingHome_dic:
      eventually_die_within3weeks_IncludingHome=1
    else:
      eventually_die_within3weeks_IncludingHome=0


    if int(averaged_data_dic[PID][our_period_stamp]['adm_index']) in is_covid_adm_dic:
      is_covid_adm=1
    else:
      is_covid_adm=0

  
    
    out_line=[PID,our_period_stamp]+[averaged_data_dic[PID][our_period_stamp][feature] for feature in feature_in_head]+[eventually_die_after_66h_IncludingHome,eventually_die_after_66h_StrictDischarge,eventually_die_all_IncludingHome,eventually_die_all_StrictDischarge,eventually_die_within3weeks_IncludingHome,is_covid_adm]
    csv_write.writerow(out_line)
    

out.close()


print('csv written complete')




pickle.dump(outcome_gaps,open('outcome_gaps.pkl','wb'))

print('outcome gaps written complete')



















