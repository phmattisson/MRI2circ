# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

input_annotation_file = 'data/pop_norms.csv'
df = pd.read_csv(input_annotation_file, header=0)

## this is code to match the ids of the abcd dataset with the ids of the population norms

## HORMONES
input_annotation_file_xls = 'data/combined_abcd.xlsx'
df_xls =pd.read_excel(open(input_annotation_file_xls, 'rb'),
              sheet_name='abcd_hsss01') 
df_xls = df_xls.drop([0])

list_hrm = []
for idx in range(0, df_xls.shape[0]):
    row_xls = df_xls.iloc[idx]
    #print(idx, row_xls['subjectkey'])
    try:
        key_id = row_xls['subjectkey'].split("_")[0] + row_xls['subjectkey'].split("_")[1]
        for jdx in range(0, df.shape[0]):
            row = df.iloc[jdx]
            if key_id in row['ID'] and row_xls['eventname'].replace("_","").lower() in row['ID'].lower():
                print(key_id,row_xls['eventname'],row['ID'])
                list_hrm.append([key_id, 
                                row['Age'], 
                                row['Gender'], 
                                row_xls['hormone_scr_dhea_mean'],
                                row_xls['hormone_scr_hse_mean'],
                                row_xls['hormone_scr_ert_mean'],
                                row['TMT PRED AVG filtered']])    
                break
    except:
        continue
df_hrm = pd.DataFrame(list_hrm,columns=['id','Age','gender','DHEA','HSE','ERT','TMT PRED AVG filtered'])

df_hrm['Above average DHEA'] = df_hrm['DHEA']>=df_hrm['DHEA'].mean()
df_hrm['Above average HSE'] = df_hrm['HSE']>=df_hrm['HSE'].mean()
df_hrm['Above average ERT'] = df_hrm['ERT']>=df_hrm['ERT'].mean()
df_hrm.to_csv(path_or_buf= "data/ABCD-studies/abcd_hsss01.csv")


## INCOME
input_annotation_file_xls = 'data/combined_abcd.xlsx'
df_xls =pd.read_excel(open(input_annotation_file_xls, 'rb'),
              sheet_name='abcd_lpds01') 
df_xls = df_xls.drop([0])

list_etnic = []
# tried to do a full match as in https://www.geeksforgeeks.org/join-pandas-dataframes-matching-by-substring/
# but it takes much longer
for idx in range(0, df_xls.shape[0]):
    row_xls = df_xls.iloc[idx]
    #print(idx, row_xls['subjectkey'])
    try:
        key_id = row_xls['subjectkey'].split("_")[0] + row_xls['subjectkey'].split("_")[1]
        for jdx in range(0, df.shape[0]):
            row = df.iloc[jdx]
            if key_id in row['ID']:
                if not pd.isna(row_xls['demo_comb_income_v2_l']):
                    list_etnic.append([key_id, 
                                row['Age'], 
                                row['Gender'], 
                                float(row_xls['demo_comb_income_v2_l']),
                                row['TMT PRED AVG filtered']])    
                break
    except:
        continue
    
df_etnic = pd.DataFrame(list_etnic,columns=['id','Age','gender','Total house income','TMT PRED AVG filtered'])            
df_etnic['Income Above $50,000'] = (df_etnic['Total house income']>=7) & (df_etnic['Total house income']<=10)          
df_etnic.to_csv(path_or_buf= "data/ABCD-studies/abcdabcd_lpds01_income.csv")

## ACTIVITY
input_annotation_file_xls = 'data/combined_abcd.xlsx'
df_xls =pd.read_excel(open(input_annotation_file_xls, 'rb'),
              sheet_name='internat_physical_activ01') 
df_xls = df_xls.drop([0])

def to_number(input_x):
    if pd.isna(input_x):
        return 0
    else:
        return float(input_x)
list_ph_act = []

# Light total = ipaq_light_acts * (ipaq_light_acts_min/60 + ipaq_light_acts_hrs)
# Moderate total = ipaq_mod_acts * (ipaq_mod_acts_hrs+ ipaq_mod_acts_min/60)
# Vig total = ipaq_vig_acts * (ipaq_vig_acts_hrs + ipaq_mod_acts_min/60)
# No activity = ipaq_inactive_hrs +ipaq_inactive_min /60

for idx in range(0, df_xls.shape[0]):
    row_xls = df_xls.iloc[idx]
    #print(idx, row_xls['subjectkey'])
    try:
        key_id = row_xls['subjectkey'].split("_")[0] + row_xls['subjectkey'].split("_")[1]
        for jdx in range(0, df.shape[0]):
            row = df.iloc[jdx]
            if key_id in row['ID'] and row_xls['eventname'].replace("_","").lower().replace("followupyarm1","") in row['ID']:
                light = to_number(row_xls['ipaq_light_acts']) * (to_number(row_xls['ipaq_light_acts_min'])/60 + to_number(row_xls['ipaq_light_acts_hrs']))
                moderate = to_number(row_xls['ipaq_mod_acts']) * (to_number(row_xls['ipaq_mod_acts_min'])/60 + to_number(row_xls['ipaq_mod_acts_hrs']))
                vig = to_number(row_xls['ipaq_vig_acts']) * (to_number(row_xls['ipaq_mod_acts_min'])/60 + to_number(row_xls['ipaq_vig_acts_hrs']))
                no_activity = (to_number(row_xls['ipaq_inactive_min'])/60 + to_number(row_xls['ipaq_inactive_hrs']))
                activity_level = 3*vig + 2*moderate + light 
                list_ph_act.append([key_id, 
                                   row['Age'], 
                                   row['Gender'], 
                                    vig,
                                    moderate,
                                    light,
                                    no_activity,
                                    activity_level,
                                   row['TMT PRED AVG filtered']])    
                break
    except:
        continue
    
df_phys = pd.DataFrame(list_ph_act,columns=['id','Age','gender','Vigorous',
                                            'Moderate','Light','No activity',"Activity level",
                                            'TMT PRED AVG filtered'])
df_phys.to_csv(path_or_buf= "data/ABCD-studies/internat_physical_activ01.csv")


try:
    df_phys['Activity Levels2'] = df_phys['Light'] +3* df_phys['Vigorous']+2* df_phys['Moderate'] 

    first_q = df_phys['Activity Levels2'].quantile([.25, .5, .75])[0.25]
    mean_q = df_phys['Activity Levels2'].quantile([.25, .5, .75])[0.5]
    third_q = df_phys['Activity Levels2'].quantile([.25, .5, .75])[0.75]

    act_list2 =[]
    for idx in range(0, df_phys.shape[0]):
        row = df_phys.iloc[idx]['Activity Levels2']
        if first_q > row:
            act_list2.append('<25%%')
        elif mean_q > row > first_q:
            act_list2.append('25%%-50%%')
        elif third_q > row > mean_q:
            act_list2.append('50%%-75%%')
        elif row > third_q:
            act_list2.append('75%%>')
        else:
            act_list2.append("No data")
            
    df_phys['Activity Levels Quantiles'] = act_list2 
except:
    pass

## BMI vs TMT: abcd_ant01
input_annotation_file_xls = 'data/combined_abcd.xlsx'
df_xls =pd.read_excel(open(input_annotation_file_xls, 'rb'),
              sheet_name='abcd_ant01') 
df_xls = df_xls.drop([0])
list_ant = []

# tried to do a full match as in https://www.geeksforgeeks.org/join-pandas-dataframes-matching-by-substring/
# but it takes much longer
for idx in range(0, df_xls.shape[0]):
    row_xls = df_xls.iloc[idx]
    for jdx in range(0, df.shape[0]):
        row = df.iloc[jdx]
        try:
            key_id = row_xls['subjectkey'].split("_")[0] + row_xls['subjectkey'].split("_")[1]
            if key_id in row['ID'] and  row_xls['eventname'].replace("_","").lower() in row['ID'].lower():
                if not pd.isna(row_xls['anthro_1_height_in']) and not pd.isna(row_xls['anthroweight1lb']):
                    list_ant.append([key_id, row_xls['eventname'].replace("_","").lower(),
                                    row['Age'], 
                                    row['Gender'], 
                                    float(row_xls['anthro_1_height_in']), float(row_xls['anthroweight1lb']),
                                        703 * row_xls['anthroweight1lb'] / row_xls['anthro_1_height_in']**2,
                                    row['TMT PRED AVG filtered']])    
                    break
        except:
            continue
                
df_ant = pd.DataFrame(list_ant,columns=['id',"Visit",'Age','gender','Height',"Weight","BMI",'TMT PRED AVG filtered'])
df_ant.to_csv(path_or_buf= "data/ABCD-studies/abcd_bmi.csv")
df_ant = pd.read_csv("data/ABCD-studies/abcd_bmi.csv", header=0)

## check the parents estimate 
input_annotation_file_xls = 'data/combined_abcd.xlsx'
df_xls_disorders =pd.read_excel(open(input_annotation_file_xls, 'rb'),
              sheet_name='eating_disorders_p01') 
df_xls_disorders = df_xls_disorders.drop([0])
# ksads_eatingdis_raw_359_p

list_par_estimate = []
for idx in range(0, df_ant.shape[0]):
    row = df_ant.iloc[idx]
    for jdx in range(0, df_xls_disorders.shape[0]):
        row_xls = df_xls_disorders.iloc[jdx]
        try:
            key_id = row_xls['subjectkey'].split("_")[0] + row_xls['subjectkey'].split("_")[1]
            
            if str(key_id) in str(row['id']) and str(row_xls['eventname'].replace("_","").lower()) in str(row['Age']) \
            and row_xls['ksads_eatingdis_raw_359_p']!="":
                height_ft  = int(row_xls['ksads_eatingdis_raw_359_p'].split("feet:")[1].split("inches:")[0]) * 12 + \
                int(row_xls['ksads_eatingdis_raw_359_p'].split("inches:")[1].split("weight:")[0])
                
                weight = int(row_xls['ksads_eatingdis_raw_359_p'].split("weight:")[1])
                
                list_par_estimate.append([row['id'],row['Age'],height_ft, weight,
                                        row['Visit'],row['gender'],row['Height'],
                                        row["Weight"],row["BMI"],row['TMT PRED AVG filtered']])
        except:
            continue
                
df_ant_w_estimate = pd.DataFrame(list_par_estimate,columns=['id',"Visit","Parent_est_ft","Parent_est_weight",
                                                            'Age','gender','Height',"Weight",
                                                            "BMI",'TMT PRED AVG filtered'])

df_ant_w_estimate.to_csv(path_or_buf= "data/ABCD-studies/abcd_bmi_w_estimate.csv")

first_q = df_ant['BMI'].quantile([.25, .5, .75])[0.25]
mean_q = df_ant['BMI'].quantile([.25, .5, .75])[0.5]
third_q = df_ant['BMI'].quantile([.25, .5, .75])[0.75]

act_list2 = []
for idx in range(0, df_ant.shape[0]):
    row = df_ant.iloc[idx]['BMI']
    if first_q > row:
        act_list2.append('<25%%, '+str(round(first_q,2)))
    elif mean_q > row > first_q:
        act_list2.append('25%%-50%%, '+ str(round(first_q,2)) + "-"+ str(round(mean_q,2)))
    elif third_q > row > mean_q:
        act_list2.append('50%%-75%%, '+ str(round(mean_q,2)) + "-"+ str(round(third_q,2)))
    elif row > third_q:
        act_list2.append('75%%>, '+ str(round(third_q,2)))
    else:
        act_list2.append("No data")
        
df_ant['BMI Quantiles'] = act_list2
first_q = df_ant['Height'].quantile([.25, .5, .75])[0.25]
mean_q = df_ant['Height'].quantile([.25, .5, .75])[0.5]
third_q = df_ant['Height'].quantile([.25, .5, .75])[0.75]

act_list2 = []
for idx in range(0, df_ant.shape[0]):
    row = df_ant.iloc[idx]['Height']
    if first_q > row:
        act_list2.append('<25%%, '+str(round(first_q,2)))
    elif mean_q > row > first_q:
        act_list2.append('25%%-50%%, '+ str(round(first_q,2)) + "-"+ str(round(mean_q,2)))
    elif third_q > row > mean_q:
        act_list2.append('50%%-75%%, '+ str(round(mean_q,2)) + "-"+ str(round(third_q,2)))
    elif row > third_q:
        act_list2.append('75%%>, '+ str(round(third_q,2)))
    else:
        act_list2.append("No data")
        
df_ant['Height Quantiles'] = act_list2
df_ant.to_csv(path_or_buf= "data/ABCD-studies/abcd_bmi_2.csv")

## 8. Steps activity levels abcd_fbwpas01
input_annotation_file_xls = 'data/combined_abcd.xlsx'
df_xls =pd.read_excel(open(input_annotation_file_xls, 'rb'),
              sheet_name='abcd_fbwpas01') 
df_xls = df_xls.drop([0])

list_steps = []
# tried to do a full match as in https://www.geeksforgeeks.org/join-pandas-dataframes-matching-by-substring/
# but it takes much longer
for idx in range(0, df_xls.shape[0]):
    row_xls = df_xls.iloc[idx]
    try:
        key_id = row_xls['subjectkey'].split("_")[0] + row_xls['subjectkey'].split("_")[1]
        for jdx in range(0, df.shape[0]):
            row = df.iloc[jdx]
            #print( row['ID'])
            if key_id in row['ID'] and  row_xls['eventname'].replace("_","").lower() in row['ID'].lower():
                if not pd.isna(row_xls['fit_ss_perday_totalsteps']):
                    #print(idx, row_xls['subjectkey'], row_xls['eventname'])
                    list_steps.append([key_id, 
                                row['Age'], 
                                row['Gender'], 
                                float(row_xls['fit_ss_wk_total_steps']),
                                row['TMT PRED AVG filtered']])    
                break
    except:
        continue

df_step = pd.DataFrame(list_steps,columns=['id','Age','Gender',"Total steps",'TMT PRED AVG filtered'])
df_step.to_csv(path_or_buf= "data/ABCD-studies/abcd_steps.csv")

first_q = round(df_step['Total steps'].quantile([.25, .5, .75])[0.25],2)
mean_q = round(df_step['Total steps'].quantile([.25, .5, .75])[0.5],2)
third_q = round(df_step['Total steps'].quantile([.25, .5, .75])[0.75],2)

act_list2 =[]
for idx in range(0, df_step.shape[0]):
    row = df_step.iloc[idx]['Total steps']
    if first_q > row:
        act_list2.append('<25%%, '+str(round(first_q,2)))
    elif mean_q > row > first_q:
        act_list2.append('25%%-50%%, '+ str(round(first_q,2)) + "-"+ str(round(mean_q,2)))
    elif third_q > row > mean_q:
        act_list2.append('50%%-75%%, '+ str(round(mean_q,2)) + "-"+ str(round(third_q,2)))
    elif row > third_q:
        act_list2.append('75%%>, '+ str(round(third_q,2)))
    else:
        act_list2.append("No data")
        
df_step['Total steps Quantiles'] = act_list2

## 9. Calorical Intake (abcd_bkfs01)
input_annotation_file_xls = 'data/combined_abcd.xlsx'
df_xls =pd.read_excel(open(input_annotation_file_xls, 'rb'),
              sheet_name='abcd_bkfs01') 
df_xls = df_xls.drop([0])
df_2 = df.loc[df['ID'].str.contains("2year|3year", case=False)]

list_cals = []
for idx in range(0, df_xls.shape[0]):
    row_xls = df_xls.iloc[idx]
    for jdx in range(0, df_2.shape[0]):
        row = df_2.iloc[jdx]
        try:
            key_id = row_xls['subjectkey'].split("_")[0] + row_xls['subjectkey'].split("_")[1]
            if key_id in row['ID'] and row_xls['eventname'].replace("_","").lower() in row['ID'].lower():
                if not pd.isna(row_xls['bkfs_dt_kcal']):
                    list_cals.append([key_id, 
                                row['Age'], 
                                row['Gender'], 
                                row_xls['bkfs_gi'], row_xls['bkfs_gl'],row_xls['bkfs_dt_kcal'],
                                row['TMT PRED AVG filtered']])    
                break
        except:
            continue
            
df_cals = pd.DataFrame(list_cals,columns=['id','Age','Gender',"GI","GL","Dt_kcal",'TMT PRED AVG filtered'])
df_cals.to_csv(path_or_buf= "data/ABCD-studies/abcd_cals.csv")

mean_cal=str(round(df_cals['Dt_kcal'].mean(),2))
df_cals["Above average Dt_kcal level, mean="+mean_cal+" kcal"] = df_cals['Dt_kcal']>=df_cals['Dt_kcal'].mean()

input_annotation_file_xls = 'data/combined_abcd.xlsx'
df_xls =pd.read_excel(open(input_annotation_file_xls, 'rb'),
              sheet_name='abcd_ybd01') 
df_xls = df_xls.drop([0])
#blood was drawn only at 2yr visit
df_2 = df.loc[df['ID'].str.contains("2year", case=False)]

list_blood = []
for idx in range(0, df_xls.shape[0]):
    row_xls = df_xls.iloc[idx]
    try:
        key_id = row_xls['subjectkey'].split("_")[0] + row_xls['subjectkey'].split("_")[1]
        for jdx in range(0, df_2.shape[0]):
            row = df_2.iloc[jdx]
            if key_id in row['ID']:
                if not pd.isna(row_xls['biospec_blood_hemoglobin']):
                    list_blood.append([key_id, 
                                row['Age'], 
                                row['Gender'], 
                                row_xls['biospec_blood_hemoglobin'],
                                row_xls['biospec_blood_cholesterol'],
                                row_xls['biospec_blood_hdl_cholesterol'],
                                row['TMT PRED AVG filtered']])    
                break
    except:
        continue
        
df_blood = pd.DataFrame(list_blood,columns=['id','Age','Gender',
                                            "Hemoglobin","Cholesterol","HDL Cholesterol",
                                            'TMT PRED AVG filtered'])
df_blood.to_csv(path_or_buf= "data/ABCD-studies/abcd_blood.csv")

df_blood["Above average Cholesterol level, mean="+str(round(df_blood['Cholesterol'].mean(),2))] = df_blood['Cholesterol']>=df_blood['Cholesterol'].mean()
df_blood["Above average Hemoglobin level, mean="+str(round(df_blood['Hemoglobin'].mean(),2))] = df_blood['Hemoglobin']>=df_blood['Hemoglobin'].mean()

## ETNICITY
input_annotation_file_xls = 'data/combined_abcd.xlsx'
df_xls =pd.read_excel(open(input_annotation_file_xls, 'rb'),
              sheet_name='multigrp_ethnic_id_meim01') 
df_xls = df_xls.drop([0])

# 1= White/Caucasian; 2= Western European; 3= Eastern European; 
#4= Hispanic/Latino; 5= Black/African American;
#6= Afro-Carribean/Indo-Carribbean/West Indian (i.e. Jamaica= Haiti= Trinidad= Guyana);
#7= East Asian (i.e. China= Japan= Korea= Taiwan); 8= South Asian (i.e. India= Pakistan= Bangladesh;
#9= Southeast Asian (i.e. Phillipines= Vietnam= Thailand); 10= American Indian/Alaska Native#;
#11= Middle Eastern/North African; 12= Native Hawaiian or Pacific Islander; 
#13= Mixed Ethnicity; 14= Other Ethnicity; 0= None; 999= Don't know; 777= Refuse to answer

# flip iteration
list_etnic = []
for idx in range(0, df.shape[0]):
    row = df.iloc[idx]
    for jdx in range(0, df_xls.shape[0]):
        row_xls = df_xls.iloc[jdx]
        try:
            key_id = row_xls['subjectkey'].split("_")[0] + row_xls['subjectkey'].split("_")[1]
            combined_race = ""
            if key_id in row['ID']:
                #print(key_id,row['ID'])
                if row_xls['meim_ethnic_id'] ==1 or row_xls['meim_ethnic_id'] ==2 \
                    or row_xls['meim_ethnic_id'] ==3:
                    combined_race='White/European'
                elif row_xls['meim_ethnic_id'] == 4:
                    combined_race='Latino'
                elif row_xls['meim_ethnic_id'] == 5:
                    combined_race='Black' 
                elif row_xls['meim_ethnic_id'] == 7 or row_xls['meim_ethnic_id'] == 8 or row_xls['meim_ethnic_id'] == 9:
                    combined_race='Asian' 
                elif row_xls['meim_ethnic_id'] == 13:
                    combined_race='Mixed' 
                elif row_xls['meim_ethnic_id'] == 14 or row_xls['meim_ethnic_id'] == 12 or row_xls['meim_ethnic_id'] ==10\
                    or row_xls['meim_ethnic_id'] == 11 or row_xls['meim_ethnic_id'] == 6:
                    combined_race='Other' 
                        
                if combined_race != "":  
                    list_etnic.append([key_id, 
                                    row['Age'], 
                                    row['Gender'], 
                                    row_xls['meim_ethnic_id'],
                                    combined_race,
                                    row['TMT PRED AVG filtered']])    
                #break
        except:
            continue
    
df_etnic = pd.DataFrame(list_etnic,columns=['id',
                                            'Age',
                                            'Gender',
                                            "Etnic id",
                                            'Race',
                                            'TMT PRED AVG filtered'])
df_etnic.to_csv(path_or_buf= "data/ABCD-studies/abcd_enticity.csv")
