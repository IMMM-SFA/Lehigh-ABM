# File description @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ABM code for the coupling with RiverWare
# Writen by Shih-Yu Huang
# The ABM is built based on Bayesian Inference combined with Cost-loss model
# This script needs to be compiled into an excecutable file and called by RiverWare
# To compile to .exe, use commend
# pyinstaller -F filename.py

# How to run the model @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Reset the values in interaction_info.txt to [0; 0] when starting the simulation

# Model Parameters @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Model parameters of each agent are stored in ABM_params.csv with
# L1-L3: lambda, risk perception parameters
# z_val: Cost-loss ratio
# area_inc: optimal area increment (calibrated parameters)
# inci_loss_rate: incidental loss rate (given by RiverWare)
# eff: water use efficiency (given by RiverWare)
# max_cap: max diversion capacity (given by RiverWare)

# global model parameters are stored in model_params.csv with
# time_window: time window for calculating extremity, default 30 (yrs)
# min_SJRB: min outflow requirement of SJRB (from RiverWare)
# Div_SS: fixed diversion during shortage sharing (from RiverWare)
# Navajo_elec_SS: Navajo elevation threshold during shortage sharing (from RiverWare)
# conv_div: unit conversion of diversion (from RiverWare)
# area_unc: uncertainty associated with irrigation area expansion, default 30%

# Model Outputs @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 1: Daily irrigation area, diversion, and ET are stored in Irr_all.txt, Div_all.txt, 
#    and ET_all.txt respectively
# 2: Annual model outputs are stored in Anuual_record.txt
#    col 0: starting year
#    col 1: ending year
#    col 2-4: precipitations (upstream, Animas river, downstream)
#    col 5: NIIP diversion
#    col 6: downstream # of flow violation days
#    col 7: Navajo Reservoir elevation on 12/31
#    col 8: shortage sharing index
#    col 9-24: index of expanding/reducing irrigation area for each agent (1: expand, 0: reduce)
#    col 25-40: annual irrigation area

# import necessary modules @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import os
import numpy as np
import pandas as pd
import datetime
import xlsxwriter

from Agent_BI_CL_CAWS import G1_func, G2_func, G3_func # import ABM subroutine for different groups of agents

# defined functions @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# subroutine of calsulating # of days in a year (leap & normal)
def days_in_year(yr):
    if yr % 4 == 0:
        days_in_yr = 366
    else:
        days_in_yr = 365
    return days_in_yr

# general file name & path @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Directories of ABM and RiverWare inputs/outputs
R_to_A_path = os.path.join('RiverWare_to_ABM')   # path: RiverWare to ABM
A_to_R_path = os.path.join('ABM_to_RiverWare')   # path: ABM to RiverWare
Ext_Inputs_path = os.path.join('External_Inputs')# path: external data inputs (Precipitation, NIIP diversion)

# file name & path: winter precipitation & NIIP diversion data
Prep_winter_fname = 'Prep_winter_SJRB.txt'       # file name: winter precipitation
NIIP_div_fname = 'NIIP_SJRB.txt'                 # file name: NIIP diversion
Prep_winter_path = os.path.join(Ext_Inputs_path, Prep_winter_fname)     # path: winter precipitation file
NIIP_div_path = os.path.join(Ext_Inputs_path, NIIP_div_fname)           # path: NIIP diversion file

# file name & path: store all irrigation area, diversion, and ET data
R_to_A_Irr_all_fname = 'Irr_all.txt'              # file name: irrigation area (all agents)
R_to_A_Div_all_fname = 'Div_all.txt'              # file name: diversion (all agents)
R_to_A_ET_all_fname = 'ET_all.txt'                # file name: ET (all agents)
R_to_A_Annual_record_fname = 'Annual_record.txt'  # file name: annual records
R_to_A_Irr_all_path = os.path.join(R_to_A_path, R_to_A_Irr_all_fname)             # path: irrigation area file (all agents)
R_to_A_Div_all_path = os.path.join(R_to_A_path, R_to_A_Div_all_fname)             # path: diversion (all agents)
R_to_A_ET_all_path = os.path.join(R_to_A_path, R_to_A_ET_all_fname)               # path: ET file (all agents)
R_to_A_Annual_record_path = os.path.join(R_to_A_path, R_to_A_Annual_record_fname) # path: annual records (all agents)

# file name & path: outlet flowrate of SJRB & Navajo Reservoir elevation (generated from RiverWare)
ISFbluff_fname = 'ISF_Bluff_Outflow.txt'                        # file name: outlet flowrate of SJRB
Navajo_elev_fname = 'Navajo_Elevation.txt'                      # file name: Navajo Reservoir elevation
ISFbluff_path = os.path.join(R_to_A_path, ISFbluff_fname)       # path: outlet flowrate of SJRB
Navajo_elev_path = os.path.join(R_to_A_path, Navajo_elev_fname) # path: Navajo Reservoir elevation

# file name & load: global and ABM model parameters
model_params_fname = 'model_params.csv'                         # file name: model parameters
model_params = pd.read_csv(model_params_fname)                  # load model parameters
ABM_params_fname = 'ABM_params.csv'                             # file name: ABM paramaters
ABM_params = pd.read_csv(ABM_params_fname, index_col = 0)       # load ABM parameters

# load SJRB outflow and Navajo elevation data @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open(ISFbluff_path) as f_ISFbluff:
    SJRB_outflow_headers = f_ISFbluff.readlines()[0:6]   # load headers of outflow file
    f_ISFbluff.seek(0)                                   # reset searching point
    SJRB_outflow = f_ISFbluff.readlines()[6:]            # load outflow data

with open(Navajo_elev_path) as f_Navajo_elev:
    Navajo_elev_headers = f_Navajo_elev.readlines()[0:6] # load headers of Navajo elevation file
    f_Navajo_elev.seek(0)                                # reset searching point
    Navajo_elev = f_Navajo_elev.readlines()[6:]          # load elevation data

# data cleaning (remove \n in each line)
SJRB_outflow = np.array(list(map(lambda s: s.strip(), SJRB_outflow)),dtype = float) # remove \n in the data, change fomrat to np array
Navajo_elev = np.array(list(map(lambda s: s.strip(), Navajo_elev)),dtype = float)   # remove \n in the datam change format to np array

data_len = len(SJRB_outflow)                            # data length

# setup model time @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
start_date = datetime.datetime(1928, 9, 30)             # 1st day of model period
end_date = datetime.datetime(2013, 9, 30)               # last day of model period
total_days = (end_date - start_date).days + 1           # total # of days
date_seq = [start_date + datetime.timedelta(days = x) \
            for x in range(0, total_days)]              # sequence of date (entire model period)
yr_list = np.arange(1928,2014,1)                        # list of year

# Agent names @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
agt_names = ['JicarillaIrr','NMPineRiverAreaIrr','ArchuletaDitch','CitizenDitch',   
    'TurleyDitch','Hammond','TwinRocks','Ralston','NMAnimasIrr','FarmingtonGlade',
    'EchoDitch','FarmersMutual','FruitlandAndCambridge',
    'JewettValley','Hogback','CudeiCanal']
agt_num = len(agt_names)                                    # number of agents

# global model parameters @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
min_SJRB_outflow = model_params['min_SJRB_outflow'].values  # min outflow requirement of SJRB
Div_SS = model_params['Div_SS'].values                      # fixed diversion during shortage sharing
Navajo_elev_SS = model_params['Navajo_elev_SS'].values      # Navajo elevation threshold during shortage sharing
conv_div = model_params['conv_div'].values                  # unit conversion of diversion
twd = model_params['time_window'].values                    # time window for computing extremities
area_unc = model_params['area_unc'].values                  # uncertainty associated with irrigation area expansion

# ABM calculation @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# check if first year interaction
R_A_int_1st_fname = 'interaction_info.txt'                  # file name: day and year number for RiverWare-River interaction
check_1st_yr = np.loadtxt(R_A_int_1st_fname)                # load values from interaction_info.txt to check if 1st year

if check_1st_yr[0] == 0:                                    # if 1st year (1928/09/30)
    interaction_info = check_1st_yr                         # reset interaction information
    yr_idx = 0                                              # year number (0 = 1928, 1 = 1929,...), number of year = yr_idx+1 due to zero-based index
    d_start = 1                                             # starting day (start from 2nd day as the first day is nan)
    d_in_yr = days_in_year(yr_list[yr_idx + 1])             # number of days in the coming year (1929 here) to determine the year of day in the water year
    d_num = np.arange(d_start, d_start + d_in_yr)           # day number sequence    
    d_y_seq = date_seq[d_num[0]:d_num[-1] + 1]              # date_year sequence of the current year
    eoy_idx = d_y_seq.index(datetime.datetime(year=int(yr_list[yr_idx]), 
                                              month=12, day=31)) # find location of the end of calendar year (to obtain Navajo elevation for calculating extremity)
    
    # load winter precipitation & NIIP diversion data
    Prep_winter = np.loadtxt(Prep_winter_path,delimiter=",")   # load winter precipitation data
    NIIP_div = np.loadtxt(NIIP_div_path)                       # load NIIP diversion data
    
    # create annual data martrix for storing results
    Annual_data_mat = np.zeros([85,57])                 # create annual data matrix
    Annual_data_mat[:,0:5] = Prep_winter[:]             # 0:4: start year, end year, upstream annual precip, Animas precip, downstream precip (mm)
    Annual_data_mat[:,5] = NIIP_div                     # 5: NIIP diversion
    Annual_data_mat[yr_idx,6] = np.array(np.where(SJRB_outflow[d_num] < min_SJRB_outflow)).shape[1] # 6: days of flow violation during 1928/10/01 - 1929/09/30
    Annual_data_mat[yr_idx,7] = Navajo_elev[d_num[eoy_idx]]  # 7: Navajo Reservoir elevation at the end of calendar year in ft
    Annual_data_mat[yr_idx,9:] = 1                      # 9:24: assuming all 16 agents increased irrigated area at the 1st year
                                                        # 25:40: annual irrigation area (will be filled after every year's calculation)
                                                        # 41:57: P values
                                                        # 8: shortage sharing index (1: Yes, 0, No)
    # loop over all agents
    for i in range(1,agt_num + 1,1):
        
        # file name & path: irrigation area, diversion, and ET data from the RiverWare_to_ABM folder (from Cycle 3)
        Irr_fname = str(i).zfill(2) + '_' + agt_names[i-1] + '_IrrArea.txt' # file name: irrigation area of agent i
        Div_fname = str(i).zfill(2) + '_' + agt_names[i-1] + '_Div.txt'     # file name: diversion of agent i
        ET_fname = str(i).zfill(2) + '_' + agt_names[i-1] + '_ET.txt'       # file name: ET of agent i        
        Irr_path = os.path.join(R_to_A_path, Irr_fname) # path: irritation area of agent i, RiverWare_to_ABM folder
        Div_path = os.path.join(R_to_A_path, Div_fname) # path: diversion of agent i, RiverWare_to_ABM folder
        ET_path = os.path.join(R_to_A_path, ET_fname)   # path: ET of agent i, RiverWare_to_ABM folder
        
        # load irrigation area, diversion, ET data and file headers
        with open(Irr_path) as f_Irr:
            # Irr_headers = f_Irr.readlines()[0:6]      # ori format: irrigation area file headers
            Irr_headers = pd.DataFrame(f_Irr.readlines()[0:6], columns = [str(i).zfill(2) + '_' + agt_names[i - 1]]) # read headers (irrigation area)
            f_Irr.seek(0)
            Irr = f_Irr.readlines()[6:]                 # retrieve irrigation area data
        
        with open(Div_path) as f_Div:
            # Div_headers = f_Div.readlines()[0:6]      # ori format: diversion file headers
            Div_headers = pd.DataFrame(f_Div.readlines()[0:6], columns = [str(i).zfill(2) + '_' + agt_names[i - 1]]) # read headers (diversion)
            f_Div.seek(0)
            Div = f_Div.readlines()[6:]                 # retrieve diversion data        
            
        with open(ET_path) as f_ET:
            # ET_headers = f_ET.readlines()[0:6]        # ori format: ET file headers
            ET_headers = pd.DataFrame(f_ET.readlines()[0:6], columns = [str(i).zfill(2) + '_' + agt_names[i - 1]])  # read headers (ET)
            f_ET.seek(0)
            ET = f_ET.readlines()[6:]                   # retrieve ET data    
        
        # data cleaning (remove \n in each line)
        Irr_headers = Irr_headers.replace('\n','', regex=True)
        Div_headers = Div_headers.replace('\n','', regex=True)
        ET_headers = ET_headers.replace('\n','', regex=True)        
        Irr = np.array(list(map(lambda s: s.strip(), Irr)),dtype = float)           # change Irr format to np array
        Div = np.array(list(map(lambda s: s.strip(), Div)),dtype = float)           # change Div format to np array
        ET = np.array(list(map(lambda s: s.strip(), ET)),dtype = float)             # change ET format to np array
        
        # create matrices for headers
        if i == 1:
            Irr_headers_all = Irr_headers   # irrigation area headers
            Div_headers_all = Div_headers   # diversion headers
            ET_headers_all = ET_headers     # ET headers
        else:
            Irr_headers_all = pd.concat([Irr_headers_all, Irr_headers], axis = 1) # concatenate Irr headers
            Div_headers_all = pd.concat([Div_headers_all, Div_headers], axis = 1) # concatenate Div headers
            ET_headers_all = pd.concat([ET_headers_all, ET_headers], axis = 1)    # concatenate ET headers
        
        # create big matrix for storing irrigation area, diversion, and ET (daily values)
        if i == 1:
            Irr_all = np.zeros((Irr.shape[0], agt_num)) # irrigation area matrix (all agents)
            Div_all = np.zeros((Div.shape[0], agt_num)) # diversion area matrix (all agents)
            ET_all = np.zeros((ET.shape[0], agt_num))   # ET area matrix (all agents)
        
        # assign values to the matrix for each agent (daily and annually)
        Irr_all[:,i - 1] = Irr
        Div_all[:,i - 1] = Div
        ET_all[:,i - 1] = ET
        
        Annual_data_mat[yr_idx, 24 + i] = Irr_all[1, i - 1]    # assign the value of irrigation area at 10/01 to the annual matrix
        
        # save irrigation area, diversion, ET data to ABM_to_RiverWare folder
        A_to_R_Irr_path = os.path.join(A_to_R_path, Irr_fname) # path: irrigation area of agent i, ABM_to_RiverWare folder
        A_to_R_Div_path = os.path.join(A_to_R_path, Div_fname) # path: diversion of agent i, ABM_to_RiverWare folder
        A_to_R_ET_path = os.path.join(A_to_R_path, ET_fname)   # path: ET of agent i, ABM_to_RiverWare folder
        
        with open(A_to_R_Irr_path,'w') as f_Irr:               # write irrigation area data to agt_names.txt
            # f_Div.writelines(Irr_headers)                    # ori format for writing headers (with \n) 
            f_Irr.write('\n'.join(list(Irr_headers[str(i).zfill(2) + '_' + str(agt_names[i - 1])])))
            # f_Irr.write('\n'.join(map(str, Irr)))
            f_Irr.write('\n')
            f_Irr.write('\n'.join(map(lambda x: "{:.2f}".format(x), Irr)))
        
        with open(A_to_R_Div_path,'w') as f_Div:               # write diversion data to agt_names.txt
            # f_Div.writelines(Div_headers)                    # ori format for writing headers (with \n) 
            f_Div.write('\n'.join(list(Div_headers[str(i).zfill(2) + '_' + str(agt_names[i - 1])])))
            # f_Div.write('\n'.join(map(str, Div)))
            f_Div.write('\n')
            f_Div.write('\n'.join(map(lambda x: "{:.2f}".format(x), Div)))
            
        # may not be necessarily to output ET data *******    
        with open(A_to_R_ET_path,'w') as f_ET:                 # write diversion data to agt_names.txt
            # f_ET.writelines(ET_headers)                      # ori format for writing headers (with \n) 
            f_ET.write('\n'.join(list(ET_headers[str(i).zfill(2) + '_' + str(agt_names[i - 1])])))
            # f_ET.write('\n'.join(map(str, ET)))
            f_ET.write('\n')
            f_ET.write('\n'.join(map(lambda x: "{:.5f}".format(x), ET)))
    
    # save all irrigation area, diversion, and ET data to a big file for saving simulation time (folder: RiverWare_to_ABM)        
    np.savetxt(R_to_A_Irr_all_path, Irr_all, fmt = '%.2f', delimiter = ',') # save all irrigation area data
    np.savetxt(R_to_A_Div_all_path, Div_all, fmt = '%.2f', delimiter =',')  # save all diversion data
    np.savetxt(R_to_A_ET_all_path,ET_all, fmt = '%.5f', delimiter =',')     # save all ET data
    np.savetxt(R_to_A_Annual_record_path, Annual_data_mat, fmt = '%.5f', delimiter =',') # save annual records of all variables
    
    # save all headers and export to excel files
    xlsx_writer = pd.ExcelWriter('headers.xlsx', engine='xlsxwriter')
    Irr_headers_all.to_excel(xlsx_writer, sheet_name='Irr_headers')
    Div_headers_all.to_excel(xlsx_writer, sheet_name='Div_headers')
    ET_headers_all.to_excel(xlsx_writer, sheet_name='ET_headers')
    xlsx_writer.save()    
    
    # update interaction information and export
    interaction_info = [yr_idx + 1, d_num[-1] + 1]          # [yr_idx (0=1928,1=1929,...), next year starting date] 
    np.savetxt('interaction_info.txt', interaction_info, fmt = '%i') 
    
else: 
    # data processing: load data and information from previous time step @@@@@@       
    
    # load data of all variables from the previous year
    interaction_info = np.loadtxt('interaction_info.txt',dtype = 'int')     # load current yr and time information
    Irr_all = np.loadtxt(R_to_A_Irr_all_path, dtype='float', delimiter=',') # irrigation area data
    Div_all = np.loadtxt(R_to_A_Div_all_path, delimiter=',')                # diversion data
    ET_all = np.loadtxt(R_to_A_ET_all_path, delimiter=',')                  # ET data
    Annual_data_mat = np.loadtxt(R_to_A_Annual_record_path, delimiter=',')  # annual records

    # extract model time information 
    yr_idx = interaction_info[0]        # current year of interaction (1: 1929, 2:1930,...)
    d_start = interaction_info[1]       # starting date of interaction 
    d_in_yr = days_in_year(yr_list[yr_idx + 1])         # number of days in the coming year for determine the window of current year
    d_num = np.arange(d_start, d_start + d_in_yr)       # day number sequence
    d_y_seq = date_seq[d_num[0]:d_num[-1] + 1]          # date_year sequence of current year
    eoy_idx = d_y_seq.index(datetime.datetime(year=int(yr_list[yr_idx]), month=12, day=31)) # find location of the end of calendar year (to obtain Navajo elevation for calculating extremity)
    
    # load file headers for irrigation area, diversion, and ET for writing RiverWare files
    Irr_headers_all = pd.read_excel("headers.xlsx", sheet_name = 'Irr_headers')
    Div_headers_all = pd.read_excel("headers.xlsx", sheet_name = 'Div_headers')
    ET_headers_all = pd.read_excel("headers.xlsx", sheet_name = 'ET_headers')
    
    # quantities for calculating extremities 
    Annual_data_mat[yr_idx, 6] = np.array(np.where(SJRB_outflow[d_num] < min_SJRB_outflow)).shape[1] # days of flow violation during previous year (if 1st time: 1928/10/01 - 1929/09/30)
    Annual_data_mat[yr_idx, 7] = Navajo_elev[d_num[eoy_idx]]        # Navajo Reservoir elevation at the end of calendar year (12/31) in ft
    
    # Bayesian Inference Mapping for risk perception analysis 
    NIIP_div_thr = np.mean(Annual_data_mat[:yr_idx + 1, 5])         # threshold of NIIP diversion: mean up to current year
    flow_vio_thr = np.mean(Annual_data_mat[:yr_idx + 1, 6])         # threshold of flow violation: mean up to current year
    Navajo_thr = np.mean(Annual_data_mat[:yr_idx + 1, 7])           # threshold of Navajo elevation: mean up to current year
    prep_winter_thr_G1 = np.mean(Annual_data_mat[:yr_idx + 2, 2])   # precipitation threshold: group 1: mean up to the coming year
    prep_winter_thr_G2 = np.mean(Annual_data_mat[:yr_idx + 2, 3])   # precipitation threshold: group 2: mean up to the coming year
    prep_winter_thr_G3 = np.mean(Annual_data_mat[:yr_idx + 2, 4])   # precipitation threshold: group 3: mean up to the coming year
    
    # load ABM parameters for each agent
    z_val = ABM_params.loc['z_val'].values                      # z values
    opt_area_inc = ABM_params.loc['area_inc'].values            # optimal area increment
    inci_loss_rate = ABM_params.loc['inci_loss_rate'].values    # incidental loss rate
    eff = ABM_params.loc['eff'].values                          # water use efficiency 
    max_cap = ABM_params.loc['max_cap'].values                  # maximum diversion capacity 
    
    # compute probabilities of exceedances (with time window controled by twd)
    if yr_idx + 1 <= twd:                                       # if # of years (yr_idx + 1) <= 30
        ynum_p = np.arange(0, yr_idx + 1, 1)
        if yr_idx + 1 == twd:                                   # if year 30 => forecast has 31 year data to use, need to shift
            ynum_f = np.arange(1,yr_idx + 2, 1)                 # to use data from year 2:31
        else:
            ynum_f = np.arange(0,yr_idx + 2, 1)
    else:
        ynum_p = np.arange(yr_idx - twd + 1, yr_idx  + 1, 1)
        ynum_f = np.arange(yr_idx - twd + 2, yr_idx + 2, 1)

    prob_NIIP_div = np.asarray((np.where(Annual_data_mat[ynum_p, 5] \
                                         < NIIP_div_thr))).shape[1]/len(ynum_p)  # exceedance: P(NIIP_div < Thr)       
    prob_flow_vio = np.asarray((np.where(Annual_data_mat[ynum_p, 6] \
                                         < flow_vio_thr))).shape[1]/len(ynum_p)  # exceedance: P(flow_vio < Thr)
    prob_Navajo_elev = np.asarray((np.where(Annual_data_mat[ynum_p, 7] \
                                            > Navajo_thr))).shape[1]/len(ynum_p) # exceedance: P(Navajo Elev > Thr)
    prob_prep_winter_G1 = np.asarray((np.where(Annual_data_mat[ynum_f, 2] \
                                               > prep_winter_thr_G1))).shape[1]/len(ynum_f) # exceedance G1: P(prep_winter > Thr)
    prob_prep_winter_G2 = np.asarray((np.where(Annual_data_mat[ynum_f, 3] \
                                               > prep_winter_thr_G2))).shape[1]/len(ynum_f) # exceedance G2: P(prep_winter > Thr) 
    prob_prep_winter_G3 = np.asarray((np.where(Annual_data_mat[ynum_f, 4] \
                                               > prep_winter_thr_G3))).shape[1]/len(ynum_f) # exceedance G3: P(prep_winter > Thr)

    # compute extremities for each preceding factor
    extr_prep_G1 = abs(Annual_data_mat[ynum_f[-1],2]/max(Annual_data_mat[ynum_f,2]) - 0.5)
    extr_prep_G2 = abs(Annual_data_mat[ynum_f[-1],3]/max(Annual_data_mat[ynum_f,3]) - 0.5)
    extr_prep_G3 = abs(Annual_data_mat[ynum_f[-1],4]/max(Annual_data_mat[ynum_f,4]) - 0.5)
    extr_NIIP_div = abs(Annual_data_mat[ynum_p[-1],5]/max(Annual_data_mat[ynum_p,5]) - 0.5)
    extr_flow_vio = abs(Annual_data_mat[ynum_p[-1],6]/max(Annual_data_mat[ynum_p,6]) - 0.5)
    extr_Navajo_elev = abs(Annual_data_mat[ynum_p[-1],7]/max(Annual_data_mat[ynum_p,7]) - 0.5)
        
    # loop over all agents
    for i in range(1,agt_num + 1,1):
        # load data from RiverWare simulation
        Irr_fname = str(i).zfill(2) + '_' + agt_names[i-1] + '_IrrArea.txt'    # file name: irrigation area of agent i
        Div_fname = str(i).zfill(2) + '_' + agt_names[i-1] + '_Div.txt'        # file name: diversion of agent i        
        ET_fname = str(i).zfill(2) + '_' + agt_names[i-1] + '_ET.txt'          # file name: evapotranspiration of agent i        
        Irr_path = os.path.join(R_to_A_path, Irr_fname)                        # path: irritation area of agent i, RiverWare_to_ABM folder
        Div_path = os.path.join(R_to_A_path, Div_fname)                        # path: diversion of agent i, RiverWare_to_ABM folder
        Irr_prev = Irr_all[d_start - 1, i - 1]                                 # previous year irrigation area
        prob_expa_Irr = np.asarray(np.where(Annual_data_mat[ynum_p, 8 + i]     # P(expanding irrigation area for agent i)
                            == 1)).shape[1]/(len(ynum_p) - 1)
        
        # generate irrigation area change amount with uncertainty
        np.random.seed()
        rnd = np.random.random()        # randomness of changing irrigation area
        np.random.seed()
        sign_val = np.random.randint(2) # randomness of plus minus sign from opt_area_inc
        
        if sign_val > 0:
            area_inc = opt_area_inc[i - 1] * (1 + rnd * area_unc) # determine the amounts of irrigation area change
        else:
            area_inc = opt_area_inc[i - 1] * (1 - rnd * area_unc)
            
        # decision-making process using BI mapping
        L_vec = ABM_params[str(i).zfill(2)+'_'+agt_names[i-1]][0:3].values # load lambda values (risk perception paramaters) for each agent

        if i == 1 or i == 2:                        # Group 1: agent 1, 2
            # BI mapping for Group 1 agents: return updated irrigation area & decision (1: expand, 0: reduce)
            P_agt, Irr_agt, dec_var = G1_func(prob_prep_winter_G1, prob_flow_vio, prob_Navajo_elev, \
                                       prob_expa_Irr, L_vec, extr_prep_G1, extr_flow_vio, \
                                       z_val[i - 1], Irr_prev, area_inc)   
            print('Done Calculation of Agent ' + str(i) + ', year ' + str(yr_list[yr_idx]))                   
        elif i >= 7 and i <= 12:                    # Group 2: agent 7, 8, 9, 10, 11, 12
            if i == 12 and Annual_data_mat[yr_idx - 1, 7] < Navajo_elev_SS:
                Irr_agt = float('nan')              # assign nan to the irrigation area if shortage sharing occur
                Annual_data_mat[yr_idx, 8] = 1      # index of shortage sharing occurrence
                dec_var = 0
            else:
                # BI mapping for Group 2 agents: return updated irrigation area & decision (1: expand, 0: reduce)
                P_agt, Irr_agt, dec_var = G2_func(prob_prep_winter_G1, prob_prep_winter_G2, prob_flow_vio, \
                                           prob_Navajo_elev, prob_expa_Irr, L_vec, extr_prep_G2, \
                                           extr_flow_vio, z_val[i - 1], Irr_prev, area_inc)
                print('Done Calculation of Agent ' + str(i) + ', year ' + str(yr_list[yr_idx]))                   
        else:                                       # Group 3: agent 3, 4, 5, 6, 13, 14, 15, 16
            if (i == 4 or i == 6 or i == 13 or i == 14 or i == 15) and Annual_data_mat[yr_idx - 1, 7] < Navajo_elev_SS:
                Irr_agt = float('nan')              # assign nan to the irrigation area if shortage sharing occur
                Annual_data_mat[yr_idx, 8] = 1      # index of shortage sharing occurrence
                dec_var = 0
            else:
                # BI mapping for Group 3 agents: return updated irrigation area & decision (1: expand, 0: reduce)
                P_agt, Irr_agt, dec_var = G3_func(prob_prep_winter_G1, prob_prep_winter_G3, prob_flow_vio, \
                                           prob_Navajo_elev, prob_NIIP_div, prob_expa_Irr, L_vec, \
                                           extr_prep_G3, extr_flow_vio, extr_NIIP_div, z_val[i - 1], Irr_prev, area_inc)
                print('Done Calculation of Agent ' + str(i) + ', year ' + str(yr_list[yr_idx]))                   
        Annual_data_mat[yr_idx, 8 + i] = dec_var    # save decision variables for each agent
        Annual_data_mat[yr_idx, 24 + i] = Irr_agt   # save updated irrigation area for each agent
        Annual_data_mat[yr_idx, 40 + i] = P_agt
        
        # update irrigation area and diversion for each agent
        Irr_all[d_num, i - 1] = Irr_agt                                 # update irrigation area
        Div_agt = Irr_all[d_num, i - 1] * ET_all[d_num, i - 1] * \
                (1 + inci_loss_rate[i - 1]) / eff[i - 1] * conv_div     # calculate diversion by unit conversion
        over_cnt = np.where(Div_agt > max_cap[i - 1])                   # check if diversion exceed max_cap     
        Div_agt[over_cnt] = max_cap[i - 1]                              # assign max_cap to diversion if the value exceeds max_cap                     
        Div_all[d_num, i - 1] = Div_agt                                 # update diversion
        
        # save irrigation area, diversion, ET data to ABM_to_RiverWare folder
        A_to_R_Irr_path = os.path.join(A_to_R_path, Irr_fname) # path: irrigation area of agent i, ABM_to_RiverWare folder
        A_to_R_Div_path = os.path.join(A_to_R_path, Div_fname) # path: diversion of agent i, ABM_to_RiverWare folder
        A_to_R_ET_path = os.path.join(A_to_R_path, ET_fname)   # path: ET of agent i, ABM_to_RiverWare folder

        with open(A_to_R_Irr_path,'w') as f_Irr:               # write irrigation area data to agt_names.txt
            f_Irr.write('\n'.join(list(Irr_headers_all[str(i).zfill(2) + '_' + agt_names[i - 1]])))
            f_Irr.write('\n')
            f_Irr.write('\n'.join(map(lambda x: "{:.2f}".format(x), Irr_all[:,i - 1])))
        
        with open(A_to_R_Div_path,'w') as f_Div:               # write diversion data to agt_names.txt
            f_Div.write('\n'.join(list(Div_headers_all[str(i).zfill(2) + '_' + agt_names[i - 1]])))
            f_Div.write('\n')
            f_Div.write('\n'.join(map(lambda x: "{:.2f}".format(x), Div_all[:,i - 1])))
    
    # save all irrigation area and diversion data to the big file for saving simulation time
    np.savetxt(R_to_A_Irr_all_path, Irr_all, fmt = '%.2f', delimiter = ',')              # save all irrigation area data (daily)
    np.savetxt(R_to_A_Div_all_path, Div_all, fmt = '%.2f', delimiter =',')               # save all diversion data (daily)
    np.savetxt(R_to_A_Annual_record_path, Annual_data_mat, fmt = '%.5f', delimiter =',') # save annual records of all variables (daily)
    
    # update and export interaction information
    interaction_info = [yr_idx + 1, d_num[-1] + 1]             # [yr_idx (1=1929,2=1930...), next year starting date] 
    np.savetxt('interaction_info.txt', interaction_info, fmt = '%i') 
    
    
