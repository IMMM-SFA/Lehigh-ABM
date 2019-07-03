# Subroutine of Bayesian Inference mapping and Cost-loss Model in ABM
# to model human decision making processes, San Juan River Basin (Group 1 to 3)
# Writen by Shih-Yu Huang

import numpy as np 

# Group 1 agent @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def G1_func(P_prep, P_vio, P_elev, P_area_exp, L_vec, extr_prep, extr_vio, z, Irr_prev, area_inc): 
    # correspond variables in the main program:  
    # prob_prep_winter_G1, prob_flow_vio, prob_Navajo_elev,  prob_expa_Irr, L_vec
    # extr_prep_G1, extr_flow_vio, z_val, Irr_prev, area_inc
    
    max_extr = max([extr_prep, extr_vio])   # find the maximum extremity
    
    if max_extr == extr_prep:               # choose precipitation as reference (prep => decision)
        # BI mapping: prep => decision
        P_update = P_area_exp*L_vec[2]*P_prep/(L_vec[2]*P_prep + (1-L_vec[2])*(1-P_prep))+ \
                    (1-P_area_exp)*((1-L_vec[2])*P_prep)/((1-L_vec[2])*P_prep + L_vec[2]*(1-P_prep))
    else:                                   # choose violation as reference (prep => Navajo elevation => violation => decision)
        # BI mapping: prep => Navajo elevation
        P_prep_elev = P_elev*L_vec[0]*P_prep/(L_vec[0]*P_prep + (1-L_vec[0])*(1-P_prep)) + \
                    (1-P_elev)*((1-L_vec[0])*P_prep)/((1-L_vec[0])*P_prep + L_vec[0]*(1-P_prep))
        if P_prep_elev == float('nan'):
            P_prep_elev = 0
        # BI mapping: Navajo elevation => flow violation (lambda = 1, i.e., certain causal relationship)
        P_elev_vio = P_vio*1*P_prep_elev/(1*P_prep_elev + (1-1)*(1-P_prep_elev)) + \
                    (1-P_vio)*((1-1)*P_prep_elev)/((1-1)*P_prep_elev + 1*(1-P_prep_elev))
        if P_elev_vio == float('nan'):
            P_elev_vio = 0
        # BI mapping: flow violation => decision
        P_update = P_area_exp*L_vec[1]*P_elev_vio/(L_vec[1]*P_elev_vio + (1-L_vec[1])*(1-P_elev_vio)) + \
                    (1-P_area_exp)*((1-L_vec[1])*P_elev_vio)/((1-L_vec[1])*P_elev_vio + L_vec[1]*(1-P_elev_vio))  
    
    if P_update == float('nan'):            # assign P_update = 0 (certainly reduce irrigation area) if nan occurs
            P_update = 0
    
    if P_update >= z:                       # Compare P and z and determine either expanding/reducing irrigation area
        Irr_agt = Irr_prev * (1 + area_inc)
        decision_var = 1
    else:
        Irr_agt = Irr_prev * (1 - area_inc)
        decision_var = 0
    
    return P_update, Irr_agt, decision_var

# Group 2 agent @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
def G2_func(P_prep1, P_prep2, P_vio, P_elev, P_area_exp, L_vec, extr_prep2, extr_vio, z, Irr_prev, area_inc): 
    # correspond variables in the main program: 
    # prob_prep_winter_G1, prob_prep_winter_G2, prob_flow_vio, prob_Navajo_elev,  prob_expa_Irr, L_vec
    # extr_prep_G2, extr_flow_vio, z_val, Irr_prev, area_inc

    max_extr = max([extr_prep2, extr_vio])    # find the maximum extremity
    
    if max_extr == extr_prep2:                # choose precipitation as reference (prep2 => decision)
        # BI mapping: prep2 => decision
        P_update = P_area_exp*L_vec[2]*P_prep2/(L_vec[2]*P_prep2 + (1-L_vec[2])*(1-P_prep2))+ \
                    (1-P_area_exp)*((1-L_vec[2])*P_prep2)/((1-L_vec[2])*P_prep2 + L_vec[2]*(1-P_prep2))       
    else:                                     # choose violation as reference (prep1 => Navajo elevation => violation => decision)
        # BI mapping: prep1 => Navajo elevation
        P_prep_elev = P_elev*L_vec[0]*P_prep1/(L_vec[0]*P_prep1 + (1-L_vec[0])*(1-P_prep1)) + \
                    (1-P_elev)*((1-L_vec[0])*P_prep1)/((1-L_vec[0])*P_prep1 + L_vec[0]*(1-P_prep1))
        if P_prep_elev == float('nan'):
            P_prep_elev = 0
        # BI mapping: Navajo elevation => flow violation (lambda = 1, i.e., certain causal relationship)
        P_elev_vio = P_vio*1*P_prep_elev/(1*P_prep_elev + (1-1)*(1-P_prep_elev)) + \
                    (1-P_vio)*((1-1)*P_prep_elev)/((1-1)*P_prep_elev + 1*(1-P_prep_elev))
        if P_elev_vio == float('nan'):
            P_elev_vio = 0
        # BI mapping: flow violation => decision
        P_update = P_area_exp*L_vec[1]*P_elev_vio/(L_vec[1]*P_elev_vio + (1-L_vec[1])*(1-P_elev_vio)) + \
                    (1-P_area_exp)*((1-L_vec[1])*P_elev_vio)/((1-L_vec[1])*P_elev_vio + L_vec[1]*(1-P_elev_vio))  
    
    if P_update == float('nan'):              # assign P_update = 0 (certainly reduce irrigation area) if nan occurs
            P_update = 0
    
    if P_update >= z:                         # Compare P and z and determine either expanding/reducing irrigation area
        Irr_agt = Irr_prev * (1 + area_inc)
        decision_var = 1
    else:
        Irr_agt = Irr_prev * (1 - area_inc)
        decision_var = 0
    
    return P_update, Irr_agt, decision_var    
    
# Group 3 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
def G3_func(P_prep1, P_prep3, P_vio, P_elev, P_NIIP, P_area_exp, L_vec, extr_prep3, extr_vio, extr_NIIP, z, Irr_prev, area_inc): 
    # correspond variables in the main program: 
    # prob_prep_winter_G1, prob_prep_winter_G3, prob_flow_vio, prob_Navajo_elev, prob_NIIP_div, prob_expa_Irr, L_vec
    # extr_prep_G3, extr_flow_vio, extr_NIIP_div, z_val, Irr_prev, area_inc
    
    max_extr = max([extr_prep3, extr_vio, extr_NIIP])
    
    if max_extr == extr_prep3:                # choose precipitation as reference (prep3 => decision)
        # BI mapping: prep3 => decision
        P_update = P_area_exp*L_vec[2]*P_prep3/(L_vec[2]*P_prep3 + (1-L_vec[2])*(1-P_prep3))+ \
                    (1-P_area_exp)*((1-L_vec[2])*P_prep3)/((1-L_vec[2])*P_prep3 + L_vec[2]*(1-P_prep3))
    elif max_extr == extr_vio:                # choose violation as reference (prep1 => Navajo elevation => violation => decision)
        # BI mapping: prep1 => Navajo elevation
        P_prep_elev = P_elev*L_vec[0]*P_prep1/(L_vec[0]*P_prep1 + (1-L_vec[0])*(1-P_prep1)) + \
                    (1-P_elev)*((1-L_vec[0])*P_prep1)/((1-L_vec[0])*P_prep1 + L_vec[0]*(1-P_prep1))
        if P_prep_elev == float('nan'):
            P_prep_elev = 0
        # BI mapping: Navajo elevation => flow violation (lambda = 1, i.e., certain causal relationship)
        P_elev_vio = P_vio*1*P_prep_elev/(1*P_prep_elev + (1-1)*(1-P_prep_elev)) + \
                    (1-P_vio)*((1-1)*P_prep_elev)/((1-1)*P_prep_elev + 1*(1-P_prep_elev))
        if P_elev_vio == float('nan'):
            P_elev_vio = 0
        # BI mapping: flow violation => decision
        P_update = P_area_exp*L_vec[1]*P_elev_vio/(L_vec[1]*P_elev_vio + (1-L_vec[1])*(1-P_elev_vio)) + \
                    (1-P_area_exp)*((1-L_vec[1])*P_elev_vio)/((1-L_vec[1])*P_elev_vio + L_vec[1]*(1-P_elev_vio))  
    else:                                     # choose NIIP diversion as reference (prep1 => Navajo elevation => NIIP diversion => decision)
        # prep1 => Navajo elevation
        P_prep_elev = P_elev*L_vec[0]*P_prep1/(L_vec[0]*P_prep1 + (1-L_vec[0])*(1-P_prep1)) + \
                    (1-P_elev)*((1-L_vec[0])*P_prep1)/((1-L_vec[0])*P_prep1 + L_vec[0]*(1-P_prep1))
        if P_prep_elev == float('nan'):
            P_prep_elev = 0
        # Navajo elevation => NIIP diversion (lambda = 1, i.e., certain causal relationship)
        P_elev_NIIP = P_NIIP*1*P_prep_elev/(1*P_prep_elev + (1-1)*(1-P_prep_elev)) + \
                    (1-P_NIIP)*((1-1)*P_prep_elev)/((1-1)*P_prep_elev + 1*(1-P_prep_elev))
        if P_elev_NIIP == float('nan'):
            P_elev_NIIP = 0
        # NIIP diversion => decision (lambda = 1, i.e., certain causal relationship)
        P_update = P_area_exp*1*P_elev_NIIP/(1*P_elev_NIIP + (1-1)*(1-P_elev_NIIP)) + \
                    (1-P_area_exp)*((1-1)*P_elev_NIIP)/((1-1)*P_elev_NIIP + 1*(1-P_elev_NIIP))
    
    if P_update == float('nan'):              # assign P_update = 0 (certainly reduce irrigation area) if nan occurs
        P_update = 0
    
    if P_update >= z:                         # Compare P and z and determine either expanding/reducing irrigation area
        Irr_agt = Irr_prev * (1 + area_inc)
        decision_var = 1
    else:
        Irr_agt = Irr_prev * (1 - area_inc)
        decision_var = 0
    
    return P_update, Irr_agt, decision_var 
