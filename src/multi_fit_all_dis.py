"""

"""
import os,sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import pyskew.plot_skewness as psk
import pyskew.utilities as utl
import pyskew.plot_gravity as pg
from geographiclib.geodesic import Geodesic
import pyrot.apwp as apwp
from pyrot.rot import Rot,get_pole_arc_misfit_uncertainty,fast_fit_circ,plot_error_from_points
import pyrot.max as pymax
from multi_circ_inv import fit_circ
from functools import reduce
from time import time
from scipy.interpolate import PchipInterpolator
import pmagpy.pmag as pmag
from mpl_toolkits.mplot3d import Axes3D

def subsets(lst, n):
    """Returns all subsets of lst of size exactly n in any order.
    lst is a Python list, and n is a non-negative integer.

    >>> three_subsets = subsets(list(range(5)), 3)
    >>> three_subsets.sort()  # Uses syntax we don't know yet to sort the list.
    >>> for subset in three_subsets:
    ...     print(subset)
    [0, 1, 2]
    [0, 1, 3]
    [0, 1, 4]
    [0, 2, 3]
    [0, 2, 4]
    [0, 3, 4]
    [1, 2, 3]
    [1, 2, 4]
    [1, 3, 4]
    [2, 3, 4]
    """
    sub_lst = [[l] for l in lst]
    for i in range(n-1):
        sub_lst = [ k + [j] for k in sub_lst for j in lst if j not in k and j > max(k)]
    return sub_lst

def gcd(lat1,lon1,lat2,lon2):
    rlat1,rlon1,rlat2,rlon2 = np.deg2rad((lat1,lon1,lat2,lon2))
    rdlon = abs(rlon1-rlon2)
    num = np.sqrt((np.cos(rlat2)*np.sin(rdlon))**2 + (np.cos(rlat1)*np.sin(rlat2) - np.sin(rlat1)*np.cos(rlat2)*np.cos(rdlon))**2)
    den = np.sin(rlat1)*np.sin(rlat2) + np.cos(rlat1)*np.cos(rlat2)*np.cos(rdlon)
    return np.rad2deg(np.arctan2(num,den))

padding = .05
geoid = Geodesic(6371.,0.)
hi_color = "#C4A58E"#plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
em_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
tr_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
undated_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
mean_color = "k"#plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
title_fontsize = 22
label_fontsize = 18
fontsize = 14

fig_dis = plt.figure(figsize=(9,4*len(sys.argv[1:])),dpi=200)
fig_dis.subplots_adjust(left=padding+.05,bottom=padding,right=1-padding,top=1-padding,wspace=0.05,hspace=0.00)
dis_pos = 100*len(sys.argv[1:]) + 11
age_step_dis = 0.1

for inv_data_path in sys.argv[1:]:

    inv_data = pd.read_csv(inv_data_path,index_col=0,dtype={"BendLat":float,"BendLon":float,"EMDis":float,"EMLon":float,"EMLat":float,"EMErr":float,"EM_Start_Dis":float,"EM_start_azi":float,"EMpols":object,"EMsds":object,"HIDis":float,"HILon":float,"HILat":float,"HIErr":float,"HI_Start_Dis":float,"HI_start_azi":float,"HIpols":object,"HIsds":object}).T

    for hs in inv_data.columns:
        try:
            inv_data[hs]["HIpols"] = list(map(float,inv_data[hs]["HIpols"].strip("[ ]").split()))
            inv_data[hs]["HIsds"] = list(map(float,inv_data[hs]["HIsds"].strip("[ ]").split()))
            inv_data[hs]["EMpols"] = list(map(float,inv_data[hs]["EMpols"].strip("[ ]").split()))
            inv_data[hs]["EMsds"] = list(map(float,inv_data[hs]["EMsds"].strip("[ ]").split()))
        except ValueError:
            inv_data[hs]["HIpols"] = list(map(float,inv_data[hs]["HIpols"].strip("[ ]").split(",")))
            inv_data[hs]["HIsds"] = list(map(float,inv_data[hs]["HIsds"].strip("[ ]").split(",")))
            inv_data[hs]["EMpols"] = list(map(float,inv_data[hs]["EMpols"].strip("[ ]").split(",")))
            inv_data[hs]["EMsds"] = list(map(float,inv_data[hs]["EMsds"].strip("[ ]").split(",")))

    ax_dis = fig_dis.add_subplot(dis_pos)

    if inv_data_path==sys.argv[-1]:
        ax_dis.tick_params(
        axis='both',          # changes the axis to apply changes
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge
        top=True,         # ticks along the top edge
        left=True,         # ticks along the left edge
        right=True,         # ticks along the right edge
        labelbottom=True, # labels along the bottom edge are var
        labelleft=True, # labels along the left edge are on
        labelright=False, # labels along the right edge are off
        tickdir="in",
#        length=10,
#        width=1,
        labelsize=fontsize) #changes size of tick labels
        ax_dis.set_yticks(np.arange(-1000,2500,500))
    else:
        ax_dis.tick_params(
        axis='both',          # changes the axis to apply changes
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge
        top=True,         # ticks along the top edge
        left=True,         # ticks along the left edge
        right=True,         # ticks along the right edge
        labelbottom=False, # labels along the bottom edge are var
        labelleft=True, # labels along the left edge are on
        labelright=False, # labels along the right edge are off
        tickdir="in",
#        length=10,
#        width=1,
        labelsize=fontsize) #changes size of tick labels
        ax_dis.set_yticks(np.arange(-500,2500,500))

    for k,(hs1,hs2) in enumerate(subsets(["Hawaii","Rurutu","Louisville HK19"],2)):
#        print(hs1,"-",hs2)
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][k]
        hs1_data = pd.read_excel("../data/pa_seamount_ages_subareal_included.xlsx",hs1)
        hs1_data = hs1_data[hs1_data["Quality"]=="g"]
        hs2_data = pd.read_excel("../data/pa_seamount_ages_subareal_included.xlsx",hs2)
        hs2_data = hs2_data[hs2_data["Quality"]=="g"]
        print("Circular Unc: ",np.sqrt(2)*(33./111.113)*(inv_data[hs1]["HIErr"]/(len(hs1_data.index)-3)),np.sqrt(2)*(33./111.113)*(inv_data[hs2]["HIErr"]/(len(hs2_data.index)-3)))

        if "Louisville" in hs1 and hs2=="Rurutu": hs1,hs2="Rurutu","Louisville HK19"

        if hs1!="Rurutu" and hs2!="Rurutu":
            inter_hs_dis,inter_hs_dis_sds,azis,ages,south_dis,south_sds = [],[],[],np.arange(0.,80.+age_step_dis,age_step_dis),[],[]
        else:
            if ("BendAge" in inv_data[hs1] and "BendAge" in inv_data[hs2]): bend_age = max(inv_data[hs1]["BendAge"],inv_data[hs2]["BendAge"])
            inter_hs_dis,inter_hs_dis_sds,azis,ages,south_dis,south_sds = [],[],[],np.array(list(np.arange(0.,10.+age_step_dis,age_step_dis))+list(np.arange(bend_age,72.+age_step_dis,age_step_dis))),[],[]
    #    if hs1!="Rurutu" and hs2!="Rurutu":
    #        hi_inter_hs_dis,hi_inter_hs_dis_sds,hi_azis,ages,hi_south_dis,hi_south_sds = [],[],[],np.arange(0.,85.,age_step_dis),[],[]
    #    else:
    #        hi_inter_hs_dis,hi_inter_hs_dis_sds,hi_azis,ages,hi_south_dis,hi_south_sds = [],[],[],np.arange(49.,85.,age_step_dis),[],[]
    #    em_inter_hs_dis,em_inter_hs_dis_sds,em_azis,em_south_dis,em_south_sds = [],[],[],[],[]
        for age in ages:
#            print("Age: ",age)

            if ("BendAge" in inv_data[hs1] and age<inv_data[hs1]["BendAge"]) or ("BendAge" not in inv_data[hs1] and age<48.5):
                hs1_b = np.sqrt(2)*(33./111.113)*np.sqrt(inv_data[hs1]["HIErr"]/(len(hs1_data.index)-6))
                if len(inv_data[hs1]["HIpols"])==2:
                    age_dis = ((age-inv_data[hs1]["HIpols"][1])/inv_data[hs1]["HIpols"][0] - inv_data[hs1]["HI_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs1]["HIDis"]))
                    hs1_a = np.sqrt(2*(((inv_data[hs1]["HIsds"][1]**2)*(1/inv_data[hs1]["HIpols"][0])**2) + ((inv_data[hs1]["HIsds"][0]**2)*(-(age-inv_data[hs1]["HI_mean_age"]-inv_data[hs1]["HIpols"][1])/(inv_data[hs1]["HIpols"][0]**2))**2)) + hs1_b**2)
#                    hs1_a = np.sqrt(((1/(inv_data[hs1]["HIpols"][1]))*inv_data[hs1]["HIAgeSd"])**2 + hs1_b**2)
                elif len(inv_data[hs1]["HIpols"])==3:
                    age_dis = ((((-inv_data[hs1]["HIpols"][1]+np.sqrt(inv_data[hs1]["HIpols"][1]**2 - 4*inv_data[hs1]["HIpols"][0]*(inv_data[hs1]["HIpols"][2]-age)))/(2*inv_data[hs1]["HIpols"][0]))) - inv_data[hs1]["HI_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs1]["HIDis"]))
                    x = np.linspace(0,inv_data[hs1]["EM_Start_Dis"],100)
                    center_pols,center_sds = utl.polyrecenter(inv_data[hs1]["HIpols"],x,np.ones(len(x))*inv_data[hs1]["HIAgeSd"],center=inv_data[hs1]["HI_mean_dis"])
                    psi = np.sqrt(center_pols[1]**2 - 4*center_pols[0]*(center_pols[2]-(age)))
                    d_dc = -1/psi
                    d_db = (center_pols[1]/psi-1)/(2*center_pols[0])
                    d_da = -(psi-center_pols[1])/(2*center_pols[0]**2) - (center_pols[2]-(age))/(center_pols[0]*psi)
                    hs1_a = np.sqrt(2*len(x)*(((center_sds[2]**2)*(d_dc**2)) + ((center_sds[1]**2)*(d_db**2)) + ((center_sds[0]**2)*(d_da**2))) + hs1_b**2)
#                    hs1_a = np.sqrt(2)*np.sqrt(((center_sds[2]**2)*(d_dc**2)) + ((center_sds[1]**2)*(d_db**2)) + ((center_sds[0]**2)*(d_da**2)))/np.sqrt(len(x))/np.sin(np.deg2rad(inv_data[hs1]["HIDis"]))
#                    hs1_a = np.sqrt(((1/(2*inv_data[hs1]["HIpols"][0]*age_dis*np.sin(np.deg2rad(inv_data[hs1]["HIDis"]))+inv_data[hs1]["HIpols"][1]))*inv_data[hs1]["HIAgeSd"])**2 + hs1_b**2)
                else: raise ValueError("Degree %d inverse not programmed yet"%(len(inv_data[hs1]["HIpols"])-1))
                hs1_geodict = geoid.ArcDirect(inv_data[hs1]["HILat"],inv_data[hs1]["HILon"],inv_data[hs1]["HI_start_azi"]+age_dis,-inv_data[hs1]["HIDis"]) #modeled geographic point for age
                hs1_phi = (hs1_geodict["azi2"]+90.)
            else:
                hs1_b = np.sqrt(2)*(33./111.113)*np.sqrt(inv_data[hs1]["EMErr"]/(len(hs1_data.index)-6))
                if len(inv_data[hs1]["EMpols"])==2:
                    age_dis = ((age-inv_data[hs1]["EMpols"][1])/inv_data[hs1]["EMpols"][0] - inv_data[hs1]["EM_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs1]["EMDis"]))
                    hs1_a = np.sqrt(2*(((inv_data[hs1]["EMsds"][1]**2)*(1/inv_data[hs1]["EMpols"][0])**2) + ((inv_data[hs1]["EMsds"][0]**2)*(-(age-inv_data[hs1]["EM_mean_age"]-inv_data[hs1]["EMpols"][1])/(inv_data[hs1]["EMpols"][0]**2))**2)) + hs1_b**2)
#                    hs1_a = np.sqrt(((1/(inv_data[hs1]["EMpols"][1]))*inv_data[hs1]["EMAgeSd"])**2 + hs1_b**2)
                elif len(inv_data[hs1]["EMpols"])==3:
                    age_dis = ((((-inv_data[hs1]["EMpols"][1]+np.sqrt(inv_data[hs1]["EMpols"][1]**2 - 4*inv_data[hs1]["EMpols"][0]*(inv_data[hs1]["EMpols"][2]-age)))/(2*inv_data[hs1]["EMpols"][0]))) - inv_data[hs1]["EM_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs1]["EMDis"]))
                    x = np.linspace(0,inv_data[hs1]["EM_Start_Dis"],100)
                    center_pols,center_sds = utl.polyrecenter(inv_data[hs1]["EMpols"],x,np.ones(len(x))*inv_data[hs1]["EMAgeSd"],center=inv_data[hs1]["EM_mean_dis"])
                    psi = np.sqrt(center_pols[1]**2 - 4*center_pols[0]*(center_pols[2]-(age)))
                    d_dc = -1/psi
                    d_db = (center_pols[1]/psi-1)/(2*center_pols[0])
                    d_da = -(psi-center_pols[1])/(2*center_pols[0]**2) - (center_pols[2]-(age))/(center_pols[0]*psi)
                    hs1_a = np.sqrt(2*len(x)*(((center_sds[2]**2)*(d_dc**2)) + ((center_sds[1]**2)*(d_db**2)) + ((center_sds[0]**2)*(d_da**2))) + hs1_b**2)
#                    hs1_a = np.sqrt(2)*np.sqrt(((center_sds[2]**2)*(d_dc**2)) + ((center_sds[1]**2)*(d_db**2)) + ((center_sds[0]**2)*(d_da**2)))/np.sqrt(len(x))
#                    hs1_a = np.sqrt(((1/(2*inv_data[hs1]["EMpols"][0]*age_dis*np.sin(np.deg2rad(inv_data[hs1]["EMDis"]))+inv_data[hs1]["EMpols"][1]))*inv_data[hs1]["EMAgeSd"])**2 + hs1_b**2)
                else: raise ValueError("Degree %d inverse not programmed yet"%(len(inv_data[hs1]["EMpols"])-1))
                hs1_geodict = geoid.ArcDirect(inv_data[hs1]["EMLat"],inv_data[hs1]["EMLon"],inv_data[hs1]["EM_start_azi"]+age_dis,-inv_data[hs1]["EMDis"]) #modeled geographic point for age
                hs1_phi = (hs1_geodict["azi2"]+90.)

            if ("BendAge" in inv_data[hs2] and age<inv_data[hs2]["BendAge"]) or ("BendAge" not in inv_data[hs1] and age<48.5):
                hs2_b = np.sqrt(2)*(33./111.113)*np.sqrt(inv_data[hs2]["HIErr"]/(len(hs2_data.index)-6))
                if len(inv_data[hs2]["HIpols"])==2:
                    age_dis = ((age-inv_data[hs2]["HIpols"][1])/inv_data[hs2]["HIpols"][0] - inv_data[hs2]["HI_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs2]["HIDis"]))
                    hs2_a = np.sqrt(2*(((inv_data[hs2]["HIsds"][1]**2)*(1/inv_data[hs2]["HIpols"][0])**2) + ((inv_data[hs2]["HIsds"][0]**2)*(-(age-inv_data[hs2]["HI_mean_age"]-inv_data[hs2]["HIpols"][1])/(inv_data[hs2]["HIpols"][0]**2))**2)) + hs2_b**2)
#                    hs2_a = np.sqrt(((1/(inv_data[hs2]["HIpols"][1]))*inv_data[hs2]["HIAgeSd"])**2 + hs2_b**2)
                elif len(inv_data[hs2]["HIpols"])==3:
                    age_dis = ((((-inv_data[hs2]["HIpols"][1]+np.sqrt(inv_data[hs2]["HIpols"][1]**2 - 4*inv_data[hs2]["HIpols"][0]*(inv_data[hs2]["HIpols"][2]-age)))/(2*inv_data[hs2]["HIpols"][0]))) - inv_data[hs2]["HI_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs2]["HIDis"]))
                    x = np.linspace(0,inv_data[hs2]["EM_Start_Dis"],100)
                    center_pols,center_sds = utl.polyrecenter(inv_data[hs2]["HIpols"],x,np.ones(len(x))*inv_data[hs2]["HIAgeSd"],center=inv_data[hs2]["HI_mean_dis"])
                    psi = np.sqrt(center_pols[1]**2 - 4*center_pols[0]*(center_pols[2]-(age)))
                    d_dc = -1/psi
                    d_db = (center_pols[1]/psi-1)/(2*center_pols[0])
                    d_da = -(psi-center_pols[1])/(2*center_pols[0]**2) - (center_pols[2]-(age))/(center_pols[0]*psi)
                    hs2_a = np.sqrt(2*len(x)*(((center_sds[2]**2)*(d_dc**2)) + ((center_sds[1]**2)*(d_db**2)) + ((center_sds[0]**2)*(d_da**2))) + hs2_b**2)
#                    hs2_a = np.sqrt(2)*np.sqrt(((center_sds[2]**2)*(d_dc**2)) + ((center_sds[1]**2)*(d_db**2)) + ((center_sds[0]**2)*(d_da**2)))/np.sqrt(len(x))
#                    hs2_a = np.sqrt(((1/(2*inv_data[hs2]["HIpols"][0]*age_dis*np.sin(np.deg2rad(inv_data[hs2]["HIDis"]))+inv_data[hs2]["HIpols"][1]))*inv_data[hs2]["HIAgeSd"])**2 + hs2_b**2)
                else: raise ValueError("Degree %d inverse not programmed yet"%(len(inv_data[hs2]["HIpols"])-1))
                hs2_geodict = geoid.ArcDirect(inv_data[hs2]["HILat"],inv_data[hs2]["HILon"],inv_data[hs2]["HI_start_azi"]+age_dis,-inv_data[hs2]["HIDis"]) #modeled geographic point for age
                hs2_phi = (hs2_geodict["azi2"]+90.)
            else:
                hs2_b = np.sqrt(2)*(33./111.113)*np.sqrt(inv_data[hs2]["EMErr"]/(len(hs2_data.index)-6))
                if len(inv_data[hs2]["EMpols"])==2:
                    age_dis = ((age-inv_data[hs2]["EMpols"][1])/inv_data[hs2]["EMpols"][0] - inv_data[hs2]["EM_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs2]["EMDis"]))
                    hs2_a = np.sqrt(2*(((inv_data[hs2]["EMsds"][1]**2)*(1/inv_data[hs2]["EMpols"][0])**2) + ((inv_data[hs2]["EMsds"][0]**2)*(-(age-inv_data[hs2]["EM_mean_age"]-inv_data[hs2]["EMpols"][1])/(inv_data[hs2]["EMpols"][0]**2))**2)) + hs2_b**2)
#                    hs2_a = np.sqrt(((1/(inv_data[hs2]["EMpols"][1]))*inv_data[hs2]["EMAgeSd"])**2 + hs2_b**2)
                elif len(inv_data[hs2]["EMpols"])==3:
                    age_dis = ((((-inv_data[hs2]["EMpols"][1]+np.sqrt(inv_data[hs2]["EMpols"][1]**2 - 4*inv_data[hs2]["EMpols"][0]*(inv_data[hs2]["EMpols"][2]-age)))/(2*inv_data[hs2]["EMpols"][0]))) - inv_data[hs2]["EM_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs2]["EMDis"]))
                    x = np.linspace(0,inv_data[hs2]["EM_Start_Dis"],100)
                    center_pols,center_sds = utl.polyrecenter(inv_data[hs2]["EMpols"],x,np.ones(len(x))*inv_data[hs2]["EMAgeSd"],center=inv_data[hs2]["EM_mean_dis"])
                    psi = np.sqrt(center_pols[1]**2 - 4*center_pols[0]*(center_pols[2]-(age)))
                    d_dc = -1/psi
                    d_db = (center_pols[1]/psi-1)/(2*center_pols[0])
                    d_da = -(psi-center_pols[1])/(2*center_pols[0]**2) - (center_pols[2]-(age))/(center_pols[0]*psi)
                    hs2_a = np.sqrt(2*len(x)*(((center_sds[2]**2)*(d_dc**2)) + ((center_sds[1]**2)*(d_db**2)) + ((center_sds[0]**2)*(d_da**2))) + hs2_b**2)
#                    hs2_a = np.sqrt(2)*np.sqrt(((center_sds[2]**2)*(d_dc**2)) + ((center_sds[1]**2)*(d_db**2)) + ((center_sds[0]**2)*(d_da**2)))/np.sqrt(len(x))/np.sin(np.deg2rad(inv_data[hs2]["HIDis"]))
#                    hs2_a = np.sqrt(((1/(2*inv_data[hs2]["EMpols"][0]*age_dis*np.sin(np.deg2rad(inv_data[hs2]["EMDis"]))+inv_data[hs2]["EMpols"][1]))*inv_data[hs2]["EMAgeSd"])**2 + hs2_b**2)
                else: raise ValueError("Degree %d inverse not programmed yet"%(len(inv_data[hs2]["EMpols"])-1))
                hs2_geodict = geoid.ArcDirect(inv_data[hs2]["EMLat"],inv_data[hs2]["EMLon"],inv_data[hs2]["EM_start_azi"]+age_dis,-inv_data[hs2]["EMDis"]) #modeled geograpEMc point for age
                hs2_phi = (hs2_geodict["azi2"]+90.)

#            print(hs1_geodict["lat2"],hs1_geodict["lon2"],hs2_geodict["lat2"],hs2_geodict["lon2"])
            IHS_geodict = geoid.Inverse(hs1_geodict["lat2"],hs1_geodict["lon2"],hs2_geodict["lat2"],hs2_geodict["lon2"])
            hs1_sd = (hs1_a*hs1_b)/np.sqrt((hs1_a*np.sin(np.deg2rad(hs1_phi-IHS_geodict["azi1"])))**2 + (hs1_b*np.cos(np.deg2rad(hs1_phi-IHS_geodict["azi1"])))**2)/np.sqrt(2)
            hs2_sd = (hs2_a*hs2_b)/np.sqrt((hs2_a*np.sin(np.deg2rad(hs2_phi-IHS_geodict["azi2"])))**2 + (hs2_b*np.cos(np.deg2rad(hs2_phi-IHS_geodict["azi2"])))**2)/np.sqrt(2)
            inter_hs_dis.append(IHS_geodict["a12"])
            azis.append(IHS_geodict["azi1"])
            inter_hs_dis_sds.append(np.sqrt(hs1_sd**2+hs2_sd**2))

    #        south_dis.append(IHS_geodict["a12"]*np.cos(np.deg2rad(180-IHS_geodict["azi1"])))
            south_dis.append(abs(hs1_geodict["lat2"]-hs2_geodict["lat2"]))
            hs1_ssd = (hs1_a*hs1_b)/np.sqrt((hs1_a*np.sin(np.deg2rad(hs1_phi-180)))**2 + (hs1_b*np.cos(np.deg2rad(hs1_phi-180)))**2)/np.sqrt(2)
            hs2_ssd = (hs2_a*hs2_b)/np.sqrt((hs2_a*np.sin(np.deg2rad(hs2_phi-0)))**2 + (hs2_b*np.cos(np.deg2rad(hs2_phi-0)))**2)/np.sqrt(2)
            south_sds.append(np.sqrt(hs1_ssd**2+hs2_ssd**2))

        if ("BendAge" in inv_data[hs1] and "BendAge" in inv_data[hs2]):
            hi_bend_age = round(min([inv_data[hs1]["BendAge"],inv_data[hs2]["BendAge"]]),1)
            em_bend_age = round(max([inv_data[hs1]["BendAge"],inv_data[hs2]["BendAge"]]),1)
            if hs1 != "Rurutu" and hs2 != "Rurutu": bend_age = (inv_data[hs1]["BendAge"]+inv_data[hs2]["BendAge"])/2
            else: bend_age = round(max([inv_data[hs1]["BendAge"],inv_data[hs2]["BendAge"]]),1)
            hi_idx = np.argmin((ages-hi_bend_age)**2)-2
            em_idx = np.argmin((ages-em_bend_age)**2)+2
        else:
            hi_bend_age = 48.5
            em_bend_age = 48.5
            bend_age = 48.5
            hi_idx = np.argmin((ages-hi_bend_age)**2)
            em_idx = np.argmin((ages-em_bend_age)**2)

        if "Louisville" in hs1: title1,title2 = "Louisville",hs2
        elif "Louisville" in hs2: title1,title2 = hs1,"Louisville"
        else: title1,title2 = hs1,hs2
        if hs1=="Rurutu" or hs2=="Rurutu":
            hi_idx = int(10./age_step_dis)
            ax_dis.plot(ages[:hi_idx],111.113*(np.array(inter_hs_dis)-inter_hs_dis[0])[:hi_idx],color=color,label="%s-%s"%(title1,title2))
            ax_dis.fill_between(ages[:hi_idx], 111.113*(np.array(inter_hs_dis)-inter_hs_dis[0]+2*np.array(inter_hs_dis_sds))[:hi_idx], 111.113*(np.array(inter_hs_dis)-inter_hs_dis[0]-2*np.array(inter_hs_dis_sds))[:hi_idx], color=color, alpha=0.15,zorder=-1)
            ax_dis.plot(ages[hi_idx+1:],111.113*(np.array(inter_hs_dis)-inter_hs_dis[0])[hi_idx+1:],color=color)
            ax_dis.fill_between(ages[hi_idx+1:], 111.113*(np.array(inter_hs_dis)-inter_hs_dis[0]+2*np.array(inter_hs_dis_sds))[hi_idx+1:], 111.113*(np.array(inter_hs_dis)-inter_hs_dis[0]-2*np.array(inter_hs_dis_sds))[hi_idx+1:], color=color, alpha=0.15,zorder=-1)
        else:
            ax_dis.plot(ages,111.113*(np.array(inter_hs_dis)-inter_hs_dis[0]),color=color,label="%s-%s"%(title1,title2))
    #        ax_dis.plot(ages,111.113*(np.array(inter_hs_dis)-inter_hs_dis[0]+2*np.array(inter_hs_dis_sds)),color=color,linestyle=":")
    #        ax_dis.plot(ages,111.113*(np.array(inter_hs_dis)-inter_hs_dis[0]-2*np.array(inter_hs_dis_sds)),color=color,linestyle=":")
            ax_dis.fill_between(ages, 111.113*(np.array(inter_hs_dis)-inter_hs_dis[0]+2*np.array(inter_hs_dis_sds)), 111.113*(np.array(inter_hs_dis)-inter_hs_dis[0]-2*np.array(inter_hs_dis_sds)), color=color, alpha=0.15,zorder=-1)

        rates = 111.113*(np.diff(np.array(inter_hs_dis)-inter_hs_dis[0])/age_step_dis)
        south_rates = 111.113*(np.diff(np.array(south_dis))/age_step_dis)
    #    south_rates = rates*np.cos(np.deg2rad(180-np.array(azis[:-1])))
        rate_sds = 111.113*np.sqrt((np.array(inter_hs_dis_sds[:-1])**2 + np.array(inter_hs_dis_sds[1:])**2))#/len(inter_hs_dis_sds))
        if hs1 != "Rurutu" and hs2 != "Rurutu": rate_sds[:hi_idx] = rate_sds[:hi_idx]/bend_age
        rate_sds[em_idx:] = rate_sds[em_idx:]/(80-bend_age)
        print(bend_age,rate_sds[em_idx:].min(),np.median(rate_sds[em_idx:]),rate_sds[em_idx:].mean(),rate_sds[em_idx:].max())
#        if hs1!="Rurutu" and hs2!="Rurutu":
        print("-------------------------%s-%s"%(hs1,hs2))
        print("\tMax Dis",111.113*(inter_hs_dis[-1] - inter_hs_dis[0]))
        print("\tMax Dis Unc",111.113*inter_hs_dis_sds[-1])
        print("\tMax HI Dis", 111.113*((inter_hs_dis[hi_idx] - inter_hs_dis[0])))
        print("\tMax Dis Unc",111.113*inter_hs_dis_sds[hi_idx])

        print("-------------------------")

#        if hs1!="Rurutu" and hs2!="Rurutu":
        print("\tHI Avg Rate",111.113*((inter_hs_dis[hi_idx] - inter_hs_dis[0])/(ages[hi_idx]-ages[0])),sum(rates[:hi_idx])/len(rates[:hi_idx]))
        print("\tHI Rate Range",min(rates[:hi_idx]),max(rates[:hi_idx]))
        print("\tHI Avg. Rate Unc",111.113*np.sqrt((inter_hs_dis_sds[hi_idx]**2 + inter_hs_dis_sds[0]**2)/((ages[hi_idx]-ages[0])**2)))
#        print("\tMean HI Rate Unc",np.sqrt(sum(rate_sds[:hi_idx]**2)/len(rate_sds[:hi_idx])))
#        print("\tHI Avg Southward Rate",111.113*((south_dis[hi_idx] - south_dis[0])/(ages[hi_idx]-ages[0])),sum(south_rates[:hi_idx])/len(south_rates[:hi_idx]))
#        print("\tHI Southward Rate Range",min(south_rates[:hi_idx]),max(south_rates[:hi_idx]))
#        print("\tHI Avg. Southward Rate Unc Old Method",(111.113/(ages[hi_idx]-ages[0]))*np.sqrt((south_sds[0]**2 + south_sds[hi_idx]**2)))

        print("-------------------------")

        print("\tEM Avg Rate",111.113*((inter_hs_dis[-1] - inter_hs_dis[em_idx])/(ages[-1]-ages[em_idx])),sum(rates[em_idx:])/len(rates[em_idx:]))
        print("\tEM Rate Range",min(rates[em_idx:]),max(rates[em_idx:]))
        print("\tEM Avg. Rate Unc",111.113*np.sqrt((inter_hs_dis_sds[-1]**2 + inter_hs_dis_sds[em_idx]**2)/((ages[-1]-ages[em_idx])**2)))
#        print("\tMean EM Rate Unc",np.sqrt(sum(rate_sds[em_idx:]**2)/len(rate_sds[em_idx:])))
#        print("\tEM Avg Southward Rate",111.113*((south_dis[-1] - south_dis[em_idx])/(ages[-1]-ages[em_idx])),sum(south_rates[em_idx:])/len(south_rates[em_idx:]))
#        print("\tEM Southward Rate Range",min(south_rates[em_idx:]),max(south_rates[em_idx:]))
#        print("\tEM Avg. Southward Rate Unc Old Method",(111.113/(ages[-1]-ages[em_idx]))*np.sqrt((south_sds[em_idx]**2 + south_sds[-1]**2)))

        if "Louisville" in hs1: title1,title2 = "Louisville",hs2
        elif "Louisville" in hs2: title1,title2 = hs1,"Louisville"
        else: title1,title2 = hs1,hs2

    #    m.set_global()
    #    plt.show()
    #    sys.exit()

    if inv_data_path==sys.argv[1]:
        ax_dis.legend(loc="upper left",framealpha=.7,fontsize=fontsize)
        ax_dis.set_title("Inter-Hotspot Distances",fontsize=title_fontsize)
        fig_dis.text(0.0001, 0.5, "Change in Hotspot Distance (km)", va='center', rotation='vertical',fontsize=label_fontsize)
        ax_dis.set_ylabel("",fontsize=label_fontsize)
#        ax_dis.set_ylabel("Change in Hotspot Distance (km)",fontsize=label_fontsize)
    if inv_data_path==sys.argv[-1]: ax_dis.set_xlabel("Age (Ma)",fontsize=label_fontsize)
    ax_dis.annotate(chr(64+dis_pos%10+3)+")",xy=(1-0.04,0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="bottom",ha="right")
    ax_dis.axhline(0.,color="k",linestyle="--",zorder=0)
    ax_dis.set_ylim(-1000.,2000.)
    ax_dis.set_xlim(0.,80.)

    dis_pos += 1

fig_dis.savefig("../results/Bend_IHSD.png",dpi=200,bbox_inches="tight")
fig_dis.savefig("../results/Bend_IHSD.pdf",dpi=200,bbox_inches="tight",transparent=True)
