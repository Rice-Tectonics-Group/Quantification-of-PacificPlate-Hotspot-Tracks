import os,sys
import pandas as pd
import numpy as np
from time import time
import pyskew.plot_skewness as psk
import pyskew.plot_geographic as pgeo
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
from pyrot.rot import *
from pyrot.reconstruction import PlateReconstruction
from functools import cmp_to_key
from glob import glob
import pyskew.plot_gravity as pg
import cartopy.crs as ccrs
import pmagpy.ipmag as ipmag

def pole_arc_fitfunc(tpole,ppoles,bend,sml_circ,l1_dis,ignore_bend):
    (tlat,tlon,tdis),_,dis,s1_errors = get_pole_arc_misfit_uncertainty(tpole,ppoles,bend,sml_circ=sml_circ,ignore_bend=ignore_bend)
#    if round(geoid.Inverse(*bend,*tpole)["a12"],1)!=tdis: return inf
    if l1_dis: tot_error = sum((abs(dis-tdis)/s1_errors))
    else: tot_error = sum(((dis-tdis)/s1_errors)**2)
    return tot_error

def get_pole_arc_misfit_uncertainty(tpole,ppoles,bend,sml_circ=False,ignore_bend=False):
    tlat,tlon = tpole
    dis,azis,s1_errors=[],[],[]
    for plat,plon,a,b,phi in ppoles:
        geo_dict = geoid.Inverse(tlat,tlon,plat,plon)
        dis.append(geo_dict['a12'])
        azi = geo_dict['azi1']
        azis.append(azi)
        s1_errors.append(((a*b)/sqrt(((cos(phi-azi)*b)**2 + (sin(phi-azi)*a)**2)))/sqrt(2))
    if sml_circ:
        if not ignore_bend: tdis = geoid.Inverse(tlat,tlon,*bend)["a12"]
        else: tdis = get_max_likelyhood_small_circ_radius(np.array(dis),np.array(s1_errors))
#    if sml_circ: tdis = get_max_likelyhood_small_circ_radius(dis,s1_errors)
    else:
        tdis = 90
        geo_dict = geoid.Inverse(tlat,tlon,*bend)
        dis.append(geo_dict['a12'])
        azi = geo_dict['azi1']
        azis.append(azi)
        s1_errors.append(.01)
    dis,s1_errors,azis = array(dis),array(s1_errors),array(azis)
    return (tlat,tlon,tdis),(min(azis),max(azis)),dis,s1_errors

def get_max_likelyhood_small_circ_radius(dis,s1_errors): #From Gordon and Cox 1984 (eqn. A13)
    return sum(dis/(s1_errors**2))/sum(1/(s1_errors**2))

def fit_circ(ppoles1,ppoles2,bend,gridspace=1,pvalue=.05,lat_range=(-90,90),lon_range=(0,360),sml_circ=False,north_hemisphere=True,finish=True,l1_dis=False,ignore_bend=False): #Follows method laid out in Gordon and Cox 1984 Appendix A
    """
    Finds great or small circle which minimizes pole_arc_fitfunc or the distance between the circle and a group of poles

    Based on the grid search algorithm laid out in Cox and Gordon 1984 Appendix A

    Parameters
    --------------

    ppoles: nested list or array of values encoding poles and elliptical uncertainties
            i.e. [[lat1,lon1,a1,b1,phi1],[lat2,lon2,a2,b2,phi2],...]
    gridspace: spacing in degrees to use for lat-lon grid search
    pvalue: the pvalue to use for calculation of within confidence points during gridsearch
    lat_range: the domain over which to grid search latitude
    lon_range: the domain over which to grid search longitude
    sml_circ: a boolean controlling if the function fits a great or small circle to the data
    north_hemisphere: a boolean controlling if the function returns the northern or southern hemisphere pole
    finish: boolean controlling if the grid search is finished with a simplex algorithm to move off grid
    l1_dis: boolean controlling if the misfit function uses a standard l2 norm or an l1 norm

    Returns
    --------------

    ep_lat: 
    """
    #initialize variables for iteration
#    if pvalue<1 and pvalue>0: chi2_cutoff1 = chi2.ppf(1-pvalue,len(ppoles1)-3)
#    else: raise ValueError("pvalue must be less than 1 and greater than 0")
#    if pvalue<1 and pvalue>0: chi2_cutoff2 = chi2.ppf(1-pvalue,len(ppoles2)-3)
#    else: raise ValueError("pvalue must be less than 1 and greater than 0")
    sml_circ1,sml_circ2 = sml_circ,sml_circ
    error_points1,error_points2,a,b,phi = [],[],0,1e9,0

    lons = arange(*lon_range,gridspace)
    lats = arange(*lat_range,gridspace)
    lon_mesh, lat_mesh = meshgrid(lons,lats)
    ep1_chi2_surf = np.zeros(lon_mesh.shape)
    ep2_chi2_surf = np.zeros(lon_mesh.shape)

    #find euler pole and all poles that are within confidence
    min_tot_error1,min_tot_error2 = np.inf,np.inf
    for i,tlon in enumerate(lons):
        for j,tlat in enumerate(lats):

            #First Inv.
            (tlat1,tlon1,tdis1),(tmin_azi1,tmax_azi1),dis1,s1_errors1 = get_pole_arc_misfit_uncertainty((tlat,tlon),ppoles1,bend,sml_circ=sml_circ1,ignore_bend=ignore_bend)
            if l1_dis: tot_error1 = sum((abs(dis1-tdis1)/s1_errors1))
            else: tot_error1 = sum(((dis1-tdis1)/s1_errors1)**2)
            if tot_error1<min_tot_error1:
                min_tot_error1,ep1_lat,ep1_lon,ep1_dis,min_azi1,max_azi1 = tot_error1,tlat1,tlon1,tdis1,tmin_azi1,tmax_azi1 #check new best
            ep1_chi2_surf[j,i] = tot_error1 #check in confidence limits

            #Second Inv.
            (tlat2,tlon2,tdis2),(tmin_azi2,tmax_azi2),dis2,s2_errors2 = get_pole_arc_misfit_uncertainty((tlat,tlon),ppoles2,bend,sml_circ=sml_circ2,ignore_bend=ignore_bend)
            if l1_dis: tot_error2 = sum((abs(dis2-tdis2)/s2_errors2))
            else: tot_error2 = sum(((dis2-tdis2)/s2_errors2)**2)
            if tot_error2<min_tot_error2:
                min_tot_error2,ep2_lat,ep2_lon,ep2_dis,min_azi2,max_azi2 = tot_error2,tlat2,tlon2,tdis2,tmin_azi2,tmax_azi2 #check new best
            ep2_chi2_surf[j,i] = tot_error2 #check in confidence limits

    if finish:
        (ep1_lat,ep1_lon),min_tot_error1,_,_,_ = fmin(pole_arc_fitfunc,(ep1_lat,ep1_lon),args=(ppoles1,bend,sml_circ1,l1_dis,ignore_bend),full_output=True)
        (ep1_lat,ep1_lon,ep1_dis),(min_azi1,max_azi1),_,_ = get_pole_arc_misfit_uncertainty((ep1_lat,ep1_lon),ppoles1,bend,sml_circ=sml_circ1)

        (ep2_lat,ep2_lon),min_tot_error2,_,_,_ = fmin(pole_arc_fitfunc,(ep2_lat,ep2_lon),args=(ppoles2,bend,sml_circ2,l1_dis,ignore_bend),full_output=True)
        (ep2_lat,ep2_lon,ep2_dis),(min_azi2,max_azi2),_,_ = get_pole_arc_misfit_uncertainty((ep2_lat,ep2_lon),ppoles2,bend,sml_circ=sml_circ2)
        min_tot_error = min_tot_error1+min_tot_error2

    #ensure proper antipode is returned
#    if north_hemisphere and ep1_lat<0: ep1_lat,ep1_lon,ep1_dis = -ep1_lat,(180+ep1_lon)%360,180-ep1_dis
#    elif not north_hemisphere and ep1_lat>0: ep1_lat,ep1_lon,ep1_dis = -ep1_lat,(180+ep1_lon)%360,180-ep1_dis
#    if north_hemisphere and ep2_lat<0: ep2_lat,ep2_lon,ep2_dis = -ep2_lat,(180+ep2_lon)%360,180-ep2_dis
#    elif not north_hemisphere and ep2_lat>0: ep2_lat,ep2_lon,ep2_dis = -ep2_lat,(180+ep2_lon)%360,180-ep2_dis

    #Return within confidence poles to determine error ellipse
    #exclude current euler pole and antipodal points without messing up hemispheres
#    corrected_error_points1 = list(filter(lambda x: sign(x[2]-90)==sign(ep1_dis-90), error_points1))
#    if [ep1_lat,ep1_lon,ep1_dis] in corrected_error_points1: corrected_error_points1.remove([ep1_lat,ep1_lon,ep1_dis])

#    corrected_error_points2 = list(filter(lambda x: sign(x[2]-90)==sign(ep2_dis-90), error_points2))
#    if [ep2_lat,ep2_lon,ep2_dis] in corrected_error_points2: corrected_error_points2.remove([ep2_lat,ep2_lon,ep2_dis])

    return ep1_lat,ep1_lon,ep1_dis,ep2_lat,ep2_lon,ep2_dis,min_tot_error,lon_mesh,lat_mesh,ep1_chi2_surf,ep2_chi2_surf

##################################################################################################################

inv_data_path = "/home/kevin/Projects/DispursionOfHSAges/data/Bend_Inversion_data_3const.csv"
#rec_path = "/home/kevin/Projects/DispursionOfHSAges/data/reconstructions/AGHJ06.csv"
#rec_path = "/home/kevin/Projects/DispursionOfHSAges/data/reconstructions/Koiv2014.csv"
rec_path = "/home/kevin/Projects/DispursionOfHSAges/data/reconstructions/PA_nhotspot_inversion.csv"
hi_color = "#C4A58E"#plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
em_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
bend_resolution = .1
geoid = Geodesic(6731.,0.)#Geodesic.WGS84
ignore_bend = True

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

PaHsReconst = PlateReconstruction.read_csv(rec_path)

#---------------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(9,9),dpi=200)
m = pgeo.create_basic_map(projection="npstere",stereo_bound_lat=40,resolution="10m",center_lon=-60)

for rot in PaHsReconst.get_rots():
#    if rot.age_f%5: continue
    if rot.age_f<40 or rot.age_f>55: continue
#    elif rot.age_f%
    lat,lon = rot.lat,rot.lon
    a,b,azi = cov_to_ellipse(lat,lon,rot.cov[:-1,:-1])
    if lat<0: lat,lon=-lat,(lon+180)%360
    print("Age:",rot.age_f)
    print(lat,lon,a,b,azi)
    print("----------------------------------------------------------")
    if a==0 or b==0: a,b=.01,.01
    if rot.age_f==47: color,zorder,fill_ell = "tab:red",5,True
    elif rot.age_f==48: color,zorder,fill_ell = "tab:pink",4,True
    else: color,zorder,fill_ell = "tab:blue",3,False
    m = psk.plot_pole(lon,lat,azi,2*a,2*b,marker='o',color=color,m=m,label="Pacific Reconstruction Pole %.1f"%rot.age_f,zorder=zorder,alpha=.3,filled=fill_ell)

fig.savefig("./poles.png")

#---------------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(9,9),dpi=200)
m = pgeo.create_basic_map(projection="stere",center_lat=40,center_lon=260,resolution="10m")

hi_rot = PaHsReconst[0:47]
em_rot = PaHsReconst[48:80]
for i,rot in enumerate([hi_rot,em_rot]):
    if i==0: lat,lon,_,_ = PaHsReconst[47:0].rotate(rot.lat,rot.lon)
    else: lat,lon,_,_ = PaHsReconst[80:0].rotate(rot.lat,rot.lon)
    a,b,azi = cov_to_ellipse(lat,lon,rot.cov[:-1,:-1])
    if lat<0: lat,lon=-lat,(lon+180)%360
    print("Age:",rot.age_i,rot.age_f)
    print(lat,lon,a,b,azi)
    print("----------------------------------------------------------")
    if a==0 or b==0: a,b=.01,.01
    if i==0: color,marker = hi_color,"o"
    elif i==1: color,marker = em_color,"s"
    else: color = "tab:blue"
    m = psk.plot_pole(lon,lat,azi,2*a,2*b,marker=marker,color=color,m=m,label="Pacific Reconstruction Pole %.1f"%rot.age_f,zorder=10,alpha=.3)

for hs in inv_data.columns:

    if "Louisville" in hs: hs = "Louisville HK19"

    data = pd.read_excel("/home/kevin/Projects/DispursionOfHSAges/data/pa_seamount_ages_subareal_included.xlsx",hs)
    data["Maj"] = (33.0/111.113)*np.sqrt(2) #seamount 1d 1sigma from Chengzu
    data["Min"] = (33.0/111.113)*np.sqrt(2) #seamount 1d 1sigma from Chengzu
    data["Azi"] = 0.0 #Circular

    if hs == "Hawaii":
        hi_data = data[data["Latitude"]<33.]
        tr_data = data[(data["Latitude"]>33.) & (data["Latitude"]<33.)]
        em_data = data[data["Latitude"]>33.]
    elif "Louisville" in hs:
        hi_data = data[data["Longitude"]>-169.]
        tr_data = data[(data["Longitude"]>-169.) & (data["Longitude"]<=-169.)]
        em_data = data[data["Longitude"]<-169.]
    elif hs == "Rurutu":
        cut_age = 20.
        hi_data = data[data["Age (Ma)"]<cut_age]
        tr_data = data[(data["Age (Ma)"]>cut_age) & (data["Age (Ma)"]<cut_age)]
        em_data = data[data["Age (Ma)"]>=cut_age]

    good_hi_data = hi_data[hi_data["Quality"]=="g"]
    good_em_data = em_data[em_data["Quality"]=="g"]
    em_poles = good_em_data[["Latitude","Longitude","Maj","Min","Azi"]].values.tolist()
    hi_poles = good_hi_data[["Latitude","Longitude","Maj","Min","Azi"]].values.tolist()

    good_geo_data = data[data["Quality"]=="g"]
    all_poles = good_geo_data[["Latitude","Longitude","Maj","Min","Azi"]].values.tolist()
    good_em_data = em_data[em_data["Quality"]=="g"]
    good_geo_data["IncOnly"] = True
    good_geo_data["DecOnly"] = False
    good_geo_data["PaleoCoLatitude"] = 90.
    good_geo_data["Major (SE)"] = 33./111.113
    good_geo_data["Site Latitude"] = good_geo_data["Latitude"]
    good_geo_data["Site Longitude"] = good_geo_data["Longitude"]
    good_geo_data["Name"] = good_geo_data["Seamount"]
    good_geo_data["Age"] = good_geo_data["Age (Ma)"]

    ep1_lat,ep1_lon,ep1_dis,ep2_lat,ep2_lon,ep2_dis,min_tot_error,lon_mesh,lat_mesh,ep1_chi2_surf,ep2_chi2_surf = fit_circ(em_poles,hi_poles,(inv_data[hs]["BendLat"],inv_data[hs]["BendLon"]),gridspace=0.2,pvalue=.05,lat_range=(-90,90),lon_range=(0,360),sml_circ=True,north_hemisphere=True,finish=True,l1_dis=False,ignore_bend=ignore_bend) #Follows method laid out in Gordon and Cox 1984 Appendix A
    print("-------------------------------------------------------------------------------------------------------")
    print(hs)
    print(min_tot_error/((len(em_poles)+len(hi_poles))-6))
    print(ep2_lat,ep2_lon,ep2_dis)
    print(inv_data[hs]["HILat"],inv_data[hs]["HILon"],inv_data[hs]["HIDis"])
    print(ep1_lat,ep1_lon,ep1_dis)
    print(inv_data[hs]["EMLat"],inv_data[hs]["EMLon"],inv_data[hs]["EMDis"])
    print("-------------------------------------------------------------------------------------------------------")
    inv_data[hs]["HILat"],inv_data[hs]["HILon"],inv_data[hs]["HIDis"] = ep2_lat,ep2_lon,ep2_dis
    inv_data[hs]["EMLat"],inv_data[hs]["EMLon"],inv_data[hs]["EMDis"] = ep1_lat,ep1_lon,ep1_dis

    if hs == "Hawaii": color = "tab:green"
    elif hs == "Rurutu": color = "tab:red"
    else: color = "tab:purple"
    if inv_data[hs]["HILat"]<0: inv_data[hs]["HILat"],inv_data[hs]["HILon"] = -inv_data[hs]["HILat"],(360+180+inv_data[hs]["HILon"])%360
    if inv_data[hs]["EMLat"]<0: inv_data[hs]["EMLat"],inv_data[hs]["EMLon"] = -inv_data[hs]["EMLat"],(360+180+inv_data[hs]["EMLon"])%360
    if hs == "Hawaii" and inv_data[hs]["EMLat"]>0: inv_data[hs]["EMLat"],inv_data[hs]["EMLon"] = -inv_data[hs]["EMLat"],(360+180+inv_data[hs]["EMLon"])%360
    m = psk.plot_pole(inv_data[hs]["HILon"],inv_data[hs]["HILat"],0.,0.01,0.01,marker='o',color=color,m=m,zorder=10,alpha=.3)
    m.contour(lon_mesh,lat_mesh,ep2_chi2_surf,levels = [min_tot_error+2*4], colors=[color], linewidths=[1], transform=ccrs.PlateCarree(), zorder=10)
    m = psk.plot_pole(inv_data[hs]["EMLon"],inv_data[hs]["EMLat"],0.,0.01,0.01,marker='s',color=color,m=m,zorder=10,alpha=.3)
    m.contour(lon_mesh,lat_mesh,ep1_chi2_surf,levels = [min_tot_error+2*4], colors=[color], linewidths=[1], transform=ccrs.PlateCarree(), zorder=10)
    np.savetxt("%s_HISubTrack.txt"%hs,ep2_chi2_surf)
    np.savetxt("%s_EMSubTrack.txt"%hs,ep1_chi2_surf)

hi_fmean = ipmag.fisher_mean(inv_data.T["HILon"].tolist(),inv_data.T["HILat"].tolist())
em_fmean = ipmag.fisher_mean(inv_data.T["EMLon"].tolist(),inv_data.T["EMLat"].tolist())
print(hi_fmean)
print(em_fmean)
m = psk.plot_pole(hi_fmean["dec"],hi_fmean["inc"],0.,.01,.01,marker='X',color=hi_color,m=m,zorder=10,alpha=.3)
m = psk.plot_pole(em_fmean["dec"],em_fmean["inc"],0.,.01,.01,marker='X',color=em_color,m=m,zorder=10,alpha=.3)

m.set_extent([220,300,-20,90],ccrs.PlateCarree())

fig.savefig("./poles_check.png",facecolor="white")

#---------------------------------------------------------------------------------------------------------

plt.show()
