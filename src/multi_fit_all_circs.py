"""

"""
import os,sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib as mpl
mpl.use("WXAgg")
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
from multi_circ_inv import fit_circ,pole_arc_fitfunc
from functools import reduce
from time import time
from scipy.interpolate import PchipInterpolator
import pmagpy.pmag as pmag
from mpl_toolkits.mplot3d import Axes3D

#TODO
#Search models where the bend is an adjustable parameter anywhere between 40-60 Ma

###############################################Fixed Values
hi_color = "#C4A58E"#plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
em_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
tr_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
undated_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
bend_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
mean_color = "k"#plt.rcParams['axes.prop_cycle'].by_key()['color'][2]

hi_a,hi_b,hi_phi = .1,.1,0.
geoid = Geodesic(6371.,0.)
land_resolution = "10m" #options: 10m, 50m, 110m
good_marker_size,bad_marker_size = 15,None

def find_nearest(array, value):
    array = np.nan_to_num(np.asarray(array),nan=1e9)
    idx = (np.abs(array - value)).argmin()
    return idx

def cmp_age_lat_data(idx1,idx2=None):
    if idx2==None: return 0
    datum1 = em_data.loc[idx1]
    datum2 = em_data.loc[idx2]
    if np.isnan(datum1["Age (Ma)"]) or np.isnan(datum2["Age (Ma)"]): return datum1["Latitude"]-datum2["Latitude"]
    else: return datum1["Age (Ma)"]-datum2["Age (Ma)"]

def subsets(lst, n):
    """Returns all subsets of lst of size exactly n in any order.
    lst is a Python list, and n is a non-negative integer.

    >>> three_subsets = subsets(list(range(5)), 3)
    >>> three_subsets.sort()
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

def polyfit_with_fixed_points(n, x, y, xf, yf) :
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]

###############################################Parse Flags
if "-h" in sys.argv: help(__name__)
if "-r" in sys.argv: resolution = float(sys.argv[sys.argv.index("-r")+1])
else: resolution = 1.0
if "-br" in sys.argv: bend_resolution = float(sys.argv[sys.argv.index("-br")+1])
else: bend_resolution = 1.0
if "-sml" in sys.argv: sml_circ = True
else: sml_circ = False
if "-l1" in sys.argv: l1_dis = True
else: l1_dis = False
if "-deg" in sys.argv: deg = int(sys.argv[sys.argv.index("-deg")+1])
else: deg = 1
if "-dis" in sys.argv: oth_hs = sys.argv[sys.argv.index("-dis")+1]
else: oth_hs = None
if "-dec" in sys.argv: decimation = float(sys.argv[sys.argv.index("-dec")+1])
else: decimation = 10
if "-herm" in sys.argv:
    HERM=True
    n_herm = int(sys.argv[sys.argv.index("-herm")+1])
else: HERM=False
if "-interp" in sys.argv: INTERP=True
else: INTERP=False
if "-inv" in sys.argv: inv_path=sys.argv[sys.argv.index("-inv")+1]
else: inv_path=False
if "-bag" in sys.argv: bend_age_gridsearch = True
else: bend_age_gridsearch = False
if "-smth" in sys.argv: smooth_model = True
else: smooth_model = False
if "-5D" in sys.argv: Dim5Inv = True
else: Dim5Inv = False
#if "-fit2" in sys.argv: fit2 = True
#else: fit2 = False
padding = .05
inv_data = {}
hs_names = ["Hawaii","Rurutu","Louisville HK19"]
geo_age_uncs = []

fig_poles = plt.figure(figsize=(9,20),dpi=200)
pole_proj = ccrs.Orthographic(central_longitude=265)
mp = fig_poles.add_subplot(111,projection=pole_proj)
mp.outline_patch.set_linewidth(0.5)
mp.coastlines(linewidth=2,color="k",resolution=land_resolution)

#pc = pycirc.PlateCircuit.read_excel("/home/kevin/Projects/MidCretaceousHSMotion/Data/Rotations/Global_Model.xlsx")
###############################Define Euler Pole
#Koivisto et al. 2014
#hi_rot = Rot(64.55,-67.95,34.29,47.91,0.)
#em_rot = Rot(51.05,-76.67,41.74,67.7,0.)
##Wessel and Kronke 2008
##hi_rot = Rot(63+1/60,-(66+41/60),34.6,47.91,0.)
##em_rot = Rot(47+18/60,-82.1,48.8,83.5,0.)
#em_rot = ~(hi_rot+em_rot.reverse_time())
#print(em_rot)
if "subareal" in sys.argv[1]: is_subair=True
else: is_subair=False

fig_res = plt.figure(figsize=(9,9),dpi=200)
fig_res.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.05,hspace=0.2)
res_pos = 321
bfig = plt.figure(figsize=(9,9),dpi=200)
#bfig.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.05,hspace=0.12)
bax_pos = 131
fig = plt.figure(figsize=(9,20*3/4),dpi=200)
fig.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.05,hspace=0.12)
ax_pos = 321
if inv_path:
    inv_data = pd.read_csv(inv_path,index_col=0,dtype={"BendLat":float,"BendLon":float,"EMDis":float,"EMLon":float,"EMLat":float,"EMErr":float,"EM_Start_Dis":float,"EM_start_azi":float,"EMpols":object,"EMsds":object,"HIDis":float,"HILon":float,"HILat":float,"HIErr":float,"HI_Start_Dis":float,"HI_start_azi":float,"HIpols":object,"HIsds":object}).T
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
    inv_data = inv_data.to_dict()

hs_chi2s = {hs_name:[[],[]] for hs_name in hs_names}

for hs_i,hs_name in enumerate(hs_names):
    print("-----------------------------------------------------%s"%hs_name)

    ###############################################Data Read
    data = pd.read_excel(sys.argv[1],hs_name)
    data["Maj"] = (33.0/111.113)*np.sqrt(2) #seamount 1d 1sigma from Chengzu
    data["Min"] = (33.0/111.113)*np.sqrt(2) #seamount 1d 1sigma from Chengzu
    data["Azi"] = 0.0 #Circular
    bend_color = "tab:green"

    if hs_name == "Hawaii":
        heb_age_start,heb_age_end,heb_age = 47.,49.,48.
#        hi_data = data[data["Age (Ma)"]<heb_age_start]
#        tr_data = data[(data["Age (Ma)"]>heb_age_start) & (data["Age (Ma)"]<heb_age_end)]
#        em_data = data[data["Age (Ma)"]>=heb_age_end]
        hi_data = data[data["Latitude"]<33.]
        tr_data = data[(data["Latitude"]>33.) & (data["Latitude"]<33.)]
        em_data = data[data["Latitude"]>33.]
        window = [150.,210.,13.,55.]
        bend_window=[31.,35.,170.,174.]
        if HERM: age_min,age_max,age_step = 0.,75.,5.
        else: age_min,age_max,age_step = 0.,80.,5.
        title = "Emperor Chain"
        em_a,em_b,em_phi = 7.0,1.0,40. # Hawaii
        sml_circ_hi = True
    elif hs_name == "Louisville" or hs_name == "Louisville HK19":
        heb_age_start,heb_age_end,heb_age = 47.,49.,48.
#        hi_data = data[data["Age (Ma)"]<heb_age_start]
#        tr_data = data[(data["Age (Ma)"]>heb_age_start) & (data["Age (Ma)"]<heb_age_end)]
#        em_data = data[data["Age (Ma)"]>=heb_age_end]
        hi_data = data[data["Longitude"]>-169.]
        tr_data = data[(data["Longitude"]>-169.) & (data["Longitude"]<=-169.)]
        em_data = data[data["Longitude"]<-169.]
        window = [175.,225.,-57.,-20.]
#        bend_window=[-38.5,-35.5,189.,193.]
        bend_window=[-39.,-35.,189.,193.]
        if HERM: age_min,age_max,age_step = 0.,75.,5.
        else: age_min,age_max,age_step = 0.,80.,5.
#        if "Heaton" in hs_name: title = "Louisville Chain (Adjusted for trapped $^{40}$Ar)"
#        else: title = "Louisville Chain"
        title = "Louisville Chain"
        em_a,em_b,em_phi = 6.5,0.50,-32. # Louisville
        sml_circ_hi = True
    elif hs_name == "Rurutu":
        heb_age_start,heb_age_end,heb_age = 47.,49.,48.
        cut_age = 20.
        hi_data = data[data["Age (Ma)"]<cut_age]
        tr_data = data[(data["Age (Ma)"]>cut_age) & (data["Age (Ma)"]<cut_age)]
        em_data = data[data["Age (Ma)"]>=cut_age]
        konrad_ru_bend = [-8.56,178.48]
        window = [160.,215.,-30.,10.]
#        bend_window=[-13.,0.,176.,181.]
        bend_window=[-13.,5.,172.,190.]
        if deg==2: age_min,age_max,age_step = 0.,70.,5.
        else: age_min,age_max,age_step = 0.,70.,5.
        title = "Rurutu Chain"
        em_a,em_b,em_phi = 6.5,0.50,-32. # Rurutu
        sml_circ_hi = True
    else: raise IOError("No HS Track named %s known must edit script"%hs_name)

#    em_data = em_data.reindex(sorted(em_data.index,key=cmp_age_lat_data))

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

    ###############################################Fit Circles
#    hi_lat,hi_lon,hi_rot,hi_dis,min_bend_err,hi_corrected_error_points = fast_fit_circ(hi_poles,pvalue=.05,sml_circ=sml_circ,north_hemisphere=True,final_resolution=resolution,l1_dis=l1_dis)
#    print(hi_lat,hi_lon,hi_rot,hi_dis,min_bend_err/(len(hi_poles)))
    if not INTERP:# and hs_name!="Louisville HK19":
        if not inv_path:
            if not Dim5Inv:
                (bend_lat,bend_lon),(em_lat,em_lon,em_dis),(hi_lat,hi_lon,hi_dis),min_bend_err,ppoles1,ppoles2,(lon_mesh,lat_mesh,chi2_surf) = fit_circ(em_poles,hi_poles,sml_circ1=sml_circ,sml_circ2=sml_circ_hi,north_hemisphere=True,gridspace=resolution,l1_dis=l1_dis,finish=True,bend_window=bend_window,bend_gridspace=bend_resolution)
                if hs_name=="Rurutu":
                    _,(em_lat,em_lon,em_dis),(hi_lat,hi_lon,hi_dis),min_bend_err,ppoles1,ppoles2,_ = fit_circ(em_poles,hi_poles,sml_circ1=sml_circ,sml_circ2=sml_circ_hi,north_hemisphere=True,gridspace=resolution,l1_dis=l1_dis,finish=True,bend_window=[konrad_ru_bend[0],konrad_ru_bend[0]+bend_resolution,konrad_ru_bend[1],konrad_ru_bend[1]+bend_resolution],bend_gridspace=bend_resolution)
                if is_subair: chi2_surf_path = "Bend_Inversion_%s_%.2f_%.2f_chi2surf_subair.txt"%(hs_name,resolution,bend_resolution)
                else: chi2_surf_path = "Bend_Inversion_%s_%.2f_%.2f_chi2surf.txt"%(hs_name,resolution,bend_resolution)
                np.savetxt(chi2_surf_path,chi2_surf)
                inv_data[hs_name] = {"HILat":hi_lat,"HILon":hi_lon,"HIDis":hi_dis,"HIErr":min_bend_err,"EMLat":em_lat,"EMLon":em_lon,"EMDis":em_dis,"EMErr":min_bend_err,"BendLat":bend_lat,"BendLon":bend_lon,"Chi2SurfPath":chi2_surf_path}
        else:
            bend_lat,bend_lon,em_lat,em_lon,em_dis,hi_lat,hi_lon,hi_dis,min_bend_err = pd.DataFrame(inv_data)[hs_name][["BendLat","BendLon","EMLat","EMLon","EMDis","HILat","HILon","HIDis","EMErr"]]
            chi2_surf_path = inv_data[hs_name]["Chi2SurfPath"]
            chi2_surf = np.loadtxt(chi2_surf_path)
            blat_range = bend_window[:2]
            blat_range[1] += bend_resolution
            blon_range = bend_window[2:]
            blon_range[1] += bend_resolution
            X = np.arange(*blon_range,bend_resolution)
            Y = np.arange(*blat_range,bend_resolution)
            lon_mesh, lat_mesh = np.meshgrid(X, Y)
        hi_err = pole_arc_fitfunc((hi_lat,hi_lon),hi_poles,(bend_lat,bend_lon),sml_circ,l1_dis)
        em_err = pole_arc_fitfunc((em_lat,em_lon),em_poles,(bend_lat,bend_lon),sml_circ,l1_dis)
        print(bend_lat,bend_lon)
        print("Hawaiian Result:",hi_lat,hi_lon,hi_dis)
        print("Emperor Result:",em_lat,em_lon,em_dis)
        print("Misfit",hi_err,em_err,min_bend_err)
        print("DOFs",(len(hi_poles)-3),len(em_poles)-3,len(hi_poles)+len(em_poles)-6)
        print("Geo1sigma",33.*np.sqrt(hi_err/(len(hi_poles)-3)),33.*np.sqrt(em_err/(len(em_poles)-3)),33.*np.sqrt(min_bend_err/(len(hi_poles)+len(em_poles)-6)))
        if em_lon<180: em_lat,em_lon,em_dis=-em_lat,em_lon+180,180-em_dis
        if hi_lon<180: hi_lat,hi_lon,hi_dis=-hi_lat,hi_lon+180,180-hi_dis
        if hs_name=="Rurutu" and hi_lat<0: hi_lat,hi_lon,hi_dis=-hi_lat,hi_lon+180,180-hi_dis
#        max_data = pymax.parse_apwp_df_or_list_to_max(good_geo_data)
#        (geo_lat,geo_lon,geo_mag,geo_maj_se,geo_min_se,geo_phi),geo_chisq,geo_dof = pymax.max_likelihood_pole(max_data)
#        if geo_lon<180: geo_lat,geo_lon=-geo_lat,geo_lon+180
#        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][hs_i]
#        print("Going into Pole Plot: ", geo_lon,geo_lat,geo_phi,np.sqrt(geo_chisq/geo_dof)*geo_maj_se,np.sqrt(geo_chisq/geo_dof)*geo_min_se)
#        psk.plot_pole(geo_lon,geo_lat,geo_phi,np.sqrt(geo_chisq/geo_dof)*geo_maj_se,np.sqrt(geo_chisq/geo_dof)*geo_min_se,color=color,alpha=.5,m=mp,label=title,zorder=3)
#        f_mean = pmag.fisher_mean(good_geo_data[["Longitude","Latitude"]].values.tolist())
#        mp.plot([f_mean["dec"],geo_lon],[f_mean["inc"],geo_lat],color=color,zorder=2,transform=ccrs.Geodetic())
#        mp.plot([f_mean["dec"],geo_lon],[f_mean["inc"],geo_lat],color="k",zorder=1,linewidth=1.5,transform=ccrs.Geodetic())
##        mp.scatter(em_lon,em_lat,color=color,label=title,marker="o",transform=ccrs.Geodetic(),zorder=2)
##        poly = plot_error_from_points(em_corrected_error_points,pole_proj,color=color,alpha=.5,zorder=1)
##        mp.add_patch(poly)
#        print((geo_lat,geo_lon,geo_mag,geo_maj_se,geo_min_se,geo_phi),geo_chisq,geo_dof,geo_chisq/geo_dof)
#        good_geo_data["GCRes"] = 0.
#        for i,datum in good_geo_data.iterrows():
#            geodict = geoid.Inverse(datum["Latitude"],datum["Longitude"],em_lat,em_lon)
#            mgeodict = geoid.ArcDirect(em_lat,em_lon,geodict["azi2"],90.)
#            good_geo_data.at[i,"GCRes"] = (geodict["a12"]-mgeodict["a12"])
#        max_gc_res = max(abs(good_geo_data["GCRes"]))
#        max_idx = np.argmax(abs(good_geo_data["GCRes"]))
#        min_gc_res = min(abs(good_geo_data["GCRes"]))
#        min_idx = np.argmin(abs(good_geo_data["GCRes"]))
#        med_gc_res = np.median(abs(good_geo_data["GCRes"]))
#        med_idx = abs(good_geo_data["GCRes"]).to_list().index(np.percentile(abs(good_geo_data["GCRes"]),50,interpolation='nearest'))
#        print("Minimum Great Circle Residual: ",good_geo_data["Seamount"].loc[min_idx]," = ",111.113*min_gc_res," km")
#        print("Median Great Circle Residual: ",good_geo_data["Seamount"].iloc[med_idx]," = ",111.113*med_gc_res," km")
#        print("Maximum Great Circle Residual: ",good_geo_data["Seamount"].loc[max_idx]," = ",111.113*max_gc_res," km")

    #Visualize Chain with Fits
    proj = ccrs.Mercator(central_longitude=180.)
    m = fig.add_subplot(ax_pos,projection=proj)
    m.set_xticks(np.arange(0, 365, 10.), crs=ccrs.PlateCarree())
    m.set_yticks(np.arange(-80, 85, 5.), crs=ccrs.PlateCarree())
    m.tick_params(grid_linewidth=1.0,grid_linestyle=":",color="grey",labelsize=8,tickdir="in",left=True,top=True)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    m.xaxis.set_major_formatter(lon_formatter)
    m.yaxis.set_major_formatter(lat_formatter)
    m.outline_patch.set_linewidth(0.5)
    m.coastlines(linewidth=2,color="k",resolution=land_resolution)
    m.annotate(chr(64+ax_pos%10)+")",xy=(1-0.04,0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="bottom",ha="right")

    all_lons,all_lats,all_grav = pg.get_sandwell(window,decimation,sandwell_files_path="../PySkew/raw_data/gravity/Sandwell/*.tiff")

    print("Plotting Gravity")
    start_time = time()
    print(all_lons.shape,all_lats.shape,all_grav.shape)
#    potental cmaps: cividis
    fcm = m.contourf(all_lons, all_lats, all_grav, cmap="Blues_r", alpha=.75, transform=ccrs.PlateCarree(), zorder=0)
    print("Runtime: ",time()-start_time)

    m.set_extent(window, ccrs.PlateCarree())

    m.contour(lon_mesh,lat_mesh,chi2_surf,levels = [min_bend_err+2*4], colors=[bend_color], linewidths=[1], transform=ccrs.PlateCarree(), zorder=10)

    for i,datum in hi_data.iterrows():
        if datum["Quality"]=="g": marker,marker_size = "o",good_marker_size
        else: marker,marker_size = "X",bad_marker_size
        if np.isnan(datum["Age (Ma)"]): color = undated_color
        else: continue
        m = psk.plot_pole(datum["Longitude"],datum["Latitude"],datum["Azi"],2*datum["Maj"],2*datum["Min"],edgecolors="k",facecolors=color,color=color,marker=marker,m=m,zorder=4,s=marker_size)
#        m = psk.plot_pole(datum[1],datum[0],datum[-1],2*datum[2],2*datum[3],edgecolors="k",facecolors=hi_color,color=hi_color,marker="o",m=m,zorder=4,s=good_marker_size)
    if INTERP:
        m.plot(good_hi_data["Longitude"],good_hi_data["Latitude"],transform=ccrs.Geodetic(),color=hi_color,zorder=3,linewidth=2)
        m.plot(good_hi_data["Longitude"],good_hi_data["Latitude"],transform=ccrs.Geodetic(),color="k",zorder=2,linewidth=3)
    else:
        m = psk.plot_small_circle(hi_lon,hi_lat,hi_dis,m=m,color="k",linewidth=3,zorder=2)
        m = psk.plot_small_circle(hi_lon,hi_lat,hi_dis,m=m,color=hi_color,linewidth=2,zorder=3)

    for i,datum in em_data.iterrows():
        if datum["Quality"]=="g": marker,marker_size = "o",good_marker_size
        else: marker,marker_size = "X",bad_marker_size
        if np.isnan(datum["Age (Ma)"]): color = undated_color
        else: continue
        m = psk.plot_pole(datum["Longitude"],datum["Latitude"],datum["Azi"],2*datum["Maj"],2*datum["Min"],edgecolors="k",facecolors=color,color=color,marker=marker,m=m,zorder=4,s=marker_size)
#        m = psk.plot_pole(datum[1],datum[0],datum[-1],2*datum[2],2*datum[3],edgecolors="k",facecolors=em_color,color=em_color,marker="o",m=m,zorder=4,s=good_marker_size)
    if INTERP:
        m.plot(good_em_data["Longitude"],good_em_data["Latitude"],transform=ccrs.Geodetic(),color=em_color,zorder=3,linewidth=2)
        m.plot(good_em_data["Longitude"],good_em_data["Latitude"],transform=ccrs.Geodetic(),color="k",zorder=2,linewidth=3)
    else:
        m = psk.plot_small_circle(em_lon,em_lat,em_dis,m=m,color="k",linewidth=3,zorder=2)
        m = psk.plot_small_circle(em_lon,em_lat,em_dis,m=m,color=em_color,linewidth=2,zorder=3)

    m = psk.plot_pole(bend_lon,bend_lat,0.,.1,.1,edgecolors="k",facecolors=bend_color,color=bend_color,marker="s",m=m,zorder=100000,s=20,alpha_all=True,alpha=.8)
    if hs_name=="Rurutu": m = psk.plot_pole(konrad_ru_bend[1],konrad_ru_bend[0],0.,.1,.1,edgecolors="k",facecolors="tab:olive",color="tab:olive",marker="s",m=m,zorder=100000,s=20,alpha_all=True,alpha=.8)

    #Plot Bend Figures
#    if hs_name!="Louisville HK19":
    bproj = ccrs.Mercator(central_longitude=180.)
    bm = bfig.add_subplot(bax_pos,projection=bproj)
    bm.set_xticks(np.arange(0, 362, 2.), crs=ccrs.PlateCarree())
    bm.set_yticks(np.arange(-80, 82, 2.), crs=ccrs.PlateCarree())
    bm.tick_params(grid_linewidth=1.0,grid_linestyle=":",color="grey",labelsize=8,tickdir="in",left=True,top=True)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    bm.xaxis.set_major_formatter(lon_formatter)
    bm.yaxis.set_major_formatter(lat_formatter)
    bm.outline_patch.set_linewidth(0.5)
    bm.coastlines(linewidth=2,color="k",resolution=land_resolution)
    bm.annotate(chr(64+bax_pos%10)+")",xy=(1-0.04,0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="bottom",ha="right")

    bend_extent = [bend_window[2],bend_window[3],bend_window[0],bend_window[1]]
    all_lons,all_lats,all_grav = pg.get_sandwell(bend_extent,1,sandwell_files_path="../PySkew/raw_data/gravity/Sandwell/*.tiff")

    print("Plotting Gravity")
    start_time = time()
    print(all_lons.shape,all_lats.shape,all_grav.shape)
#    potental cmaps: cividis
    fcm = bm.contourf(all_lons, all_lats, all_grav, cmap="Greys_r", alpha=.75, transform=ccrs.PlateCarree(), zorder=0)
    print("Runtime: ",time()-start_time)

    print(bend_extent)
    if hs_name=="Rurutu": bm.set_extent([176.,180.,-9.25,-4.25], ccrs.PlateCarree())
    else: bm.set_extent(bend_extent, ccrs.PlateCarree())

    print(lon_mesh.shape,lat_mesh.shape,chi2_surf.shape)
    cm = bm.contour(lon_mesh,lat_mesh,chi2_surf,levels = [min_bend_err+2*4], colors=[bend_color], linewidths=[1], transform=ccrs.PlateCarree(), zorder=10)
    cm = bm.contour(lon_mesh,lat_mesh,np.sqrt(chi2_surf-min_bend_err)/np.sqrt(2), levels = np.arange(1,6,1), colors=["k",bend_color,"k","k","k"], linewidths=[.3,1,.3,.3,.3], transform=ccrs.PlateCarree(), zorder=2)
    fcm = bm.contourf(lon_mesh,lat_mesh,np.sqrt(chi2_surf-min_bend_err)/np.sqrt(2), levels = np.arange(0,6,1), vmax=5, vmin=0., cmap="viridis", transform=ccrs.PlateCarree(), zorder=1, alpha=.3, extend="max")
    if hs_name==hs_names[-1]:
        cbar = bfig.colorbar(fcm,ticks=np.arange(1,6,1),extend="max",format="%.0f$\sigma$",shrink=.45)
#    bm.clabel(cm, inline=True, fontsize=6, fmt=r"8$\sigma$")
#    cbar = bfig.get_axis().colorbar(fcm)#,ticks=np.arange(0.0,vmax+step,step))

    for i,datum in hi_data.iterrows():
        if datum["Quality"]=="g": marker,marker_size = "o",good_marker_size
        else: marker,marker_size = "X",bad_marker_size
        if np.isnan(datum["Age (Ma)"]): color = undated_color
        else: continue
        bm = psk.plot_pole(datum["Longitude"],datum["Latitude"],datum["Azi"],2*datum["Maj"],2*datum["Min"],edgecolors="k",facecolors=color,color=color,marker=marker,m=bm,zorder=4,s=marker_size)
#        m = psk.plot_pole(datum[1],datum[0],datum[-1],2*datum[2],2*datum[3],edgecolors="k",facecolors=hi_color,color=hi_color,marker="o",m=m,zorder=4,s=good_marker_size)
    if INTERP:
        bm.plot(good_hi_data["Longitude"],good_hi_data["Latitude"],transform=ccrs.Geodetic(),color=hi_color,zorder=3,linewidth=2)
        bm.plot(good_hi_data["Longitude"],good_hi_data["Latitude"],transform=ccrs.Geodetic(),color="k",zorder=2,linewidth=3)
    else:
        print("HICIRC",hi_lon,hi_lat,hi_dis,geoid.Inverse(hi_lat,hi_lon,bend_lat,bend_lon)["a12"])
        bm = psk.plot_small_circle(hi_lon,hi_lat,hi_dis,m=bm,color="k",linewidth=3,zorder=2,geoid=geoid)
        bm = psk.plot_small_circle(hi_lon,hi_lat,hi_dis,m=bm,color=hi_color,linewidth=2,zorder=3,geoid=geoid)

    for i,datum in em_data.iterrows():
        if datum["Quality"]=="g": marker,marker_size = "o",good_marker_size
        else: marker,marker_size = "X",bad_marker_size
        if np.isnan(datum["Age (Ma)"]): color = undated_color
        else: continue
        bm = psk.plot_pole(datum["Longitude"],datum["Latitude"],datum["Azi"],2*datum["Maj"],2*datum["Min"],edgecolors="k",facecolors=color,color=color,marker=marker,m=bm,zorder=4,s=marker_size)
#        m = psk.plot_pole(datum[1],datum[0],datum[-1],2*datum[2],2*datum[3],edgecolors="k",facecolors=em_color,color=em_color,marker="o",m=m,zorder=4,s=good_marker_size)
    if INTERP:
        bm.plot(good_em_data["Longitude"],good_em_data["Latitude"],transform=ccrs.Geodetic(),color=em_color,zorder=3,linewidth=2)
        bm.plot(good_em_data["Longitude"],good_em_data["Latitude"],transform=ccrs.Geodetic(),color="k",zorder=2,linewidth=3)
    else:
        print("EMCIRC",em_lon,em_lat,em_dis,geoid.Inverse(em_lat,em_lon,bend_lat,bend_lon)["a12"])
        bm = psk.plot_small_circle(em_lon,em_lat,em_dis,m=bm,color="k",linewidth=3,zorder=2,geoid=geoid)
        bm = psk.plot_small_circle(em_lon,em_lat,em_dis,m=bm,color=em_color,linewidth=2,zorder=3,geoid=geoid)

    print("BEND",bend_lat,bend_lon)
    bm = psk.plot_pole(bend_lon,bend_lat,0.,.1,.1,edgecolors="k",facecolors=bend_color,color=bend_color,marker="s",m=bm,zorder=4,s=good_marker_size)
    if hs_name=="Rurutu":
        print("KONRAD BEND",*konrad_ru_bend)
        bm = psk.plot_pole(konrad_ru_bend[1],konrad_ru_bend[0],0.,.1,.1,edgecolors="k",facecolors="tab:olive",color="tab:olive",marker="s",m=bm,zorder=100000,s=20,alpha_all=True,alpha=.8)

    if "Louisville" in hs_name: title = "Louisville"
    else: title = hs_name
    bm.set_title(title)

    ###############################################Remove Undated Seamounts
#    hi_data = hi_data[hi_data["Age (Ma)"].notnull()]
    dated_hi_data = hi_data[hi_data["Age (Ma)"].notnull()]
    dated_em_data = em_data[em_data["Age (Ma)"].notnull()]
    ax_pos += 1

    if hs_name=="Rurutu": bend_lat,bend_lon = konrad_ru_bend

    ###############################################Visualize Dispursion of Age With Distance along circles

    geodict = geoid.Inverse(dated_hi_data.iloc[0]["Latitude"],dated_hi_data.iloc[0]["Longitude"],hi_lat,hi_lon)
    hi_start_azi = geodict["azi2"]
    for i,datum in dated_hi_data.iterrows():
        geodict = geoid.Inverse(datum["Latitude"],datum["Longitude"],hi_lat,hi_lon)
        dis = (geodict["azi2"]-hi_start_azi)*np.sin(np.deg2rad(hi_dis))
        dated_hi_data.at[i,"Dis"] = dis
    dated_hi_data.sort_values("Dis",inplace=True)

    subair_dated_hi_data = dated_hi_data[dated_hi_data["Age (Ma)"]<9]
    good_dated_hi_data = dated_hi_data[dated_hi_data["Quality"]=="g"]
    bad_dated_hi_data = dated_hi_data[~(dated_hi_data["Quality"]=="g")]

    geodict = geoid.Inverse(bend_lat,bend_lon,hi_lat,hi_lon)
    em_start_dis = (geodict["azi2"]-hi_start_azi)*np.sin(np.deg2rad(hi_dis)) #bend distance

#    print("Extrapolation Amount:",dated_hi_data.iloc[-1]["Dis"],dated_hi_data.iloc[-1]["Age (Ma)"],dated_em_data.iloc[0]["Age (Ma)"],hi_pols[0])
    if INTERP:
        idx_start_dated_data = dated_em_data.iloc[0].name
        for i,datum in dated_em_data.iterrows():
            dis = 0
            prev_datum = dated_em_data.iloc[0]
            for j,next_datum in good_em_data.loc[idx_start_dated_data:].iterrows():
                dis += geoid.Inverse(prev_datum["Latitude"],prev_datum["Longitude"],next_datum["Latitude"],next_datum["Longitude"])["a12"]
                if next_datum["Age (Ma)"]==datum["Age (Ma)"]: break
                prev_datum = next_datum
            if i==idx_start_dated_data:
                start_dis = dis
                dated_em_data.at[i,"Dis"] = 0.
            else: dated_em_data.at[i,"Dis"] = dis - start_dis
    else:
#        geodict = geoid.Inverse(dated_em_data.iloc[0]["Latitude"],dated_em_data.iloc[0]["Longitude"],hi_lat,hi_lon) #Starting Location Given First Seamount Used Before Bend Inv.
        geodict = geoid.Inverse(bend_lat,bend_lon,em_lat,em_lon)
        em_start_azi = geodict["azi2"]
        for i,datum in dated_em_data.iterrows():
            geodict = geoid.Inverse(datum["Latitude"],datum["Longitude"],em_lat,em_lon)
            dis = (geodict["azi2"]-em_start_azi)*np.sin(np.deg2rad(em_dis))+em_start_dis#+(dated_hi_data.iloc[-1]["Age (Ma)"]-dated_em_data.iloc[0]["Age (Ma)"])*hi_pols[0])
            dated_em_data.at[i,"Dis"] = dis
    dated_em_data.sort_values("Dis",inplace=True)

    good_dated_em_data = dated_em_data[dated_em_data["Quality"]=="g"]
    bad_dated_em_data = dated_em_data[~(dated_em_data["Quality"]=="g")]

    ####################################Bend Distance Loc for EM start location and Bend Gridsearch
    if bend_age_gridsearch:
#        if "subareal" in sys.argv[1]: hs_bend_geo_unc = {"Hawaii":[0.744812621708064,0.4715447998943002],"Louisville HK19":[2.46268789136353,1.7545323634341292],"Rurutu":[0.,0.]}
#        else: hs_bend_geo_unc = {"Hawaii":[0.7229981242497896,0.5642321740722509],"Louisville HK19":[2.46268789136353,1.7545323634341292],"Rurutu":[0.,0.]}
        if "subareal" in sys.argv[1]: hs_bend_geo_unc = {"Hawaii":[0.744812621708064/2.,0.4715447998943002/2.],"Louisville HK19":[2.641448553960279/2.,1.9796563605107313/2.],"Rurutu":[0.,0.]}
        else: hs_bend_geo_unc = {"Hawaii":[0.7229981242497896/2.,0.5642321740722509/2.],"Louisville HK19":[2.641448553960279/2.,1.9796563605107313/2.],"Rurutu":[0.,0.]}
        min_chi2,hi_chi2s,em_chi2s,bend_ages = np.inf,[],[],np.arange(30.,60.1,.01)
        for ba in bend_ages:
            #Fitting Hawaiian model fixed to ba,em_start_dis
            hi_pols = polyfit_with_fixed_points(deg+1, good_dated_hi_data["Dis"].values, good_dated_hi_data["Age (Ma)"].values, [em_start_dis], [ba])[::-1]

            #Emperor model if fixed point is also required to be smooth
            if smooth_model:
                bend_slope = sum([(len(hi_pols)-e-1)*hi_pols[e]*(em_start_dis**(len(hi_pols)-e-2)) for e in range(len(hi_pols))])
                em_pols = [bend_slope,ba-em_start_dis/bend_slope]

            #Fitting Emperor model fixed to ba,em_start_dis
            else:
                em_pols = polyfit_with_fixed_points(deg, good_dated_em_data["Dis"].values, good_dated_em_data["Age (Ma)"].values, [em_start_dis], [ba])[::-1]

            yp = np.polyval(hi_pols,good_dated_hi_data["Dis"])
#            if hs_name!="Rurutu":
##                print("Hawaiian Geo Unc, Slowness, and mapped Geo Unc: ", hs_bend_geo_unc[hs_name][0],sum([(len(hi_pols)-e-1)*hi_pols[e]*(em_start_dis**(len(hi_pols)-e-2)) for e in range(len(hi_pols))]),(hs_bend_geo_unc[hs_name][0]*sum([(len(hi_pols)-e-1)*hi_pols[e]*(em_start_dis**(len(hi_pols)-e-2)) for e in range(len(hi_pols))])))
#                if len(hi_pols)==2: hi_chi2 = sum(((good_dated_hi_data["Age (Ma)"]-yp)/np.sqrt(1+hs_bend_geo_unc[hs_name][0]*hi_pols[0]**2))**2)
#                else: hi_chi2 = sum(((good_dated_hi_data["Age (Ma)"]-yp)/(np.sqrt(1+(hs_bend_geo_unc[hs_name][0]*sum([(len(hi_pols)-e-1)*hi_pols[e]*(em_start_dis**(len(hi_pols)-e-2)) for e in range(len(hi_pols))]))**2)))**2)
#            else: hi_chi2 = sum(((good_dated_hi_data["Age (Ma)"]-yp))**2)
            hi_chi2 = sum(((good_dated_hi_data["Age (Ma)"]-yp))**2)
            yp = np.polyval(em_pols,good_dated_em_data["Dis"])
#            if hs_name!="Rurutu":
##                print("Emperor Geo Unc, Slowness, and mapped Geo Unc: ",hs_bend_geo_unc[hs_name][1],sum([(len(em_pols)-e-1)*em_pols[e]*(em_start_dis**(len(em_pols)-e-2)) for e in range(len(em_pols))]),hs_bend_geo_unc[hs_name][1]*sum([(len(em_pols)-e-1)*em_pols[e]*(em_start_dis**(len(em_pols)-e-2)) for e in range(len(em_pols))]))
#                if len(em_pols)==2: em_chi2 = sum(((good_dated_em_data["Age (Ma)"]-yp)/np.sqrt(1+hs_bend_geo_unc[hs_name][1]*em_pols[0]**2))**2)
#                else: em_chi2 = sum(((good_dated_em_data["Age (Ma)"]-yp)/(np.sqrt(1+(hs_bend_geo_unc[hs_name][1]*sum([(len(em_pols)-e-1)*em_pols[e]*(em_start_dis**(len(em_pols)-e-2)) for e in range(len(em_pols))]))**2)))**2)
#            else: em_chi2 = sum(((good_dated_em_data["Age (Ma)"]-yp))**2)
            em_chi2 = sum(((good_dated_em_data["Age (Ma)"]-yp))**2)
            hi_chi2s.append(hi_chi2);em_chi2s.append(em_chi2)
            hs_chi2s[hs_name][0].append(hi_chi2);hs_chi2s[hs_name][1].append(em_chi2)
            if (hi_chi2+em_chi2)<min_chi2:
                min_chi2 = (hi_chi2+em_chi2)
                min_hi_pols = hi_pols
                min_em_pols = em_pols
                min_bend_age = ba

        hi_pols = min_hi_pols
        hi_min_age_idx = np.argmin(hi_chi2s)
        hi_min_age = bend_ages[hi_min_age_idx]
        hi_min_chi2 = min(hi_chi2s)
        hi_idx = np.argwhere(np.diff(np.sign(np.array(hi_chi2s)-(hi_min_chi2+4)))).flatten()
        hi_bend_slowness = sum([hi_pols[e]*(em_start_dis**(len(hi_pols)-e-2)) for e in range(len(hi_pols))])
        hi_geo_unc = (hs_bend_geo_unc[hs_name][0]*hi_bend_slowness)
        hi_cor_unc = np.sqrt(((hi_min_age-bend_ages[hi_idx])/2)**2 + hi_geo_unc**2)
        print("Hawaiian Subtrack")
        print("\tBend HS Rate and Geographic 1 sigma",111.113/hi_bend_slowness, 111.113*hs_bend_geo_unc[hs_name][0],hi_geo_unc)
        print("\tMaximum Likelihood Bend Age and 2sigma: ",hi_min_age,hi_min_age-bend_ages[hi_idx],2*hi_cor_unc)
        print("\tMisfit of Bend Age and 2sigma",hi_min_chi2,np.array(hi_chi2s)[hi_idx])

        em_pols = min_em_pols
        em_min_age_idx = np.argmin(em_chi2s)
        em_min_age = bend_ages[em_min_age_idx]
        em_min_chi2 = min(em_chi2s)
        em_idx = np.argwhere(np.diff(np.sign(np.array(em_chi2s)-(em_min_chi2+4)))).flatten()
        em_bend_slowness = sum([em_pols[e]*(em_start_dis**(len(em_pols)-e-2)) for e in range(len(em_pols))])
        em_geo_unc = (hs_bend_geo_unc[hs_name][1]*em_bend_slowness)
        em_cor_unc = np.sqrt(((em_min_age-bend_ages[em_idx])/2)**2 + em_geo_unc**2)
        print("Emperor Subtrack")
        print("\tBend HS Rate and Geographic 1 sigma",111.113/em_bend_slowness, 111.113*hs_bend_geo_unc[hs_name][1],em_geo_unc)
        print("\tMaximum Likelihood Bend Age and 2sigma: ",em_min_age,em_min_age-bend_ages[em_idx],2*em_cor_unc)
        print("\tMisfit of Bend Age and 2sigma",em_min_chi2,np.array(em_chi2s)[em_idx])

        idx = np.argwhere(np.diff(np.sign(np.array(hi_chi2s)+np.array(em_chi2s)-(min_chi2+4)))).flatten()
        if hs_name!="Rurutu":
            geo_age_uncs.append(1/(1/(hi_geo_unc**2) + 1/(em_geo_unc**2)))
            com_unc = np.sqrt(((min_bend_age-bend_ages[idx])/2)**2+1/(1/(hi_geo_unc**2) + 1/(em_geo_unc**2)))
        else: com_unc = np.array([0.,0.])
        print("Combined")
        print("\tMaximum Likelihood Bend Age and 2sigma: ",min_bend_age,(hi_min_age+em_min_age)/2,min_bend_age-bend_ages[idx], 2*com_unc)
        print("\tMisfit of Bend Age and 2sigma",min_chi2,(np.array(hi_chi2s)+np.array(em_chi2s))[idx])

        if hs_name==hs_names[0]:
            ba_fig = plt.figure(figsize=(9,9),dpi=200)
            ba_fig.subplots_adjust(left=.075,bottom=.075,right=1-.075,top=1-.075,wspace=0.1,hspace=0.1)
            ba_ax_pos = 221
        else: ba_ax_pos += 1

        ba_ax = ba_fig.add_subplot(ba_ax_pos)

        bend_color = "tab:green"
        if hs_name=="Rurutu": bend_color = "tab:olive"

        ba_ax.plot(bend_ages,hi_chi2s,color=hi_color,label="Hawaiian Subtrack Misfit")
        ba_ax.plot(bend_ages,em_chi2s,color=em_color,label="Emperor Subtrack Misfit")
        ba_ax.plot(bend_ages,np.array(hi_chi2s)+np.array(em_chi2s),color="tab:red",label="Total Track Misfit")
        ba_ax.scatter([min_bend_age],[min_chi2],color=bend_color,label="Minimum Track Misfit",marker="s",edgecolors="k",zorder=10000)
#        ba_ax.axhline(min_chi2+4,color=bend_color,label=r"Uncertainty on Bend Age 2$\sigma$")
#        ba_ax.scatter(bend_ages[idx],(np.array(hi_chi2s)+np.array(em_chi2s))[idx],color=bend_color,marker="+")
        if hs_name!="Rurutu":
            ba_ax.axvline(bend_ages[idx][0],color=bend_color,linestyle="--")
            if len(idx)>1:
                ba_ax.axvline(bend_ages[idx][1],color=bend_color,linestyle="--")
                ba_ax.axvspan(*bend_ages[idx],color=bend_color,alpha=.4)
                #Show geographic uncertainty as well
                cmap = mpl.cm.get_cmap('tab20')
                extended_bend_unc_color = cmap(0.25)
                ba_ax.axvline(min_bend_age-2*com_unc[0],color=extended_bend_unc_color,linestyle="--")
                ba_ax.axvline(min_bend_age+2*com_unc[1],color=extended_bend_unc_color,linestyle="--")
                ba_ax.axvspan(min_bend_age-2*com_unc[0],bend_ages[idx][0],color=extended_bend_unc_color,alpha=.4)
                ba_ax.axvspan(bend_ages[idx][1],min_bend_age+2*com_unc[1],color=extended_bend_unc_color,alpha=.4)
#        ba_ax.legend(fontsize=10)
        if "Louisville" in hs_name: title = "Louisville"
        else: title = hs_name
        ba_ax.set_title("%s Bend Age Misfit"%title,fontsize=10)
#        ba_ax.annotate("%.1f + %.2f"%(),xy=(0.5,0.2),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="bottom",ha="right")
        ba_ax.set_xticks(np.arange(45.,56.,1.))
        ba_ax.set_xlim(45.,55.)
        if hs_name==hs_names[0]: ba_ax.axes.xaxis.set_ticklabels([])
        else: ba_ax.set_xlabel("Age (Ma)")
        if ba_ax_pos%2!=0: ba_ax.set_ylabel(r"$\chi^2$")
        ba_ax.set_ylim(0.,min_chi2+20)
        ba_ax.annotate(chr(64+ba_ax_pos%10)+")",xy=(1-0.04,0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="bottom",ha="right")
        if hs_name==hs_names[-1]:
            #plot total minimum for all three chains

            ba_ax_pos += 1

            ba_ax = ba_fig.add_subplot(ba_ax_pos)
            [spline.set_visible(False) for spline in ba_ax.spines.values()]

            ba_ax.plot([],[],color=hi_color,label="Hawaiian Subtrack Misfit")
            ba_ax.plot([],[],color=em_color,label="Emperor Subtrack Misfit")
            ba_ax.plot([],[],color="tab:red",label="Total Track Misfit")
            ba_ax.scatter([],[],color=bend_color,label="Minimum Track Misfit",marker="s",edgecolors="k",zorder=10000)
            ba_ax.axvspan(-10,-10,color=bend_color,alpha=.4, label=r"$2\sigma$ Age Only Uncertainties")
            ba_ax.axvspan(-10,-10,color=extended_bend_unc_color,alpha=.4, label=r"$2\sigma$ Full Uncertainties")


#            inc_hs = ["Hawaii","Louisville HK19"]

#            ba_ax.plot(bend_ages,sum([np.array(hs_chi2s[hs][0]) for hs in inc_hs]),color=hi_color,label="Hawaiian Subtrack Misfit")
#            ba_ax.plot(bend_ages,sum([np.array(hs_chi2s[hs][1]) for hs in inc_hs]),color=em_color,label="Emperor Subtrack Misfit")
#            combined_total_chi2 = np.array(sum([np.array(hs_chi2s[hs][0]) for hs in inc_hs]))+np.array(sum([np.array(hs_chi2s[hs][1]) for hs in inc_hs]))
#            ba_ax.plot(bend_ages,combined_total_chi2,color="tab:red",label="Combined Misfit")
#            combined_min_bend_idx = np.argmin(combined_total_chi2)
#            cidx = np.argwhere(np.diff(np.sign(combined_total_chi2-(combined_total_chi2[combined_min_bend_idx]+4)))).flatten()
#            print("------------Common Bend--------------")
#            total_s1_geo = np.sqrt(sum(geo_age_uncs)/2)
#            total_com_unc = np.sqrt(((bend_ages[combined_min_bend_idx]-bend_ages[cidx])/2)**2 + total_s1_geo**2)
#            print("\tMaximum Likelihood Bend Age and 2sigma: ",bend_ages[combined_min_bend_idx], (bend_ages[combined_min_bend_idx]-bend_ages[cidx]), 2*total_com_unc)
#            print("-------------------------------------")
#            ba_ax.scatter([bend_ages[combined_min_bend_idx]],[combined_total_chi2[combined_min_bend_idx]],color=bend_color,label="Minimum Combined",marker="s",edgecolors="k",zorder=10000)
#            ba_ax.axvline(bend_ages[cidx][0],color=bend_color,linestyle="--")
#            if len(cidx)>1:
#                ba_ax.axvline(bend_ages[cidx][1],color=bend_color,linestyle="--")
#                ba_ax.axvspan(*bend_ages[cidx],color=bend_color,alpha=.4)

#            ba_ax.set_title("Combined Bend Age Misfit",fontsize=10)
##            ba_ax.set_ylabel(r"$\chi^2$")
#    #        ba_ax.annotate("%.1f + %.2f"%(),xy=(0.5,0.2),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="bottom",ha="right")
            ba_ax.legend(loc='center')
            ba_ax.set_yticks([])
            ba_ax.grid(False)
            ba_ax.set_xticks([])
            ba_ax.set_xlim(45.,55.)
#            ba_ax.set_ylim(0.,combined_total_chi2[combined_min_bend_idx]+20)
#            ba_ax.set_xlabel("Age (Ma)")
#            ba_ax.axes.yaxis.set_ticklabels([])
#            ba_ax.annotate(chr(64+ba_ax_pos%10)+")",xy=(1-0.04,0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="bottom",ha="right")

            if is_subair:
                ba_fig.savefig("./results/tmp/BendAgeGridsearch_all_subair.png")
                ba_fig.savefig("./results/tmp/BendAgeGridsearch_all_subair.pdf")
            else:
                ba_fig.savefig("./results/tmp/BendAgeGridsearch_all.png")
                ba_fig.savefig("./results/tmp/BendAgeGridsearch_all.pdf")

        heb_age = min_bend_age
        if len(idx)>1: heb_age_start,heb_age_end = min_bend_age+(min_bend_age-bend_ages[idx])
        elif len(idx)==1: heb_age_start,heb_age_end = min_bend_age-(min_bend_age-bend_ages[idx][0]),min_bend_age+(min_bend_age-bend_ages[idx][0])
        else: print("FUCK: No error bounds found!!!!!!!!!")
        heb_age_start,heb_age_end = heb_age-com_unc[0], heb_age+com_unc[1]

        hi_pols = min_hi_pols
        yp = np.polyval(hi_pols,good_dated_hi_data["Dis"])
        hi_chi2 = sum(((good_dated_hi_data["Age (Ma)"]-yp))**2)
        hi_dof = (len(good_dated_hi_data["Dis"])-deg-1) #1 greater than normal because of constraint
        hi_red_chi2 = hi_chi2/hi_dof
        hi_r2 = (stats.pearsonr(good_dated_hi_data["Dis"], good_dated_hi_data["Age (Ma)"])[0])**2
        #construct matrix alpha described in Bev 8-23
        alpha,err = [],np.sqrt(hi_chi2/hi_dof)*np.ones(len(good_dated_hi_data["Dis"]))
        for j,a1 in enumerate(hi_pols):
            alpha.append([])
            for k,a2 in enumerate(hi_pols):
                alpha[j].append(sum((err**-2)*(good_dated_hi_data["Dis"]**j)*(good_dated_hi_data["Dis"]**k)))
        #invert alpha to get error matrix which has the varriance of the ith model parameter on it's diagonal as per Bev 8-28
        error_matrix = np.linalg.inv(alpha)
#        if (err==np.ones(len(x))).all():
#            ss = sum((good_dated_hi_data["Age (Ma)"]-yp)**2)/dof
#            sds = np.sqrt(ss*np.diag(error_matrix))[::-1]
        sds = np.sqrt(np.diag(error_matrix))[::-1]
        hi_sds = sds
        hi_poly = np.poly1d(hi_pols)
        print("HAWAIIAN FIT")
        print(hi_pols,hi_sds,hi_chi2,hi_dof,hi_red_chi2,hi_r2,min_bend_age)
        print("Covariance Matrix:\n", np.flip(error_matrix))

        em_pols = min_em_pols
        yp = np.polyval(em_pols,good_dated_em_data["Dis"])
        em_chi2 = sum(((good_dated_em_data["Age (Ma)"]-yp))**2)
        em_dof = (len(good_dated_em_data["Dis"])-deg) #1 greater than normal because of constraint
        em_red_chi2 = em_chi2/em_dof
        em_r2 = (stats.pearsonr(good_dated_em_data["Dis"], good_dated_em_data["Age (Ma)"])[0])**2
        #construct matrix alpha described in Bev 8-23
        alpha,err = [],np.sqrt(em_chi2/em_dof)*np.ones(len(good_dated_em_data["Dis"]))
        for j,a1 in enumerate(em_pols):
            alpha.append([])
            for k,a2 in enumerate(em_pols):
                alpha[j].append(sum((err**-2)*(good_dated_em_data["Dis"]**j)*(good_dated_em_data["Dis"]**k)))
        #invert alpha to get error matrix which has the varriance of the ith model parameter on it's diagonal as per Bev 8-28
        error_matrix = np.linalg.inv(alpha)
#        if (err==np.ones(len(x))).all():
#            ss = sum((good_dated_em_data["Age (Ma)"]-yp)**2)/dof
#            sds = np.sqrt(ss*np.diag(error_matrix))[::-1]
        sds = np.sqrt(np.diag(error_matrix))[::-1]
        em_sds = sds
        em_poly = np.poly1d(em_pols)

        print("EMPEROR FIT")
        print(em_pols,em_sds,em_chi2,em_dof,em_red_chi2,em_r2,min_bend_age)
        print("Covariance Matrix:\n", np.flip(error_matrix))

    ####################################Fit Hawaiian Age/Dis data
    if HERM or bend_age_gridsearch: pass
    else:
        hi_pols,hi_sds,hi_chi2,hi_dof,hi_r2,hi_res,hi_rank,hi_sv,hi_rcond = utl.polyfit(good_dated_hi_data["Dis"],good_dated_hi_data["Age (Ma)"],deg,full=True)
        hi_pols,hi_sds,hi_adj_chi2,_,hi_r2,hi_res,hi_rank,hi_sv,hi_rcond = utl.polyfit(good_dated_hi_data["Dis"],good_dated_hi_data["Age (Ma)"],deg,err=np.sqrt(hi_chi2/hi_dof)*np.ones(len(good_dated_hi_data["Dis"])),full=True)
        hi_poly = np.poly1d(hi_pols)
        print("HAWAIIAN FIT")
        print(hi_pols,hi_sds,hi_chi2,hi_dof,hi_r2,hi_res,hi_rank,hi_sv,hi_rcond)
        print("-----------------",hi_chi2/hi_dof,hi_adj_chi2/hi_dof)

    if hs_name=="Hawaii":
        subair_hi_pols,subair_hi_sds,subair_hi_chi2,subair_hi_dof,subair_hi_r2,subair_hi_res,subair_hi_rank,subair_hi_sv,subair_hi_rcond = utl.polyfit(subair_dated_hi_data["Dis"],subair_dated_hi_data["Age (Ma)"],deg,full=True)
        subair_hi_pols,subair_hi_sds,subair_hi_adj_chi2,_,subair_hi_r2,subair_hi_res,subair_hi_rank,subair_hi_sv,subair_hi_rcond = utl.polyfit(subair_dated_hi_data["Dis"],subair_dated_hi_data["Age (Ma)"],deg,err=np.sqrt(subair_hi_chi2/subair_hi_dof)*np.ones(len(subair_dated_hi_data["Dis"])),full=True)
        print(subair_hi_pols,subair_hi_sds,subair_hi_chi2,subair_hi_dof,subair_hi_r2,subair_hi_res,subair_hi_rank,subair_hi_sv,subair_hi_rcond)
        print("-----------------",subair_hi_chi2/subair_hi_dof,subair_hi_adj_chi2/subair_hi_dof)
        t_val = (hi_pols[0]-subair_hi_pols[0])/np.sqrt(((subair_hi_sds[0]**2)/(subair_hi_dof+2)) + ((hi_sds[0]**2)/(hi_dof+2)))
        p_val = stats.t.pdf(t_val,subair_hi_dof+hi_dof+2)
        print("-----------------Slope Difference Test for Subarial vs. Submarine Samples",t_val,p_val)
        t_val = (hi_pols[1]-subair_hi_pols[1])/np.sqrt(((subair_hi_sds[1]**2)/(subair_hi_dof+2)) + ((hi_sds[1]**2)/(hi_dof+2)))
        p_val = stats.t.pdf(t_val,subair_hi_dof+hi_dof+2)
        print("-----------------Intercept Difference Test for Subarial vs. Submarine Samples",t_val,p_val)

    inv_data[hs_name]["HIpols"] = hi_pols
    inv_data[hs_name]["HIsds"] = hi_sds
    inv_data[hs_name]["HIAgeSd"] = np.sqrt(hi_chi2/hi_dof)
    inv_data[hs_name]["HI_dated_data"] = dated_hi_data
    inv_data[hs_name]["HI_start_azi"] = hi_start_azi
    inv_data[hs_name]["HI_mean_age"] = good_dated_hi_data["Age (Ma)"].mean()
    inv_data[hs_name]["HI_mean_dis"] = good_dated_hi_data["Dis"].mean()
    inv_data[hs_name]["HI_dated_N"] = len(good_dated_hi_data)
    inv_data[hs_name]["HI_Start_Dis"] = inv_data[hs_name]["HI_dated_data"].iloc[0]["Dis"]

    ############################################################Fit age Dis. Data

    ax_res = fig_res.add_subplot(res_pos)
    ax = fig.add_subplot(ax_pos)
    ax.annotate(chr(64+ax_pos%10)+")",xy=(1-0.04,0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="bottom",ha="right")

    if HERM:
        good_dated_em_data.sort_values("Age (Ma)",inplace=True)
        np.random.seed(sum(map(ord,"Mary had a little lamb, little lamb, little lamb")))
        all_em_pols, num_drop = [], int(.2*len(good_dated_em_data))
        for i in range(n_herm):
            drop_indices = np.random.choice(good_dated_em_data.index, num_drop, replace=True)
            tmp_data = good_dated_em_data.drop(drop_indices)
            em_pols = PchipInterpolator(tmp_data["Age (Ma)"],tmp_data["Dis"],extrapolate=False)
            all_em_pols.append(em_pols)
        em_chi2 = sum((good_dated_em_data["Dis"]-np.nanmean([em_pols(good_dated_em_data["Age (Ma)"]) for em_pols in all_em_pols],axis=0))**2) #defined as 0. because interpolation this is a check
        em_dof = 1. #This is actually defined as 0. because it's an interpolation
        em_adj_chi2 = 0. #This is actually defined as 0. in all cases
        em_r2 = stats.pearsonr(good_dated_em_data["Dis"],good_dated_em_data["Age (Ma)"])[0]**2
        em_sds,em_res,em_sv,em_rank,em_rcond = None,None,None,None,None
    elif bend_age_gridsearch: pass
    else:
        em_pols,em_sds,em_chi2,em_dof,em_r2,em_res,em_rank,em_sv,em_rcond = utl.polyfit(good_dated_em_data["Dis"],good_dated_em_data["Age (Ma)"],deg,full=True)
        em_pols,em_sds,em_adj_chi2,_,em_r2,em_res,em_rank,em_sv,em_rcond = utl.polyfit(good_dated_em_data["Dis"],good_dated_em_data["Age (Ma)"],deg,err=np.sqrt(em_chi2/em_dof)*np.ones(len(good_dated_em_data["Dis"])),full=True)
        em_poly = np.poly1d(em_pols)
        print("EMPEROR FIT")
        print(em_pols,em_sds,em_chi2,em_dof,em_r2,em_res,em_rank,em_sv,em_rcond)
        print("-----------------",em_chi2/em_dof,em_adj_chi2/em_dof)

    inv_data[hs_name]["EMpols"] = em_pols
    inv_data[hs_name]["EMsds"] = em_sds
    inv_data[hs_name]["EMAgeSd"] = np.sqrt(em_chi2/em_dof)
    inv_data[hs_name]["EM_dated_data"] = dated_em_data
    inv_data[hs_name]["EM_start_azi"] = em_start_azi
    inv_data[hs_name]["EM_mean_age"] = good_dated_em_data["Age (Ma)"].mean()
    inv_data[hs_name]["EM_mean_dis"] = good_dated_em_data["Dis"].mean()
    inv_data[hs_name]["EM_dated_N"] = len(good_dated_em_data)
    inv_data[hs_name]["EM_Start_Dis"] = em_start_dis

    #####################Determine the age of the bend in the two models
    if len(hi_pols)==2: hi_bend_age = hi_pols[0]*em_start_dis + hi_pols[1]
    else: hi_bend_age = np.polyval(hi_pols,em_start_dis)
#    else: raise ValueError("degree %d not supported"%deg)
    hi_bend_age_sd = np.sqrt((hi_sds[0]*(em_start_dis-good_dated_hi_data["Dis"].mean()))**2 + hi_sds[1]**2)/np.sqrt(len(good_dated_hi_data))
    if len(em_pols)==2: em_bend_age = em_pols[0]*em_start_dis + em_pols[1]
    else: em_bend_age = np.polyval(em_pols,em_start_dis)
#    else: raise ValueError("degree %d not supported"%deg)
    em_bend_age_sd = np.sqrt((em_sds[0]*(em_start_dis-good_dated_em_data["Dis"].mean()))**2 + em_sds[1]**2)/np.sqrt(len(good_dated_em_data))
    em_est_start_dis = (geodict["azi2"]-em_start_azi)*np.sin(np.deg2rad(em_dis))
    print("---------------Bend Age:")
    print("\tHawaiian stage: %.1f+-%.2f"%(hi_bend_age,2*hi_bend_age_sd))
    print("\tEmperor stage: %.1f+-%.2f"%(em_bend_age,2*em_bend_age_sd))

    if not bend_age_gridsearch: min_bend_age,heb_age_start,heb_age_end = (hi_bend_age+em_bend_age)/2,min(hi_bend_age,em_bend_age),max(hi_bend_age,em_bend_age)

    inv_data[hs_name]["BendAge"] = min_bend_age
    inv_data[hs_name]["BendAge1sigma_min"],inv_data[hs_name]["BendAge1sigma_max"] = heb_age_start,heb_age_end

    #In case a higher order degree is used with no clear inverted functional form, Correlation is high enough that the difference is not of great importance
#    inv_pols,inv_sds,inv_chi2,inv_dof,inv_r2,inv_res,inv_rank,inv_sv,inv_rcond = utl.polyfit(em_data["Age (Ma)"],em_data["Dis"],deg,full=True)
#    inv_pols,inv_sds,inv_adj_chi2,_,inv_r2,inv_res,inv_rank,inv_sv,inv_rcond = utl.polyfit(em_data["Age (Ma)"],em_data["Dis"],deg,err=np.sqrt(inv_chi2/inv_dof)*np.ones(len(em_data["Dis"])),full=True)

    ebar_ms,ebar_bad_ms = 4,None

    eb = ax.errorbar(good_dated_hi_data["Dis"].tolist(), good_dated_hi_data["Age (Ma)"].tolist(), yerr=np.ones(len(good_dated_hi_data))*np.sqrt(hi_chi2/hi_dof), marker="o", capsize=6, color="k",linestyle="",markersize=ebar_ms)
    eb[-1][0].set_linestyle(':')
    if hs_name=="Louisville HK19":
        spanned_data = good_dated_hi_data[good_dated_hi_data["Age Span"]==True]
        for _,row in spanned_data.iterrows():
            eb = ax.errorbar(row["Dis"], row["Age (Ma)"], yerr=row["2-sigma"], marker="o", capsize=4, color=hi_color,linestyle="",mec="k",markersize=ebar_ms)
            eb[-1][0].set_linestyle(':')
        gaussian_data = good_dated_hi_data[~(good_dated_hi_data["Age Span"]==True)]
        ax.errorbar(gaussian_data["Dis"].tolist(), gaussian_data["Age (Ma)"].tolist(), yerr=gaussian_data["2-sigma"].tolist(), marker="o", capsize=4, color=hi_color,linestyle="",mec="k",markersize=ebar_ms)
    else:
        ax.errorbar(good_dated_hi_data["Dis"].tolist(), good_dated_hi_data["Age (Ma)"].tolist(), yerr=good_dated_hi_data["2-sigma"].tolist(), marker="o", capsize=4, color=hi_color,linestyle="",mec="k",markersize=ebar_ms)

    ax.errorbar(bad_dated_hi_data["Dis"].tolist(), bad_dated_hi_data["Age (Ma)"].tolist(), yerr=bad_dated_hi_data["2-sigma"].tolist(), marker="X", capsize=4, color=hi_color,linestyle="",mec="k",markersize=ebar_bad_ms)

    eb = ax.errorbar(good_dated_em_data["Dis"].tolist(), good_dated_em_data["Age (Ma)"].tolist(), yerr=np.ones(len(good_dated_em_data))*np.sqrt(em_chi2/em_dof), marker="o", capsize=6, color="k",linestyle="",markersize=ebar_ms)
    eb[-1][0].set_linestyle(':')
    if hs_name=="Louisville HK19":
        spanned_data = good_dated_em_data[good_dated_em_data["Age Span"]==True]
        for _,row in spanned_data.iterrows():
            eb = ax.errorbar(row["Dis"], row["Age (Ma)"], yerr=row["2-sigma"], marker="o", capsize=4, color=em_color,linestyle="",mec="k",markersize=ebar_ms)
            eb[-1][0].set_linestyle(':')
        gaussian_data = good_dated_em_data[~(good_dated_em_data["Age Span"]==True)]
        ax.errorbar(gaussian_data["Dis"].tolist(), gaussian_data["Age (Ma)"].tolist(), yerr=gaussian_data["2-sigma"].tolist(), marker="o", capsize=4, color=em_color,linestyle="",mec="k",markersize=ebar_ms)
    else:
        ax.errorbar(good_dated_em_data["Dis"].tolist(), good_dated_em_data["Age (Ma)"].tolist(), yerr=good_dated_em_data["2-sigma"].tolist(), marker="o", capsize=4, color=em_color,linestyle="",mec="k",markersize=ebar_ms)

    ax.errorbar(bad_dated_em_data["Dis"].tolist(), bad_dated_em_data["Age (Ma)"].tolist(), yerr=bad_dated_em_data["2-sigma"].tolist(), marker="X", capsize=4, color=em_color,linestyle="",mec="k",markersize=ebar_bad_ms)

    if not HERM:
        #Histogram
        plt.sca(ax_res)
        em_good_residuals = good_dated_em_data["Age (Ma)"].tolist() - np.polyval(em_pols, good_dated_em_data["Dis"].tolist())
        em_bad_residuals = bad_dated_em_data["Age (Ma)"].tolist() - np.polyval(em_pols, bad_dated_em_data["Dis"].tolist())
        if len(em_good_residuals)>=8:
            _,p_norm = stats.normaltest(em_good_residuals)
            ax_res.annotate("$p_{norm} = %.3f$"%(p_norm),xy=(1-0.04,1-0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="top",ha="right")
        ax_res.hist(em_good_residuals, np.arange(-6,7,1), color=em_color, edgecolor="k",alpha=.7)
        ax_res.hist(em_bad_residuals, np.arange(-6,7,1), color=em_color, edgecolor="r",alpha=.3)
        ax_res.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_res.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        if hs_name=="Louisville HK19" or deg==3: is_bottom=True
        else: is_bottom=False
        if is_bottom: ax_res.set_xlabel("Age Residuals (Ma)")
        plt.tick_params(
            axis='both',          # changes the axis to apply changes
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge
            top=True,         # ticks along the top edge
            left=True,         # ticks along the left edge
            right=True,         # ticks along the right edge
            labelbottom=is_bottom, # labels along the bottom edge are var
            labelleft=True, # labels along the left edge are on
            labelright=False, # labels along the right edge are off
            tickdir="in") #changes direction of ticks
        ax_res.set_xlim(-8,8)
        ax_res.set_title(title)

        #Scatter Plot
        ax_res = fig_res.add_subplot(res_pos+1)
        plt.sca(ax_res)
        ax_res.scatter(good_dated_em_data["Dis"].tolist(), em_good_residuals, marker="o", color=em_color, edgecolor="k")
        ax_res.scatter(bad_dated_em_data["Dis"].tolist(), em_bad_residuals, marker="X", color=em_color, edgecolor="k")
        ax_res.axhline(0.,0.,1.,color="k",linestyle="--",label="Zero Line")
        ax_res.axhline(np.sqrt(em_chi2/em_dof),0.,1.,color=em_color,linestyle="--",label="1-Sigma from Seamount Dispursion")
        ax_res.axhline(-np.sqrt(em_chi2/em_dof),0.,1.,color=em_color,linestyle="--")

        # Make a plot with major ticks that are multiples of 20 and minor ticks that
        # are multiples of 5.  Label major ticks with '%d' formatting but don't label
        # minor ticks.
        ax_res.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax_res.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax_res.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax_res.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        # For the minor ticks, use no labels; default NullFormatter.
        ax_res.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax_res.yaxis.set_minor_locator(mticker.MultipleLocator(.5))

        plt.tick_params(
            axis='both',          # changes the axis to apply changes
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge
            top=True,         # ticks along the top edge
            left=True,         # ticks along the left edge
            right=True,         # ticks along the right edge
            labelbottom=False, # labels along the bottom edge are off
            labelleft=False, # labels along the left edge are off
            labelright=True, # labels along the right edge are on
            tickdir="in") #changes direction of ticks
        if hs_name=="Louisville HK19" or deg==3:
            ax.set_xlabel("Distance Along Track (GCD)")
            plt.tick_params(
                axis='both',          # changes the axis to apply changes
                which='both',      # both major and minor ticks are affected
                bottom=True,      # ticks along the bottom edge
                top=True,         # ticks along the top edge
                left=True,         # ticks along the top edge
                right=True,         # ticks along the right edge
                labelbottom=True, # labels along the bottom edge
                labelleft=False, # labels along the left edge are off
                labelright=True, # labels along the right edge are on
                tickdir="in") #changes direction of ticks
        if hs_name=="Rurutu":
            ax_res.set_ylabel("Age Residuals (Ma)")
            ax_res.yaxis.set_label_position("right")
            ax_res.yaxis.set_label_coords(1.1, -0.2)

        ax_res.set_xlim(0,20)
        ax_res.set_ylim(-8,8)
        plt.sca(ax)
#        ax.set_aspect("equal")

        em_midpoint = good_dated_em_data["Dis"].mean()
        for age in np.arange(50,90,10):
#            if hs_name=="Hawaii" or hs_name=="Louisville": possible_dists = (em_poly - age).roots[~np.iscomplex((em_poly - age).roots)]
            possible_dists = (em_poly - age).roots[~np.iscomplex((em_poly - age).roots)]
            possible_dists = possible_dists[(possible_dists<=25) & (possible_dists>=-10)]
            if len(possible_dists)==0: continue
            else: possible_dists = possible_dists[0]
#            print(age,possible_dists)
#            for d,sd in enumerate(em_sds[:-1]):
#                print(d,(len(em_sds)-1-d),sd,possible_dists,em_midpoint)
#                print((sd**2)*(((len(em_sds)-1-d)*possible_dists**(len(em_sds)-2-d))**2))
            slow = (np.polyval(np.polyder(em_pols, m=1),possible_dists))
            var_slow = sum([(sd**2)*(((len(em_sds)-1-d)*(possible_dists-em_midpoint)**(len(em_sds)-2-d))**2) for d,sd in enumerate(em_sds[:-1])])
#            print(age,slow,np.sqrt(var_slow),var_slow)
            em_rate = 111.113*(1/slow)
            em_rate_s1 = 111.113*np.sqrt(((var_slow)/(slow**4)))
            print("Rate of Hotspot Motion at %.0f Ma = %.1f+-%.1f"%(age,em_rate,em_rate_s1))

    if HERM:
        age_min,age_max = good_dated_em_data["Age (Ma)"].min(),good_dated_em_data["Age (Ma)"].max()
        x = np.arange(age_min,age_max,.1)
        y = np.nanmean([em_pols(x) for em_pols in all_em_pols],axis=0)
        s1_y = np.sqrt(np.nanmean([(em_pols(x)-y)**2 for em_pols in all_em_pols],axis=0))
        print(sum(y)/len(y),sum(s1_y)/len(y))
        dy = np.nanmean([np.abs(em_pols.derivative()(x)) for em_pols in all_em_pols],axis=0)
        em_rate = 111.113*((sum(dy)/len(x)))
        em_rate_s1 = 111.113*np.sqrt((sum(np.nanmean([(np.abs(em_pols.derivative()(x))-dy)**2 for em_pols in all_em_pols],axis=0))/len(x)))
        print(em_rate,em_rate_s1)
        uy = y+2*s1_y
        ly = y-2*s1_y

        ax.plot(uy,x,color=mean_color,linestyle=':')
        ax.plot(ly,x,color=mean_color,linestyle=':')
        ax.fill_betweenx(x, uy, ly, color=mean_color, alpha=0.2,zorder=-1)
        ax.plot(y,x,color=mean_color)

#        em_rate,em_rate_s1 = 111.113*(1/(sum(np.nan_to_num(em_pols.derivative()(x),nan=0.))/len(x))),0.
#        em_rate,em_rate_s1 = 111.113*(sum(np.nan_to_num(em_pols.derivative()(x),nan=0.))/len(x)),0.
#        if len(em_pols)>1: em_rate,em_rate_s1 = 111.113*(1/em_pols[-2]), ((em_sds[-2])/em_pols[-2])*111.113*(1/em_pols[-2])
#        else: em_rate,em_rate_s1 = 0.,0.
        ax.annotate("$Rate = %.0f \pm %.0f$ mm/a\n$\sigma_{age} = = %.1f$ Myr"%(em_rate,2*em_rate_s1,np.sqrt(em_chi2)),xy=(0.04,1-0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="top",ha="left")
    else:
        #################################################################Plot the Subarial Hawaiian fit
        if hs_name=="Hawaii" and not is_subair:
            x = np.linspace(0.,10.,10000)
            y = np.polyval(subair_hi_pols,x)
            uy = y+2*utl.polyenv(subair_hi_pols,x,np.ones(len(x))*np.sqrt(subair_hi_chi2/subair_hi_dof),center=subair_dated_hi_data["Dis"].mean())/np.sqrt(len(subair_dated_hi_data))
            ly = y-2*utl.polyenv(subair_hi_pols,x,np.ones(len(x))*np.sqrt(subair_hi_chi2/subair_hi_dof),center=subair_dated_hi_data["Dis"].mean())/np.sqrt(len(subair_dated_hi_data))

            ax.plot(x,uy,color="#888a85",linestyle=':',alpha=.7,zorder=0)
            ax.plot(x,ly,color="#888a85",linestyle=':',alpha=.7,zorder=0)
            ax.fill_between(x, uy, ly, color="#888a85", alpha=0.2,zorder=-1)
            ax.plot(x,y,color="#888a85",alpha=.7,zorder=0)

        #################################################################Plot the Hawaiian fit
#        x = np.linspace(0,good_dated_hi_data.iloc[-1]["Dis"],10000)
        x = np.linspace(0,em_start_dis,10000)
        y = np.polyval(hi_pols,x)
        hi_err_env = utl.polyenv(hi_pols,x,np.ones(len(x))*np.sqrt(hi_chi2/hi_dof),center=good_dated_hi_data["Dis"].mean())/np.sqrt(hi_dof)
        uy = y+2*hi_err_env
        ly = y-2*hi_err_env

        ax.plot(x,uy,color=mean_color,linestyle=':')
        ax.plot(x,ly,color=mean_color,linestyle=':')
        ax.fill_between(x, uy, ly, color=mean_color, alpha=0.2,zorder=-1)
        ax.plot(x,y,color=mean_color)

        if bend_age_gridsearch and hs_name!="Rurutu": ax.axhspan(heb_age_start,heb_age_end,color=bend_color,alpha=.3)
        elif not bend_age_gridsearch: ax.axhspan(heb_age_start,heb_age_end,color=bend_color,alpha=.3)

        if len(hi_pols)==2: hi_rate,hi_rate_s1 = 111.113*(1/hi_pols[-2]), ((hi_sds[-2])/hi_pols[-2])*111.113*(1/hi_pols[-2])
        elif len(hi_pols)>2:
            hi_zero_age = np.polyval(hi_pols,x[0])
            hi_bend_age = np.polyval(hi_pols,x[-1])
            hi_rate = 111.113*((x[-1]-x[0])/(hi_bend_age-hi_zero_age))
            hi_rate_s1 = 111.113*np.sqrt((hi_err_env[0]*((x[0]-x[-1])/((hi_bend_age-hi_zero_age)**2)))**2 + (hi_err_env[-1]*((x[-1]-x[0])/((hi_bend_age-hi_zero_age)**2))**2))
#            age = 60.
#            hi_midpoint = good_dated_hi_data["Dis"].mean()
#            possible_dists = (hi_poly - age).roots[~np.iscomplex((hi_poly - age).roots)]
#            possible_dists = possible_dists[(possible_dists<=25) & (possible_dists>=-10)]
#            if len(possible_dists)!=0:
#                possible_dists = possible_dists[0]
#                slow = (np.polyval(np.polyder(hi_pols, m=1),possible_dists))
#                var_slow = sum([(sd**2)*(((len(hi_sds)-1-d)*(possible_dists-hi_midpoint)**(len(hi_sds)-2-d))**2) for d,sd in enumerate(hi_sds[:-1])])
#                hi_rate = 111.113*(1/slow)
#                hi_rate_s1 = 111.113*np.sqrt(((var_slow)/(slow**4)))
#            else: hi_rate,hi_rate_s1 = 0.,0.
        else: hi_rate,hi_rate_s1 = 0.,0.
#        if deg==1: ax.annotate("$Rate = %.1f \pm %.1f$ mm/a\n$\sigma_{age} = %.2f$ Myr\n$\\nu = %d$\n$r^2 = %.2f$"%(hi_rate,2*hi_rate_s1,np.sqrt(hi_chi2/hi_dof),hi_dof,hi_r2),xy=(0.04,1-0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="top",ha="left")
#        else: ax.annotate("$Rate_{60Ma} = %.1f \pm %.1f$ mm/a\n$\sigma_{age} = %.2f$ Myr\n$\\nu = %d$"%(hi_rate,2*hi_rate_s1,np.sqrt(hi_chi2/hi_dof),hi_dof),xy=(0.04,1-0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="top",ha="left")

        #################################################################Plot the Emperor fit
#        x = np.linspace(good_dated_em_data.iloc[0]["Dis"],good_dated_em_data.iloc[-1]["Dis"],10000)
        x = np.linspace(em_start_dis,good_dated_em_data.iloc[-1]["Dis"],10000)
        y = np.polyval(em_pols,x)
        em_err_env = utl.polyenv(em_pols,x,np.ones(len(x))*np.sqrt(em_chi2/em_dof),center=good_dated_em_data["Dis"].mean())/np.sqrt(em_dof)
        uy = y+2*em_err_env
        ly = y-2*em_err_env

        ax.scatter([em_start_dis],[min_bend_age],color=bend_color,marker="s",edgecolors="k",zorder=10000,s=20,alpha=.8)
        ax.plot(x,uy,color=mean_color,linestyle=':')
        ax.plot(x,ly,color=mean_color,linestyle=':')
        ax.fill_between(x, uy, ly, color=mean_color, alpha=0.2,zorder=-1)
        ax.plot(x,y,color=mean_color)

        if len(em_pols)==2: em_rate,em_rate_s1 = 111.113*(1/em_pols[-2]), 111.113*np.sqrt((em_sds[-2]**2)/(em_pols[-2]**4))#((em_sds[-2])/em_pols[-2])*111.113*(1/em_pols[-2])
        elif len(em_pols)>2:
            em_zero_age = np.polyval(em_pols,x[0])
            em_bend_age = np.polyval(em_pols,x[-1])
            em_rate = 111.113*((x[-1]-x[0])/(em_bend_age-em_zero_age))
            em_rate_s1 = 111.113*np.sqrt((em_err_env[0]*((x[0]-x[-1])/((em_bend_age-em_zero_age)**2)))**2 + (em_err_env[-1]*((x[-1]-x[0])/((em_bend_age-em_zero_age)**2))**2))
#            age = 60.
#            em_midpoint = good_dated_em_data["Dis"].mean()
#            possible_dists = (em_poly - age).roots[~np.iscomplex((em_poly - age).roots)]
#            possible_dists = possible_dists[(possible_dists<=25) & (possible_dists>=-10)]
#            if len(possible_dists)!=0:
#                possible_dists = possible_dists[0]
#                slow = (np.polyval(np.polyder(em_pols, m=1),possible_dists))
#                var_slow = sum([(sd**2)*(((len(em_sds)-1-d)*(possible_dists-em_midpoint)**(len(em_sds)-2-d))**2) for d,sd in enumerate(em_sds[:-1])])
#                em_rate = 111.113*(1/slow)
#                em_rate_s1 = 111.113*np.sqrt(((var_slow)/(slow**4)))
#            else: em_rate,em_rate_s1 = 0.,0.
        else: em_rate,em_rate_s1 = 0.,0.
        if deg==1: ax.annotate("Hawaiian:\n\t$Rate = %.0f \pm %.0f$ mm/a\n\t$\sigma_{age} = %.1f$ Myr\n\t$\\nu = %d$\n\t$r^2 = %.2f$\nEmperor:\n\t$Rate = %.0f \pm %.0f$ mm/a\n\t$\sigma_{age} = %.1f$ Myr\n\t$\\nu = %d$\n\t$r^2 = %.2f$"%(hi_rate,2*hi_rate_s1,np.sqrt(hi_chi2/hi_dof),hi_dof,hi_r2,em_rate,2*em_rate_s1,np.sqrt(em_chi2/em_dof),em_dof,em_r2),xy=(0.04,1-0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="top",ha="left",fontsize=8)
        else: ax.annotate("Hawaiian:\n\t$\sigma_{age} = %.1f$ Myr\n\t$\\nu = %d$\nEmperor:\n\t$\sigma_{age} = %.1f$ Myr\n\t$\\nu = %d$"%(np.sqrt(hi_chi2/hi_dof),hi_dof,np.sqrt(em_chi2/em_dof),em_dof),xy=(0.04,1-0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="top",ha="left",fontsize=8)
#        if deg<=1: ax.annotate(reduce(lambda x,y: x+" + "+y,["$%.1f \pm %.2f$"%(p,2*sd)+"*$x$" if (len(em_pols)-1-i)>0 else "$%.1f \pm %.2f$"%(p,2*sd) for i,(p,sd) in enumerate(zip(em_pols,em_sds))]),xy=(.95,.1),xycoords="axes fraction",va="top",ha="right",fontsize=10)
#        else:
            #general polynomial string formatter bellow
#            ax.annotate(reduce(lambda x,y: x+" + "+y,["$%.2f \pm %.2f$"%(p,2*sd)+"*$x^%d$"%(len(em_pols)-1-i) if (len(em_pols)-1-i)>0 else "$%.1f \pm %.2f$"%(p,2*sd) for i,(p,sd) in enumerate(zip(em_pols,em_sds))]),xy=(.95,.1),xycoords="axes fraction",va="top",ha="right",fontsize=8)

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    # minor ticks.
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    # For the minor ticks, use no labels; default NullFormatter.
#    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
#    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

    plt.sca(ax)

    plt.tick_params(
        axis='both',          # changes the axis to apply changes
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge
        top=False,         # ticks along the top edge
        left=False,         # ticks along the left edge
        right=True,         # ticks along the right edge
        labelbottom=True, # labels along the bottom edge are off TODO
        labelleft=False, # labels along the left edge are off
        labelright=True, # labels along the right edge are on
        tickdir="in") #changes direction of ticks
    if hs_name=="Louisville HK19":
        ax.set_xlabel("Distance Along Track (GCD)")
        plt.tick_params(
            axis='both',          # changes the axis to apply changes
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge
            top=False,         # ticks along the top edge
            left=False,         # ticks along the top edge
            right=True,         # ticks along the right edge
            labelbottom=True, # labels along the bottom edge
            labelleft=False, # labels along the left edge are off
            labelright=True, # labels along the right edge are on
            tickdir="in") #changes direction of ticks
    if hs_name=="Rurutu":
        ax.set_ylabel("Age (Ma)")
        ax.yaxis.set_label_position("right")
#        ax.yaxis.set_label_coords(1.1, -0.05)
    if INTERP: ax.set_xlim(-2.,35.)
    else: ax.set_xlim(-2.,52.)
    ax.set_ylim(0.,85.)
    if hs_name!="Hawaii": plt.setp(ax.get_yticklabels()[0], visible=False)
    plt.setp(ax.get_yticklabels()[-1], visible=False)

    ###############################################Put Predicted Nominal Ages on Seamount Chain
    glow_color = "grey"
    fontsize = 9
    prev_dis = -5
    for age in np.arange(int(age_step*(round(float(age_min)/age_step))),int(age_step*int(float(age_max)/age_step)+age_step),age_step): #take ceil of min and floor of max with the step as base to avoid extrapolating

        if HERM and age<age_min: age += age_step
        if hs_name=="Louisville" and HERM and age==75.: len_ticks = 5.5
        elif hs_name=="Hawaii" and age==80.: len_ticks = 1.0
        elif INTERP: len_ticks = 2.0
        else: len_ticks = 2.0

        if hs_name=="Rurutu" and age>10. and age<45.: continue

        if age<heb_age:
            plat,plon,pstart_azi,pdis = hi_lat,hi_lon,hi_start_azi,hi_dis
            if len(hi_pols)==2: dis = ((age-hi_pols[1])/hi_pols[0] - dated_hi_data.iloc[0]["Dis"])/np.sin(np.deg2rad(hi_dis))
            elif len(hi_pols)==3: dis = ((((-hi_pols[1]+np.sqrt(hi_pols[1]**2 - 4*hi_pols[0]*(hi_pols[2]-age)))/(2*hi_pols[0]))) - dated_hi_data.iloc[0]["Dis"])/np.sin(np.deg2rad(hi_dis))
            elif len(hi_pols)==4:
                del_0 = hi_pols[1]**2 - 3*hi_pols[0]*hi_pols[2]
                del_1 = 2*hi_pols[1]**3 - 9*hi_pols[0]*hi_pols[1]*hi_pols[2] + 27*(hi_pols[0]**2)*(hi_pols[3]-age)
                C = ((del_1 + np.sqrt(del_1**2 - 4*del_0**3))/2)**(1/3)
                dis = (-1/3*hi_pols[0])*(hi_pols[1]+C-del_0/C)
            elif HERM: dis = (np.nanmean([hi_pols(age) for hi_pols in all_hi_pols]) - dated_hi_data.iloc[0]["Dis"])/np.sin(np.deg2rad(hi_dis))
            else:
                possible_dists = (hi_poly - age).roots[~np.iscomplex((hi_poly - age).roots)]
                possible_dists = possible_dists[possible_dists>=prev_dis]
                if len(possible_dists)==0:
                    continue
                else:
                    idx = np.abs(possible_dists).argmin()
                    dis = possible_dists[idx]
        else:
            plat,plon,pstart_azi,pdis = em_lat,em_lon,em_start_azi,em_dis
            if len(em_pols)==2: dis = ((age-em_pols[1])/em_pols[0] - em_start_dis)/np.sin(np.deg2rad(em_dis))
            elif len(em_pols)==3: dis = ((((-em_pols[1]+np.sqrt(em_pols[1]**2 - 4*em_pols[0]*(em_pols[2]-age)))/(2*em_pols[0]))) - dated_em_data.iloc[0]["Dis"])/np.sin(np.deg2rad(em_dis))
            elif len(em_pols)==4:
                del_0 = em_pols[1]**2 - 3*em_pols[0]*em_pols[2]
                del_1 = 2*em_pols[1]**3 - 9*em_pols[0]*em_pols[1]*em_pols[2] + 27*(em_pols[0]**2)*(em_pols[3]-age)
                C = ((del_1 + np.sqrt(del_1**2 - 4*del_0**3))/2)**(1/3)
                dis = (-1/3*em_pols[0])*(em_pols[1]+C-del_0/C)
            elif HERM: dis = (np.nanmean([em_pols(age) for em_pols in all_em_pols]) - dated_em_data.iloc[0]["Dis"])/np.sin(np.deg2rad(em_dis))
            else:
                possible_dists = (em_poly - age).roots[~np.iscomplex((em_poly - age).roots)]
                possible_dists = possible_dists[possible_dists>=prev_dis]
                if len(possible_dists)==0:
                    continue
                else:
                    idx = np.abs(possible_dists).argmin()
                    dis = possible_dists[idx]

        if np.isnan(dis): print("%d Ma point was not invertable given age model"%age); continue

        pgeodict = geoid.ArcDirect(plat,plon,pstart_azi+dis,-pdis) #modeled geographic point for age
        print(age,plat,plon,pstart_azi,dis,pgeodict["lat2"],pgeodict["lon2"],pgeodict["azi1"])
        tick_geodict,ha,va = geoid.ArcDirect(pgeodict["lat2"],pgeodict["lon2"],pgeodict["azi2"],-len_ticks),"right","center_baseline"
        tk = m.plot([pgeodict["lon2"],tick_geodict["lon2"]],[pgeodict["lat2"],tick_geodict["lat2"]],color="k",linewidth=1.5,transform = ccrs.Geodetic(),zorder=1)
    #    tk.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w', alpha=.7)])
        if INTERP:
            strike = 0
        else:
            geodict_1 = geoid.ArcDirect(plat,plon,pstart_azi+dis+1,-pdis) #modeled geographic point for age
            strike = geoid.Inverse(pgeodict["lat2"],pgeodict["lon2"],geodict_1["lat2"],geodict_1["lon2"])["azi1"]
        if strike > 180: strike = 180-strike
        txt = m.text(tick_geodict["lon2"],tick_geodict["lat2"],"%.0f Ma"%age,va=va,ha=ha,transform=ccrs.PlateCarree(),rotation=-strike,color="k",fontsize=fontsize)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground=glow_color, alpha=.7)])
        prev_dis = dis

#######################################################Plot dated data on map with Annotations

    if INTERP: long_tick = 7.
    else: long_tick = 7.
    glow_color = "k"

    for i,datum in dated_hi_data.iterrows(): #HAWAIIAN

        #Check for age
        if np.isnan(datum["Age (Ma)"]): continue
        else: color = hi_color
        if datum["Quality"]=="g": marker,marker_size = "o",good_marker_size
        else: marker,marker_size = "X",bad_marker_size
        m = psk.plot_pole(datum["Longitude"],datum["Latitude"],datum["Azi"],2*datum["Maj"],2*datum["Min"],edgecolors="k",facecolors=color,color=color,marker=marker,m=m,zorder=4,s=marker_size)
        bm = psk.plot_pole(datum["Longitude"],datum["Latitude"],datum["Azi"],2*datum["Maj"],2*datum["Min"],edgecolors="k",facecolors=color,color=color,marker=marker,m=bm,zorder=4,s=marker_size)

        #Standard Label handles
        text = "%.0f Ma"%datum["Age (Ma)"]
        #Special Label handles
        if datum["Seamount"]=="Oahu":
            text = "%.0f-%.0f Ma"%(7.,0.)
        elif datum["Seamount"]=="Hawai'i" or datum["Seamount"]=="Kahoolawe" or datum["Seamount"]=="West Maui" or datum["Seamount"]=="Lanai" or datum["Seamount"]=="Molokai" or datum["Seamount"]=="Nihoa" or datum["Seamount"]=="Kauai": continue
        elif datum["Seamount"]=="Helsley":
            near_seamount = dated_hi_data[dated_hi_data["Seamount"]=="#63 (Clague 1996)"].iloc[0]
            text = "%.0f-%.0f Ma"%(near_seamount["Age (Ma)"],datum["Age (Ma)"])
        elif datum["Seamount"]=="#63 (Clague 1996)" or datum["Seamount"]=="Unnamed (postshield)": continue
        elif datum["Seamount"]=="North Kammu":
            text = "%.0f-%.0f Ma"%(48.,44.)
        elif datum["Seamount"]=="Yuryaku" or datum["Seamount"]=="Daikakuji": continue

        #Extend some ticks to fit them all nicely
        len_ticks = 2.
        if "Louisville" in hs_name and datum["Age (Ma)"]>33.: len_ticks = long_tick-1
        elif "Hawaii" in hs_name and datum["Age (Ma)"]>33.: len_ticks = long_tick+1

        if datum["Seamount"]=="Midway": len_ticks = long_tick
        elif datum["Seamount"]=="Atiu": len_ticks = long_tick
        elif datum["Seamount"]=="158.5°W": len_ticks = long_tick
        elif datum["Seamount"]=="160.7°W": len_ticks = long_tick
        elif datum["Seamount"]=="36.9S Guyot": len_ticks = long_tick
        elif datum["Seamount"]=="Valerie": len_ticks = long_tick+4.
        elif datum["Seamount"]=="LOU-3": len_ticks = long_tick+4.
        elif datum["Seamount"]=="165.7°W": len_ticks = long_tick+4.
        elif datum["Seamount"]=="166.6°W": len_ticks = long_tick+4.
        elif datum["Seamount"]=="167.4°W": len_ticks = long_tick+4.
        elif datum["Seamount"]=="167.7°W": len_ticks = long_tick+4.
        elif datum["Seamount"]=="168.3°W" and "Heaton" not in hs_name: len_ticks = 1.5

        #Calculate Tick and Plot text
        if INTERP:
            mgeodict = {"lat2":datum["Latitude"],"lon2":datum["Longitude"]}
        else:
            geodict = geoid.Inverse(datum["Latitude"],datum["Longitude"],hi_lat,hi_lon)
            mgeodict = geoid.ArcDirect(hi_lat,hi_lon,geodict["azi2"],-hi_dis)
        geodict_1 = geoid.ArcDirect(hi_lat,hi_lon,geodict["azi2"]+1,-hi_dis)
        strike = geoid.Inverse(mgeodict["lat2"],mgeodict["lon2"],geodict_1["lat2"],geodict_1["lon2"])["azi1"]
        tick_azi = mgeodict["azi2"]
        if "Hawaii" in hs_name and datum["Age (Ma)"]>33.: strike+=10.; tick_azi+=10.
        if datum["Seamount"]=="Rurutu": tick_azi-=20.
        elif datum["Seamount"]=="167.7°W": tick_azi-=6.
        elif datum["Seamount"]=="167.4°W": tick_azi-=3.
        elif datum["Seamount"]=="165.4°W": tick_azi+=2.
        elif datum["Seamount"]=="LOU-9": tick_azi-=4.
        elif datum["Seamount"]=="168.6°W (Hadar Guyot)" or datum["Seamount"]=="168.6°W": tick_azi-=10.
#        if hs_name=="Hawaii": strike = geoid.Inverse(mgeodict["lat2"],mgeodict["lon2"],hi_lat,hi_lon)["azi2"]
#        else: strike = -geoid.Inverse(mgeodict["lat2"],mgeodict["lon2"],hi_lat,hi_lon)["azi2"]
        tick_geodict = geoid.ArcDirect(mgeodict["lat2"],mgeodict["lon2"],tick_azi,len_ticks) #horizontal
        tk = m.plot([mgeodict["lon2"],tick_geodict["lon2"]],[mgeodict["lat2"],tick_geodict["lat2"]],color=hi_color,linewidth=1.5,transform = ccrs.Geodetic(),zorder=1)
        txt = m.text(tick_geodict["lon2"],tick_geodict["lat2"],text,va="bottom",ha="left",transform=ccrs.PlateCarree(),color=hi_color,fontsize=fontsize, rotation=-strike)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground=glow_color, alpha=.7)])

    for i,datum in dated_em_data.iterrows(): #EMPEROR

        #Check for age
        if np.isnan(datum["Age (Ma)"]): continue
        else: color = em_color
        if datum["Quality"]=="g": marker,marker_size = "o",good_marker_size
        else: marker,marker_size = "X",bad_marker_size
        m = psk.plot_pole(datum["Longitude"],datum["Latitude"],datum["Azi"],2*datum["Maj"],2*datum["Min"],edgecolors="k",facecolors=color,color=color,marker=marker,m=m,zorder=4,s=marker_size)
        bm = psk.plot_pole(datum["Longitude"],datum["Latitude"],datum["Azi"],2*datum["Maj"],2*datum["Min"],edgecolors="k",facecolors=color,color=color,marker=marker,m=bm,zorder=4,s=marker_size)

        #Special Label handles
        if datum["Seamount"]=="168.6°W" or datum["Seamount"]=="168.6°W (Hadar Guyot)":
            near_seamount = dated_em_data[dated_em_data["Seamount"]=="168.3°W"].iloc[0]
            text = "%.0f-%.0f Ma"%(datum["Age (Ma)"],near_seamount["Age (Ma)"])
        elif datum["Seamount"]=="168.3°W": continue
        elif datum["Seamount"]=="Kosciusko":
            near_seamount = dated_em_data[dated_em_data["Seamount"]=="East Niulakita"].iloc[0]
            text = "%.0f-%.0f Ma"%(datum["Age (Ma)"],near_seamount["Age (Ma)"])
        elif datum["Seamount"]=="East Niulakita": continue
        elif datum["Seamount"]=="Nukufetau":
            near_seamounts = dated_em_data[(dated_em_data["Latitude"]>=-9) & (dated_em_data["Latitude"]<=-7)]
            text = "%.0f-%.0f Ma"%(max(near_seamounts["Age (Ma)"]),min(near_seamounts["Age (Ma)"]))
        elif datum["Seamount"]=="Funafuti" or datum["Seamount"]=="Telematua" or datum["Seamount"]=="Vaitupu" or datum["Seamount"]=="Tayasa" or datum["Seamount"]=="Laupapa": continue
        elif datum["Seamount"]=="Kautu":
            near_seamounts = dated_em_data[(dated_em_data["Latitude"]>=-2) & (dated_em_data["Latitude"]<=0)]
            text = "%.0f-%.0f Ma"%(max(near_seamounts["Age (Ma)"]),min(near_seamounts["Age (Ma)"]))
        elif datum["Seamount"]=="Palutu" or datum["Seamount"]=="Beru": continue
        elif datum["Seamount"]=="Koko N. (shield)":
            if "Koko S." in dated_em_data["Seamount"]:
                near_seamount = dated_em_data[dated_em_data["Seamount"]=="Koko S."].iloc[0]
                text = "%.0f-%.0f Ma"%(datum["Age (Ma)"],near_seamount["Age (Ma)"])
            else: text = "%.0f Ma"%datum["Age (Ma)"]
        elif datum["Seamount"]=="Koko S.": continue
        elif datum["Seamount"]=="Ojin": continue
        else: text = "%.0f Ma"%datum["Age (Ma)"]

        #Extend some ticks to fit them all nicely
        if datum["Seamount"]=="Burtaritari": len_ticks = long_tick
        elif datum["Seamount"]=="Nukufetau": len_ticks += 1.5
        elif datum["Seamount"]=="Logotau": len_ticks = long_tick
        elif datum["Seamount"]=="27.6°S (Volcano 33)": len_ticks = long_tick
        elif datum["Seamount"]=="Canopus Guyot": len_ticks = long_tick
        elif datum["Seamount"]=="Kosciusko": len_ticks += 1.5
        else: len_ticks = 1.5

        #Calculate Tick and Plot text
        if INTERP:
            mgeodict = {"lat2":datum["Latitude"],"lon2":datum["Longitude"]}
        else:
            geodict = geoid.Inverse(datum["Latitude"],datum["Longitude"],em_lat,em_lon)
            mgeodict = geoid.ArcDirect(em_lat,em_lon,geodict["azi2"],-em_dis)
        tick_azi = 90.
        if datum["Seamount"]=="27.6°S (Volcano 33)": tick_azi+=5.
        elif datum["Seamount"]=="Taring Nui": tick_azi-=15.
#        if hs_name=="Hawaii": strike = geoid.Inverse(mgeodict["lat2"],mgeodict["lon2"],hi_lat,hi_lon)["azi2"]
#        else: strike = -geoid.Inverse(mgeodict["lat2"],mgeodict["lon2"],hi_lat,hi_lon)["azi2"]
        tick_geodict = geoid.ArcDirect(mgeodict["lat2"],mgeodict["lon2"],tick_azi,len_ticks) #horizontal
        tk = m.plot([mgeodict["lon2"],tick_geodict["lon2"]],[mgeodict["lat2"],tick_geodict["lat2"]],color=em_color,linewidth=1.5,transform = ccrs.Geodetic(),zorder=1)
        txt = m.text(tick_geodict["lon2"],tick_geodict["lat2"],text,va="center",ha="left",transform=ccrs.PlateCarree(),color=em_color,fontsize=fontsize)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground=glow_color, alpha=.7)])

    ax.set_title(title)

    ###############################################Print Predicted ages and Uncertainties
    #uses piecewise fit
    #midpoint = sum(hi_data["Dis"])/len(hi_data["Dis"])
    #for i,datum in hi_data.iterrows():
    #    page = np.polyval(hi_pols,datum["Dis"])
    #    s1_page = np.sqrt(sum([(sd*(datum["Dis"]-midpoint)**(len(hi_pols)-1-j))**2 for j,sd in enumerate(hi_sds)]))
    #    hi_data.at[i,"MAge"] = page
    #    hi_data.at[i,"MAge_2sig"] = 2*s1_page
    #    print(datum["Seamount"],":",datum["Age (Ma)"],datum["2-sigma"],page,2*s1_page)

    midpoint = sum(dated_em_data["Dis"])/len(dated_em_data["Dis"])
    print(midpoint)
    if HERM:
        dated_em_data["MAge"] = dated_em_data["Age (Ma)"] #em_pols(dated_em_data["Dis"])
        dated_em_data["MAge_2sig"] = 0. #defined 0. due to interpolation
    else:
        dated_em_data["MAge"] = np.polyval(em_pols,dated_em_data["Dis"])
        dated_em_data["MAge_2sig"] = utl.polyenv(em_pols,dated_em_data["Dis"],np.ones(len(dated_em_data["Dis"]))*np.sqrt(em_chi2/em_dof))
    for i,datum in dated_em_data.iterrows():
        print(datum["Seamount"],":",datum["Age (Ma)"],datum["2-sigma"],datum["MAge"],datum["MAge_2sig"])

    #plt.fill_between(dated_em_data["Dis"].tolist(), dated_em_data["MAge"]+dated_em_data["MAge_2sig"], dated_em_data["MAge"]-dated_em_data["MAge_2sig"], color="pink", alpha=0.6,zorder=-1)

    ###############################################Save Circle Poles & Seamounts
    #out_path = os.path.join(os.path.dirname(sys.argv[1]),"%s_HIAge_%.1f.tsv"%(hs_name,heb_age))
    #with open(out_path,"w+",newline="") as fout:
    #    fout.write("%.2f\t%.2f\t%.2f\t%.2f\t%.4f\n"%(hi_lat,hi_lon,hi_rot,hi_dis,min_bend_err))
    #    hi_data.to_csv(fout,header=1,sep="\t",index=False)

#    out_path = os.path.join(os.path.dirname(sys.argv[1]),"%s_EMAge_%.1f.tsv"%(hs_name,heb_age))
#    with open(out_path,"w+",newline="") as fout:
#        if INTERP: fout.write("%.2f\t%.2f\t%.2f\t%.2f\t%.4f\n"%(0.,0.,0.,0.,0.))
#        else: fout.write("%.2f\t%.2f\t%.2f\t%.4f\n"%(em_lat,em_lon,em_dis,min_bend_err))
#        dated_em_data.to_csv(fout,header=1,sep="\t",index=False)

    res_pos += 2
    ax_pos += 1
    bax_pos += 1

mp.set_global()
mp.legend(framealpha=.7)

df = pd.DataFrame(inv_data)
df_trimmed = df.drop("HI_dated_data",axis=0)
df_trimmed = df_trimmed.drop("EM_dated_data",axis=0)
if is_subair: df_trimmed.T.to_csv("data/Bend_Inversion_data_%.2f_%.2f_subair.csv"%(resolution,bend_resolution))
else: df_trimmed.T.to_csv("data/Bend_Inversion_data_%.2f_%.2f.csv"%(resolution,bend_resolution))

if HERM: fig.savefig("./results/tmp/AgeModelsMasterHERM_%ddeg.png"%deg,bbox_inches="tight")
else:
    if is_subair:
        fig.savefig("./results/tmp/Bend_AgeModelsMaster_%ddeg_%.2f_%.2f_subair.png"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")
    #    fig.savefig("./results/tmp/Bend_AgeModelsMaster_%ddeg.pdf"%deg,dpi=200,bbox_inches="tight")
        bfig.savefig("./results/tmp/Bend_Check_%ddeg_%.2f_%.2f_subair.pdf"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")
        bfig.savefig("./results/tmp/Bend_Check_%ddeg_%.2f_%.2f_subair.png"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")
        fig_res.savefig("./results/tmp/Bend_AgeModelsRes_%ddeg_%.2f_%.2f_subair.png"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")
        fig_poles.savefig("./results/tmp/Bend_GeoModelsMaster_%ddeg_%.2f_%.2f_subair.png"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")
    else:
        fig.savefig("./results/tmp/Bend_AgeModelsMaster_%ddeg_%.2f_%.2f.png"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")
    #    fig.savefig("./results/tmp/Bend_AgeModelsMaster_%ddeg.pdf"%deg,dpi=200,bbox_inches="tight")
        bfig.savefig("./results/tmp/Bend_Check_%ddeg_%.2f_%.2f.pdf"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")
        bfig.savefig("./results/tmp/Bend_Check_%ddeg_%.2f_%.2f.png"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")
        fig_res.savefig("./results/tmp/Bend_AgeModelsRes_%ddeg_%.2f_%.2f.png"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")
        fig_poles.savefig("./results/tmp/Bend_GeoModelsMaster_%ddeg_%.2f_%.2f.png"%(deg,resolution,bend_resolution),dpi=200,bbox_inches="tight")

######################################################Inter-Hotspot difference figure

fig_age_points = plt.figure(figsize=(8,11),dpi=200)
fig_age_points.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.05,hspace=0.2)
points_pos = 311
age_step_dis = .1
ages = np.arange(40.,55.+age_step_dis,age_step_dis)

for hs1 in ["Hawaii","Rurutu","Louisville HK19"]:
    ax_points = fig_age_points.add_subplot(points_pos,projection="3d")

    age_m_dis = []
    hi_points,hi_azis,em_points,em_azis = [],[],[],[]
    for age in ages:
        age_dis = ((age-inv_data[hs1]["HIpols"][1])/inv_data[hs1]["HIpols"][0] - inv_data[hs1]["HI_dated_data"].iloc[0]["Dis"])/np.sin(np.deg2rad(inv_data[hs1]["HIDis"]))
        hi_geodict = geoid.ArcDirect(inv_data[hs1]["HILat"],inv_data[hs1]["HILon"],inv_data[hs1]["HI_start_azi"]+age_dis,-inv_data[hs1]["HIDis"]) #modeled geographic point for age
        hi_points.append([hi_geodict["lat2"],hi_geodict["lon2"]])
        hi_azis.append(age_dis)

        age_dis = ((age-inv_data[hs1]["EMpols"][1])/inv_data[hs1]["EMpols"][0] - inv_data[hs1]["EM_dated_data"].iloc[0]["Dis"])/np.sin(np.deg2rad(inv_data[hs1]["EMDis"]))
        em_geodict = geoid.ArcDirect(inv_data[hs1]["EMLat"],inv_data[hs1]["EMLon"],inv_data[hs1]["EM_start_azi"]+age_dis,-inv_data[hs1]["EMDis"]) #modeled geographic point for age
        em_points.append([em_geodict["lat2"],em_geodict["lon2"]])
        em_azis.append(age_dis)

        age_m_dis.append(geoid.Inverse(hi_geodict["lat2"],hi_geodict["lon2"],em_geodict["lat2"],em_geodict["lon2"])["a12"])

    min_dis = 1e9
    for hi_point in hi_points:
        for em_point in em_points:
            dis = geoid.Inverse(*hi_point,*em_point)["a12"]
            if dis<min_dis:
                min_dis=dis
                hi_min_idx=hi_points.index(hi_point)
                em_min_idx=em_points.index(em_point)

    print(min_dis,hi_min_idx,hi_points[hi_min_idx],hi_azis[hi_min_idx],ages[hi_min_idx])
    print(em_min_idx,em_points[em_min_idx],em_azis[em_min_idx],ages[em_min_idx])

#    ax_points.plot(hi_lons,hi_lats,ages,color=hi_color)
    ax_points.plot(ages,age_m_dis,color=hi_color)
    print(ages[age_m_dis.index(min(age_m_dis))],min(age_m_dis))
    ax_points.set_title(hs1)

    points_pos += 1

#fig_age_points.savefig("./results/tmp/Bend_%ddeg.pdf"%deg,dpi=200,bbox_inches="tight")

fig_dis = plt.figure(figsize=(11,8),dpi=200)
fig_dis.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.05,hspace=0.2)
dis_pos = 111
age_step_dis = 1.
ax_dis = fig_dis.add_subplot(dis_pos)

fig_div = plt.figure(figsize=(8,11),dpi=200)
fig_div.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.05,hspace=0.2)
div_pos = 311

hi_ages_iter = np.arange(0.,49.,age_step_dis)
em_ages_iter = np.arange(45.,81.,age_step_dis)
for k,(hs1,hs2) in enumerate(subsets(["Hawaii","Rurutu","Louisville HK19"],2)):
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][k]
    ax_div = fig_div.add_subplot(div_pos)

    if hs1!="Rurutu" and hs2!="Rurutu":
        hi_inter_hs_dis,hi_inter_hs_dis_sds,hi_azis,hi_ages = [],[],[],[]
        for age in hi_ages_iter:
            if len(inv_data[hs1]["HIpols"])==2: age1_dis = ((age-inv_data[hs1]["HIpols"][1])/inv_data[hs1]["HIpols"][0] - inv_data[hs1]["HI_dated_data"].iloc[0]["Dis"])/np.sin(np.deg2rad(inv_data[hs1]["HIDis"]))
            elif len(inv_data[hs1]["HIpols"])==3: age1_dis = ((((-inv_data[hs1]["HIpols"][1]+np.sqrt(inv_data[hs1]["HIpols"][1]**2 - 4*inv_data[hs1]["HIpols"][0]*(inv_data[hs1]["HIpols"][2]-age)))/(2*inv_data[hs1]["HIpols"][0]))) - inv_data[hs1]["HI_dated_data"].iloc[0]["Dis"])/np.sin(np.deg2rad(inv_data[hs1]["HIDis"]))
            else: raise ValueError("Cannot determine simple inverse for Hawaiian age model of %s given model with %d parameters"%(hs1,len(inv_data[hs1]["HIpols"])))
            hs1_geodict = geoid.ArcDirect(inv_data[hs1]["HILat"],inv_data[hs1]["HILon"],inv_data[hs1]["HI_start_azi"]+age1_dis,-inv_data[hs1]["HIDis"]) #modeled geographic point for age
            hs1_a = np.sqrt(2)*np.sqrt((inv_data[hs1]["HIsds"][1]**2)*(inv_data[hs1]["HIpols"][1]/inv_data[hs1]["HIpols"][0])**2 + (inv_data[hs1]["HIsds"][0]**2)*(-(age-inv_data[hs1]["HIpols"][1])*inv_data[hs1]["HIpols"][0]**-2)**2)*np.sin(np.deg2rad(inv_data[hs1]["HIDis"]))
            hs1_b = np.sqrt(2)*(33./111.113)
            hs1_phi = (hs1_geodict["azi2"]+90.)
#            print(hs1,inv_data[hs1]["HILat"],inv_data[hs1]["HILon"],inv_data[hs1]["HI_start_azi"],age,age1_dis,hs1_geodict["lat2"],hs1_geodict["lon2"],hs1_geodict["azi1"])

            if len(inv_data[hs2]["HIpols"])==2: age2_dis = ((age-inv_data[hs2]["HIpols"][1])/inv_data[hs2]["HIpols"][0] - inv_data[hs2]["HI_dated_data"].iloc[0]["Dis"])/np.sin(np.deg2rad(inv_data[hs2]["HIDis"]))
            elif len(inv_data[hs2]["HIpols"])==3: age2_dis = ((((-inv_data[hs2]["HIpols"][1]+np.sqrt(inv_data[hs2]["HIpols"][1]**2 - 4*inv_data[hs2]["HIpols"][0]*(inv_data[hs2]["HIpols"][2]-age)))/(2*inv_data[hs2]["HIpols"][0]))) - inv_data[hs2]["HI_dated_data"].iloc[0]["Dis"])/np.sin(np.deg2rad(inv_data[hs2]["HIDis"]))
            else: raise ValueError("Cannot determine simple inverse for Hawaiian age model of %s given model with %d parameters"%(hs2,len(inv_data[hs2]["HIpols"])))
            hs2_geodict = geoid.ArcDirect(inv_data[hs2]["HILat"],inv_data[hs2]["HILon"],inv_data[hs2]["HI_start_azi"]+age2_dis,-inv_data[hs2]["HIDis"]) #modeled geographic point for age
            hs2_a = np.sqrt(2)*np.sqrt((inv_data[hs2]["HIsds"][1]**2)*(inv_data[hs2]["HIpols"][1]/inv_data[hs2]["HIpols"][0])**2 + (inv_data[hs2]["HIsds"][0]**2)*(-((age)-inv_data[hs2]["HIpols"][1])*inv_data[hs2]["HIpols"][0]**-2)**2)*np.sin(np.deg2rad(inv_data[hs2]["HIDis"]))
            hs2_b = np.sqrt(2)*(33./111.113)
            hs2_phi = (hs2_geodict["azi2"]+90.)
#            print(hs2,inv_data[hs2]["HILat"],inv_data[hs2]["HILon"],inv_data[hs2]["HI_start_azi"],age,age2_dis,hs2_geodict["lat2"],hs2_geodict["lon2"],hs2_geodict["azi1"])

            if np.isnan(hs1_geodict["lat2"]) or np.isnan(hs1_geodict["lon2"]) or np.isnan(hs2_geodict["lat2"]) or np.isnan(hs2_geodict["lon2"]): continue
            IHS_geodict = geoid.Inverse(hs1_geodict["lat2"],hs1_geodict["lon2"],hs2_geodict["lat2"],hs2_geodict["lon2"])
            hs1_sd = (hs1_a*hs1_b)/np.sqrt((hs1_a*np.sin(np.deg2rad(hs1_phi-IHS_geodict["azi1"])))**2 + (hs1_b*np.cos(np.deg2rad(hs1_phi-IHS_geodict["azi1"])))**2)/np.sqrt(2)
            hs2_sd = (hs2_a*hs2_b)/np.sqrt((hs2_a*np.sin(np.deg2rad(hs2_phi-IHS_geodict["azi2"])))**2 + (hs2_b*np.cos(np.deg2rad(hs2_phi-IHS_geodict["azi2"])))**2)/np.sqrt(2)
            hi_inter_hs_dis.append(IHS_geodict["a12"])
            hi_azis.append(IHS_geodict["azi1"])
            hi_inter_hs_dis_sds.append(np.sqrt(hs1_sd**2+hs2_sd**2))
            hi_ages.append(age)

        ax_dis.plot(hi_ages,111.113*(np.array(hi_inter_hs_dis)-hi_inter_hs_dis[0]),color=color)
        ax_dis.plot(hi_ages,111.113*(np.array(hi_inter_hs_dis)-hi_inter_hs_dis[0]+2*np.array(hi_inter_hs_dis_sds)),color=color,linestyle=":")
        ax_dis.plot(hi_ages,111.113*(np.array(hi_inter_hs_dis)-hi_inter_hs_dis[0]-2*np.array(hi_inter_hs_dis_sds)),color=color,linestyle=":")
        ax_dis.fill_between(hi_ages, 111.113*(np.array(hi_inter_hs_dis)-hi_inter_hs_dis[0]+2*np.array(hi_inter_hs_dis_sds)), 111.113*(np.array(hi_inter_hs_dis)-hi_inter_hs_dis[0]-2*np.array(hi_inter_hs_dis_sds)), color=color, alpha=0.2,zorder=-1)

        ax_div.plot(hi_ages[:-1],111.113*(np.diff(np.array(hi_inter_hs_dis)-hi_inter_hs_dis[0])/age_step_dis),color=hi_color)
        hi_rate_sds = 111.113*np.sqrt((np.array(hi_inter_hs_dis_sds[:-1])**2 + np.array(hi_inter_hs_dis_sds[1:])**2)/age_step_dis**2)
        print("%s-%s"%(hs1,hs2))
        print("\tHI Max Dis",111.113*(hi_inter_hs_dis[-1] - hi_inter_hs_dis[0]))
        print("\tHI Max Dis Unc",111.113*hi_inter_hs_dis_sds[-1])
        print("\tHI Avg Rate",111.113*((hi_inter_hs_dis[-1] - hi_inter_hs_dis[0])/(hi_ages[-1]-hi_ages[0])))
        print("\tHI Avg. Rate Unc Old Method",(111.113/30.)*np.sqrt((hi_inter_hs_dis_sds[0]**2 + hi_inter_hs_dis_sds[-1]**2)))
        print("\tHI Avg. Rate Unc",111.113*np.sqrt((hi_inter_hs_dis_sds[-1]**2 + hi_inter_hs_dis_sds[0]**2)/((hi_ages[-1]-hi_ages[0])**2)))
        print("\tMean HI Rate Unc",sum(hi_rate_sds)/len(hi_rate_sds)/((hi_ages[-1]-hi_ages[0])))
#        ax_div.plot(hi_ages[:-1],np.diff(np.array(hi_inter_hs_dis)-hi_inter_hs_dis[0])/np.diff(hi_ages) + ,color=hi_color)

    em_inter_hs_dis,em_inter_hs_dis_sds,em_azis,em_ages = [],[],[],[]
    for age in em_ages_iter:
        if len(inv_data[hs1]["EMpols"])==2: age1_dis = ((age-inv_data[hs1]["EMpols"][1])/inv_data[hs1]["EMpols"][0] - (inv_data[hs1]["EM_dated_data"].iloc[0]["Dis"]))/np.sin(np.deg2rad(inv_data[hs1]["EMDis"]))
        elif len(inv_data[hs1]["EMpols"])==3: age1_dis = ((((-inv_data[hs1]["EMpols"][1]+np.sqrt(inv_data[hs1]["EMpols"][1]**2 - 4*inv_data[hs1]["EMpols"][0]*(inv_data[hs1]["EMpols"][2]-age)))/(2*inv_data[hs1]["EMpols"][0]))) - inv_data[hs1]["EM_dated_data"].iloc[0]["Dis"])/np.sin(np.deg2rad(inv_data[hs1]["EMDis"]))
        else: raise ValueError("Cannot determine simple inverse for Emperor age model of %s given model with %d parameters"%(hs1,len(inv_data[hs1]["EMpols"])))
        hs1_geodict = geoid.ArcDirect(inv_data[hs1]["EMLat"],inv_data[hs1]["EMLon"],inv_data[hs1]["EM_start_azi"]+age1_dis,-inv_data[hs1]["EMDis"]) #modeled geographic point for age
        hs1_a = np.sqrt(2)*np.sqrt((inv_data[hs1]["EMsds"][1]**2)*(inv_data[hs1]["EMpols"][1]/inv_data[hs1]["EMpols"][0])**2 + (inv_data[hs1]["EMsds"][0]**2)*(-(age-inv_data[hs1]["EMpols"][1])*inv_data[hs1]["EMpols"][0]**-2)**2)*np.sin(np.deg2rad(inv_data[hs1]["EMDis"]))
        hs1_b = np.sqrt(2)*(33./111.113)
        hs1_phi = (hs1_geodict["azi2"]+90.)
#        print(hs1,inv_data[hs1]["EMLat"],inv_data[hs1]["EMLon"],inv_data[hs1]["EM_start_azi"],age,age1_dis,hs1_geodict["lat2"],hs1_geodict["lon2"],hs1_geodict["azi1"])

        if len(inv_data[hs2]["EMpols"])==2: age2_dis = ((age-inv_data[hs2]["EMpols"][1])/inv_data[hs2]["EMpols"][0] - (inv_data[hs2]["EM_dated_data"].iloc[0]["Dis"]))/np.sin(np.deg2rad(inv_data[hs2]["EMDis"]))
        elif len(inv_data[hs2]["EMpols"])==3: age2_dis = ((((-inv_data[hs2]["EMpols"][1]+np.sqrt(inv_data[hs2]["EMpols"][1]**2 - 4*inv_data[hs2]["EMpols"][0]*(inv_data[hs2]["EMpols"][2]-age)))/(2*inv_data[hs2]["EMpols"][0]))) - inv_data[hs2]["EM_dated_data"].iloc[0]["Dis"])/np.sin(np.deg2rad(inv_data[hs2]["EMDis"]))
        else: raise ValueError("Cannot determine simple inverse for Emperor age model of %s given model with %d parameters"%(hs2,len(inv_data[hs2]["EMpols"])))
        hs2_geodict = geoid.ArcDirect(inv_data[hs2]["EMLat"],inv_data[hs2]["EMLon"],inv_data[hs2]["EM_start_azi"]+age2_dis,-inv_data[hs2]["EMDis"]) #modeled geograpEMc point for age
        hs2_a = np.sqrt(2)*np.sqrt((inv_data[hs2]["EMsds"][1]**2)*(inv_data[hs2]["EMpols"][1]/inv_data[hs2]["EMpols"][0])**2 + (inv_data[hs2]["EMsds"][0]**2)*(-(age-inv_data[hs2]["EMpols"][1])*inv_data[hs2]["EMpols"][0]**-2)**2)*np.sin(np.deg2rad(inv_data[hs2]["EMDis"]))
        hs2_b = np.sqrt(2)*(33./111.113)
        hs2_phi = (hs2_geodict["azi2"]+90.)
#        print(hs2,inv_data[hs2]["EMLat"],inv_data[hs2]["EMLon"],inv_data[hs2]["EM_start_azi"],age,age2_dis,hs2_geodict["lat2"],hs2_geodict["lon2"],hs2_geodict["azi1"])

        if np.isnan(hs1_geodict["lat2"]) or np.isnan(hs1_geodict["lon2"]) or np.isnan(hs2_geodict["lat2"]) or np.isnan(hs2_geodict["lon2"]): continue
        IHS_geodict = geoid.Inverse(hs1_geodict["lat2"],hs1_geodict["lon2"],hs2_geodict["lat2"],hs2_geodict["lon2"])
        hs1_sd = (hs1_a*hs1_b)/np.sqrt((hs1_a*np.sin(np.deg2rad(hs1_phi-IHS_geodict["azi1"])))**2 + (hs1_b*np.cos(np.deg2rad(hs1_phi-IHS_geodict["azi1"])))**2)/np.sqrt(2)
        hs2_sd = (hs2_a*hs2_b)/np.sqrt((hs2_a*np.sin(np.deg2rad(hs2_phi-IHS_geodict["azi2"])))**2 + (hs2_b*np.cos(np.deg2rad(hs2_phi-IHS_geodict["azi2"])))**2)/np.sqrt(2)
        em_inter_hs_dis.append(IHS_geodict["a12"])
        em_azis.append(IHS_geodict["azi1"])
        em_inter_hs_dis_sds.append(np.sqrt(hs1_sd**2+hs2_sd**2))
        em_ages.append(age)

    if hs1!="Rurutu" and hs2!="Rurutu": start_dis = hi_inter_hs_dis[0]
    else: start_dis = em_inter_hs_dis[0]

    ax_dis.plot(em_ages,111.113*(np.array(em_inter_hs_dis)-start_dis),color=color,label="%s-%s"%(hs1,hs2))
    ax_dis.plot(em_ages,111.113*(np.array(em_inter_hs_dis)-start_dis+2*np.array(em_inter_hs_dis_sds)),color=color,linestyle=":")
    ax_dis.plot(em_ages,111.113*(np.array(em_inter_hs_dis)-start_dis-2*np.array(em_inter_hs_dis_sds)),color=color,linestyle=":")
    if "%s-%s"%(hs1,hs2) != "Hawaii-Louisville HK19" and "%s-%s"%(hs1,hs2) != "Louisville HK19-Hawaii": alpha=.1
    else: alpha=.3
    ax_dis.fill_between(em_ages, 111.113*(np.array(em_inter_hs_dis)-start_dis+2*np.array(em_inter_hs_dis_sds)), 111.113*(np.array(em_inter_hs_dis)-start_dis-2*np.array(em_inter_hs_dis_sds)), color=color, alpha=alpha,zorder=-1)


    ax_div.plot(em_ages[:-1],111.113*(np.diff(np.array(em_inter_hs_dis)-em_inter_hs_dis[0])/age_step_dis),color=em_color)
    em_rate_sds = 111.113*np.sqrt((np.array(em_inter_hs_dis_sds[:-1])**2 + np.array(em_inter_hs_dis_sds[1:])**2)/age_step_dis**2)
    print("%s-%s"%(hs1,hs2))
    print("\tEM Max Dis",111.113*(em_inter_hs_dis[-1] - em_inter_hs_dis[0]))
    print("\tEM Max Dis Unc",111.113*em_inter_hs_dis_sds[-1])
    print("\tEM Avg Rate",111.113*((em_inter_hs_dis[-1] - em_inter_hs_dis[0])/(em_ages[-1]-em_ages[0])))
    print("\tEM Avg. Rate Unc Old Method",(111.113/30.)*np.sqrt((em_inter_hs_dis_sds[0]**2 + em_inter_hs_dis_sds[-1]**2)))
    print("\tEM Avg. Rate Unc",111.113*np.sqrt((em_inter_hs_dis_sds[-1]**2 + em_inter_hs_dis_sds[0]**2)/((em_ages[-1]-em_ages[0])**2)))
    print("\tMean EM Rate Unc",sum(em_rate_sds)/len(em_rate_sds)/((em_ages[-1]-em_ages[0])))
    ax_div.set_title("%s-%s"%(hs1,hs2))
    ax_div.set_xlim(0.,80.)
#    ax_div.set_ylim(-10.,25.)

    dis_pos += 1
    div_pos += 1

ax_dis.legend(framealpha=.7)
ax_dis.set_xlabel("Age (Ma)")
ax_dis.set_ylabel("Change in Hotspot Distance Relative to Present (km)")
ax_dis.set_title("Inter-Hotspot Distances")
ax_dis.set_ylim(-1500.,2500.)
ax_dis.set_xlim(0.,80.)

if is_subair:
    fig_dis.savefig("./results/tmp/Bend_IHSD_%ddeg_subair.png"%deg,dpi=200,bbox_inches="tight")
    fig_dis.savefig("./results/tmp/Bend_IHSD_%ddeg_subair.pdf"%deg,dpi=200,bbox_inches="tight")
    fig_div.savefig("./results/tmp/Bend_IHSR_%ddeg_subair.pdf"%deg,dpi=200,bbox_inches="tight")
else:
    fig_dis.savefig("./results/tmp/Bend_IHSD_%ddeg.png"%deg,dpi=200,bbox_inches="tight")
    fig_dis.savefig("./results/tmp/Bend_IHSD_%ddeg.pdf"%deg,dpi=200,bbox_inches="tight")
    fig_div.savefig("./results/tmp/Bend_IHSR_%ddeg.pdf"%deg,dpi=200,bbox_inches="tight")



