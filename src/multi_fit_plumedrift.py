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
from pyrot.reconstruction import PlateReconstruction
from pyrot.rot import Rot,get_pole_arc_misfit_uncertainty,fast_fit_circ,plot_error_from_points
import pyrot.max as pymax
from multi_circ_inv import fit_circ
from functools import reduce
from time import time
from scipy.interpolate import PchipInterpolator
import pmagpy.pmag as pmag
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

data_path = sys.argv[1]
inv_data = pd.read_csv(data_path,index_col=0,dtype={"BendLat":float,"BendLon":float,"EMDis":float,"EMLon":float,"EMLat":float,"EMErr":float,"EM_Start_Dis":float,"EM_start_azi":float,"EMpols":object,"EMsds":object,"HIDis":float,"HILon":float,"HILat":float,"HIErr":float,"HI_Start_Dis":float,"HI_start_azi":float,"HIpols":object,"HIsds":object}).T
for hs in inv_data.columns:
    inv_data[hs]["HIpols"] = list(map(float,inv_data[hs]["HIpols"].strip("[ , ]").replace(",","").split()))
    inv_data[hs]["HIsds"] = list(map(float,inv_data[hs]["HIsds"].strip("[ , ]").replace(",","").split()))
    inv_data[hs]["EMpols"] = list(map(float,inv_data[hs]["EMpols"].strip("[ , ]").replace(",","").split()))
    inv_data[hs]["EMsds"] = list(map(float,inv_data[hs]["EMsds"].strip("[ , ]").replace(",","").split()))
if "Louisville" in inv_data.index: inv_data.drop("Louisville",axis=1,inplace=True)
if "Louisville (Heaton & Koppers 2019)" in inv_data.index: inv_data.drop("Louisville (Heaton & Koppers 2019)",axis=1,inplace=True)
geoid = Geodesic(6371000.,0.)
hi_color = "#C4A58E"#plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
em_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
tr_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
undated_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
mean_color = "k"#plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
#pc_paths = ["GMHRF.csv"]
#pc_paths = ["PA_nhotspot_inversion.csv","Koiv2014.csv","WK08-A.csv","WK08-D.csv","GMHRF.csv","B2014.csv"]
pc_paths = ["PA_nhotspot_inversion.csv","Koiv2014.csv","WK08-A.csv","GMHRF.csv"]
#pc_paths = ["WK08-D.csv","GMHRF.csv","B2014.csv"]
#bend_age = 48.
decimation = 100.
#norm = plt.Normalize(0.,80.,5.)
N = 17
an_fontsize = 22
fontsize = 22
label_fontsize = 32
markersize = 150
color_80Ma = "#fcfdbf"
GMHRF_smoothing = 1
norm = colors.BoundaryNorm(boundaries=np.linspace(0.,80.,N), ncolors=256)
land_resolution = "10m"
age_step = .1
additional_plume_motion_only = False

padding = .075
dfig=plt.figure(figsize=(9,9),dpi=200)
padding = 0.05
dfig.subplots_adjust(left=padding+.05,bottom=padding+.025,right=1-padding,top=1-padding,wspace=0.05,hspace=0.0)
rfig=plt.figure(figsize=(9,9),dpi=200)
rfig.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.05,hspace=0.05)

padding = .1
fig=plt.figure(figsize=(6*len(pc_paths),18),dpi=200)
fig.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.01,hspace=0.10)
if len(pc_paths)>9: raise RuntimeError("This figure is not designed to show that many hotspot model comparisions and will require a redesign")
for i,pc_path in enumerate(pc_paths):
    print(pc_path)
    pc_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
    axpos=i+1
    if pc_path=="PA_nhotspot_inversion.csv": sep = "\t"
    else: sep = ","
    pahs_rec = PlateReconstruction.read_csv(os.path.join(".","data","reconstructions",pc_path),sep=sep)
    rax_pos,dax_pos = 311,311
    for hs in ["Hawaii","Rurutu","Louisville HK19"]:
        print(hs)
        rax = rfig.add_subplot(rax_pos)
        dax = dfig.add_subplot(dax_pos)
        if "GMHRF" in pc_path and not additional_plume_motion_only:
            if hs=="Hawaii": hs_end_age,bend_age,extent,clat,clon,hs_lat,hs_lon = 78.,47.3,[-159.5-3.,-141.-.5-5.5-3.,15.5,30.5],20.,-155.,19+25/60,-(155+17/60)
        else:
            if hs=="Hawaii": hs_end_age,bend_age,extent,clat,clon,hs_lat,hs_lon = 78.,47.3,[-161.25-.2,-150.-.05,15.,26.2],20.,-155.,19+25/60,-(155+17/60)
        if hs=="Rurutu": hs_end_age,bend_age,extent,clat,clon,hs_lat,hs_lon = 72.,41.5,[-155.3,-145.-.05,-27.,-17.],-22.5,(-160.-2.5+-150.+2.5)/2,-(22+28/60),-(151+20/60)
        elif "Louisville" in hs:
            hs_end_age,bend_age,extent,clat,clon = 80.,49.3,[-145.,-130.-.05,-55.,-45.],-52.5,-145.+-130./2
            if pc_path=="Koiv2014.csv": hs_lat,hs_lon = -50.9,221.9 #Lonsdale Location
            elif "WK08" in pc_path: hs_lat,hs_lon = -(52+24/60),-137.2
            elif pc_path=="GMHRF.csv": hs_lat,hs_lon = -50.9,-138.1
            lou1_lat,lou1_lon = -50.44,-139.15

        hs_data = pd.read_excel(sys.argv[2],hs)

        #Plot Bend Figures
        proj = ccrs.Mercator(central_longitude=180.)
#        proj = ccrs.Stereographic(central_latitude=clat, central_longitude=clon)
        print(axpos)
        m = fig.add_subplot(3,len(pc_paths),axpos,projection=proj)
        m.set_xticks(np.arange(0, 365, 5.), crs=ccrs.PlateCarree())
        m.set_yticks(np.arange(-80, 85, 5.), crs=ccrs.PlateCarree())
        m.tick_params(grid_linewidth=0.5,grid_linestyle=":",color="black",labelsize=8,tickdir="in",left=(pc_path==pc_paths[0]),top=False)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        m.xaxis.set_major_formatter(lon_formatter)
        m.yaxis.set_major_formatter(lat_formatter)
#        m.gridlines()
        m.outline_patch.set_linewidth(0.5)
        m.coastlines(linewidth=2,color="k",resolution=land_resolution)
#        m.annotate(chr(64+axpos%10)+")",xy=(1-0.04,0.04),xycoords="axes fraction",bbox=dict(boxstyle="round", fc="w",alpha=.7),va="bottom",ha="right")


        #Handle tick marks
        if axpos%len(pc_paths)==0 or (hs=="Hawaii" and "GMHRF.csv" in pc_paths and axpos%len(pc_paths)==3 and not additional_plume_motion_only):
            if hs=="Hawaii": m.set_anchor('E')
            plt.tick_params(
            axis='both',          # changes the axis to apply changes
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge
            top=True,         # ticks along the top edge
            left=True,         # ticks along the left edge
            right=True,         # ticks along the right edge
            labelbottom=True, # labels along the bottom edge are var
            labelleft=False, # labels along the left edge are on
            labelright=True, # labels along the right edge are off
            tickdir="in",
            length=10,
            width=1,
            labelsize=fontsize) #changes direction of ticks
        else:
            plt.tick_params(
            axis='both',          # changes the axis to apply changes
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge
            top=True,         # ticks along the top edge
            left=True,         # ticks along the left edge
            right=True,         # ticks along the right edge
            labelbottom=True, # labels along the bottom edge are var
            labelleft=False, # labels along the left edge are on
            labelright=False, # labels along the right edge are off
            tickdir="in",
            length=10,
            width=1,
            labelsize=fontsize) #changes direction of ticks

        all_lons,all_lats,all_grav = pg.get_sandwell(extent,decimation,sandwell_files_path="../PySkew/raw_data/gravity/Sandwell/*.tiff")

        print("Plotting Gravity")
        start_time = time()
        print(all_lons.shape,all_lats.shape,all_grav.shape)
    #    potental cmaps: cividis
        fcm = m.contourf(all_lons, all_lats, all_grav, cmap="Blues_r", alpha=.75, transform=ccrs.PlateCarree(), zorder=0)
        print("Runtime: ",time()-start_time)

        ages,lons,lats,rlons,rlats,used_ages,prev_rlat,prev_rlon,dists,rates,tdist = np.arange(0.,min([80.,pahs_rec.to_df()["stop_age"].max()]),age_step),[],[],[],[],[],None,None,[],[],0.

        if "GMHRF" in pc_path:
            geodict = Geodesic(6371.,0.).Inverse(hs_lat,hs_lon,hs_data.iloc[0]["Lat"],hs_data.iloc[0]["Lon"])
            pgeo = Geodesic(6371.,0.).ArcDirect(hs_lat,hs_lon,geodict["azi1"]-90.,90.)
            r = Rot(pgeo["lat2"],pgeo["lon2"],geodict["a12"])
            hi_data,lv_data = {"age":[],"lat":[],"lon":[],"dis":[],"rate":[],"tlat":[],"tlon":[],"dist":[]},{"age":[],"lat":[],"lon":[],"dis":[],"rate":[],"tlat":[],"tlon":[],"dist":[]}
            with open("data/GMHRFPlumeMotions.txt","r") as fin:
                 for line in fin.readlines():
                     record = line.strip("\n").split()
                     if record[0]==hs.split()[0]:
                        if float(record[1])>ages[-1]: continue
                        print(line)
                        mp_lat,mp_lon,_,_ = r.rotate(float(record[2]),float(record[3]))
                        hi_data["age"].append(float(record[1]))
                        hi_data["lat"].append(mp_lat)
                        hi_data["lon"].append(mp_lon)
                        tlon,tlat = proj.transform_point(hi_data["lon"][-1],hi_data["lat"][-1],ccrs.PlateCarree())
                        hi_data["tlon"].append(tlon),hi_data["tlat"].append(tlat)
                        hi_data["dis"].append(geoid.Inverse(hs_lat,hs_lon,mp_lat,mp_lon)["s12"]/1000)
                        if len(hi_data["lat"])>1:
                            hi_data["dist"].append(geoid.Inverse(hi_data["lat"][-2],hi_data["lat"][-2],hi_data["lat"][-1],hi_data["lat"][-1])["a12"])
                            hi_data["rate"].append(111.113*(hi_data["dis"][-1]/(hi_data["age"][-1]-hi_data["age"][-2])))
                        else:
                            hi_data["dist"].append(0.)
                            hi_data["rate"].append(0.)
            hi_df = pd.DataFrame(hi_data)
#                     elif record[0]=="Louisville":
#                        print(line)
#                        lv_data["age"].append(float(record[1]))
#                        lv_data["lat"].append(float(record[2]))
#                        lv_data["lon"].append(float(record[3]))
#                        lv_data["tlon"],lv_data["tlat"] = proj.transform_point(lv_data["lon"],lv_data["lat"],ccrs.PlateCarree())
#                        if len(lv_data["lat"])>1:
#                            lv_data["dis"].append(geoid.Inverse(lv_data["lat"][-2],lv_data["lat"][-2],lv_data["lat"][-1],lv_data["lat"][-1])["a12"])
#                            lv_data["rate"].append(111.113*(lv_data["dis"][-1]/(lv_data["age"][-1]-lv_data["age"][-2])))
#                        else:
#                            lv_data["dis"].append(0.)
#                            lv_data["rate"].append(0.)

        for age in ages:
            if age<=inv_data[hs]["BendAge"]: #Do Hawaiian
                if hs=="Rurutu" and age>10: continue
                if len(inv_data[hs]["HIpols"])==2: age1_dis = ((age-inv_data[hs]["HIpols"][1])/inv_data[hs]["HIpols"][0] - inv_data[hs]["HI_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs]["HIDis"]))
                elif len(inv_data[hs]["HIpols"])==3: age1_dis = ((((-inv_data[hs]["HIpols"][1]+np.sqrt(inv_data[hs]["HIpols"][1]**2 - 4*inv_data[hs]["HIpols"][0]*(inv_data[hs]["HIpols"][2]-age)))/(2*inv_data[hs]["HIpols"][0]))) - inv_data[hs]["HI_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs]["HIDis"]))
                else: raise ValueError("Cannot determine simple inverse for Hawaiian age model of %s given model with %d parameters"%(hs,len(inv_data[hs]["HIpols"])))
                hs_geodict = geoid.ArcDirect(inv_data[hs]["HILat"],inv_data[hs]["HILon"],inv_data[hs]["HI_start_azi"]+age1_dis,-inv_data[hs]["HIDis"]) #modeled geographic point for age
                if "GMHRF" in pc_path and hs!="Rurutu" and additional_plume_motion_only:
                    idx_nearest = np.argmin((hi_df["age"]-age)**2)
                    geodict = Geodesic(6371.,0.).Inverse(hi_df["lat"][idx_nearest],hi_df["lon"][idx_nearest],hs_data.iloc[0]["Lat"],hs_data.iloc[0]["Lon"])
                    pgeo = Geodesic(6371.,0.).ArcDirect(hi_df["lat"][idx_nearest],hi_df["lon"][idx_nearest],geodict["azi1"]-90.,90.)
                    rot = (pahs_rec[age]) + Rot(pgeo["lat2"],pgeo["lon2"],geodict["a12"])
                else: rot = (pahs_rec[age])
                rlat,rlon,_,_ = rot.rotate(hs_geodict["lat2"],hs_geodict["lon2"])
                trlon,trlat = proj.transform_point(rlon,rlat,ccrs.PlateCarree())
                tlon,tlat = proj.transform_point(hs_geodict["lon2"],hs_geodict["lat2"],ccrs.PlateCarree())
                lons.append(tlon),lats.append(tlat)
                rlons.append(trlon),rlats.append(trlat)
                used_ages.append(age)
            else: #Do Emperor
                if len(inv_data[hs]["EMpols"])==2: age1_dis = ((age-inv_data[hs]["EMpols"][1])/inv_data[hs]["EMpols"][0] - inv_data[hs]["EM_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs]["EMDis"]))
                elif len(inv_data[hs]["EMpols"])==3: age1_dis = ((((-inv_data[hs]["EMpols"][1]+np.sqrt(inv_data[hs]["EMpols"][1]**2 - 4*inv_data[hs]["EMpols"][0]*(inv_data[hs]["EMpols"][2]-age)))/(2*inv_data[hs]["EMpols"][0]))) - inv_data[hs]["EM_Start_Dis"])/np.sin(np.deg2rad(inv_data[hs]["EMDis"]))
                else: raise ValueError("Cannot determine simple inverse for Hawaiian age model of %s given model with %d parameters"%(hs,len(inv_data[hs]["EMpols"])))
                hs_geodict = geoid.ArcDirect(inv_data[hs]["EMLat"],inv_data[hs]["EMLon"],inv_data[hs]["EM_start_azi"]+age1_dis,-inv_data[hs]["EMDis"]) #modeled geographic point for age
                if "GMHRF" in pc_path and hs!="Rurutu" and additional_plume_motion_only:
                    idx_nearest = np.argmin((hi_df["age"]-age)**2)
                    geodict = Geodesic(6371.,0.).Inverse(hi_df["lat"][idx_nearest],hi_df["lon"][idx_nearest],hs_data.iloc[0]["Lat"],hs_data.iloc[0]["Lon"])
                    pgeo = Geodesic(6371.,0.).ArcDirect(hi_df["lat"][idx_nearest],hi_df["lon"][idx_nearest],geodict["azi1"]-90.,90.)
                    rot = (pahs_rec[age]) + Rot(pgeo["lat2"],pgeo["lon2"],geodict["a12"])
                else: rot = (pahs_rec[age])
                rlat,rlon,_,rell = rot.rotate(hs_geodict["lat2"],hs_geodict["lon2"],a=(np.sqrt(2)*33.)/111.113,b=(np.sqrt(2)*33.)/111.113,phi=10.)
                if age==ages[-1]: rell_last = rell
                trlon,trlat = proj.transform_point(rlon,rlat,ccrs.PlateCarree())
                tlon,tlat = proj.transform_point(hs_geodict["lon2"],hs_geodict["lat2"],ccrs.PlateCarree())
                lons.append(tlon),lats.append(tlat)
                rlons.append(trlon),rlats.append(trlat)
                used_ages.append(age)
            if "GMHRF" in pc_path and hs!="Rurutu" and not additional_plume_motion_only:
                idx_nearest = np.argmin((hi_df["age"]-age)**2)
                disp_geodict = geoid.Inverse(hi_df["lat"][idx_nearest],hi_df["lon"][idx_nearest],rlat,rlon)
            else: disp_geodict = geoid.Inverse(hs_data["Lat"][0],hs_data["Lon"][0],rlat,rlon)
            if prev_rlat!=None and prev_rlon!=None:
                dgeodict = geoid.Inverse(prev_rlat,prev_rlon,rlat,rlon)
                tdist += dgeodict["a12"]
#                if "GMHRF" in pc_path and hs!="Rurutu" and int(age*10)%GMHRF_smoothing==0: rates.append((sum(111.113*disp_geodict["a12"]-np.array(dists[-GMHRF_smoothing:]))/GMHRF_smoothing)/(age_step))
#                if "GMHRF" in pc_path and hs!="Rurutu" and int(age*10)%GMHRF_smoothing==0: rates.append((111.113*disp_geodict["a12"]-sum(dists[-GMHRF_smoothing:])/GMHRF_smoothing)/(GMHRF_smoothing*age_step))
                if "GMHRF" in pc_path and hs!="Rurutu":
                    nrate = (111.113*disp_geodict["a12"]-dists[-1])/age_step
                    if len(rates)>1 and 20<abs(nrate-rates[-1]): nrate = rates[-1]
                    rates.append(nrate)
                else: rates.append((111.113*disp_geodict["a12"]-dists[-1])/age_step)
            dists.append(111.113*disp_geodict["a12"])
            prev_rlat,prev_rlon = rlat,rlon

        points = np.array([rlons,rlats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='magma', norm=norm)
        lc.set_array(np.array(used_ages))
        lc.set_linewidth(10)
        lc.set_alpha(0.7)
        line = m.add_collection(lc)

        #Make 80Ma ellipse
        psk.plot_pole(rlon, rlat, rell_last[2], 2*1.02371239802553*rell_last[0], 2*1.02371239802553*rell_last[1], color=color_80Ma, alpha=.3, zorder=1, m=m)

        if "GMHRF" in pc_path and not additional_plume_motion_only:
            points = np.array([hi_data["tlon"],hi_data["tlat"]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='magma', norm=norm, linestyle="-")
            lc.set_array(np.array(hi_data["age"]))
            lc.set_linewidth(7)
            line = m.add_collection(lc)
#            import pdb; pdb.set_trace()
            m.plot(points[:,:,0],points[:,:,1],color="k",linestyle="-",linewidth=3)

#        if "GMHRF" in pc_path:
#            linestyle="-"
        linestyle = "-"

#        if "GMHRF" in pc_path:
#            if hs!="Rurutu":
#                
##                dax.plot(hi_data["age"],hi_data["dis"],label=label,color="k",linestyle="-",linewidth=1.0)
#            else:

        rates = np.convolve(rates, np.ones((11,))/11, mode='same') #10 point running average smoothing
        if hs=="Rurutu":
            ca_idx = int(10./age_step)
            if pc_path=="PA_nhotspot_inversion.csv": label = "GWG20"
            elif pc_path=="Koiv2014.csv": label = "KAG14"
            elif pc_path=="GMHRF.csv": label = "GMHRF Misfit"
            else: label = pc_path.split(".")[0]
            rax.plot(used_ages[:ca_idx],rates[:ca_idx],label=label,color=pc_color,linestyle=linestyle)
            dax.plot(used_ages[:ca_idx],dists[:ca_idx],label=label,color=pc_color,linestyle=linestyle)
            rax.plot(used_ages[ca_idx+2:],rates[ca_idx+1:],color=pc_color,linestyle=linestyle)
            dax.plot(used_ages[ca_idx+1:],dists[ca_idx+1:],color=pc_color,linestyle=linestyle)
            if "GMHRF" in pc_path: dax.plot([],[],color=pc_color,linestyle=":",linewidth=1.5,label="GMHRF Predicted")
            rax.set_ylabel("Rate of Plume Motion (mm/a)")
            dax.set_ylabel("Additional Implied Plume Drift (km)",fontsize=fontsize)
            if pc_path==pc_paths[-1]:
                rax.legend(loc=(12/80,.05),fontsize=fontsize-2)
                dax.legend(loc=(12/80,.05),fontsize=fontsize-2)
        else:
            if "GMHRF" in pc_path:
                dax.plot(hi_data["age"],hi_data["dis"],color=pc_color,linestyle=":",linewidth=1.5,label="GMHRF Predicted")
                rax.plot(used_ages[::GMHRF_smoothing][1:],rates,color=pc_color,linestyle=linestyle)
            else: rax.plot(used_ages[1:],rates,label=pc_path,color=pc_color,linestyle=linestyle)
            dax.plot(used_ages[:],dists,label=pc_path,color=pc_color,linestyle=linestyle)
        rax.set_ylim(-50.,50.)
        rax.set_xlim(0.,80.)
        dax.set_xlim(0.,80.)
        if "Louisville" in hs: pass
        else: dax.axes.xaxis.set_ticklabels([]); rax.axes.xaxis.set_ticklabels([])
        dax.tick_params(labelsize=fontsize)
#        if pc_path==pc_paths[0] and hs_name=="Hawaii":
        if pc_path==pc_paths[0]:
#            if hs=="Hawaii": hs_name = "Hawaiian-Emperor"
            if "Louisville" in hs: hs_name = "Louisville"
            else: hs_name = hs
            dax.annotate(hs_name,xy=(0.04,1-0.04),xycoords="axes fraction",bbox=dict(boxstyle="square", fc="w",alpha=.7),va="top",ha="left",fontsize=label_fontsize)
            rax.annotate(hs_name,xy=(0.04,1-0.04),xycoords="axes fraction",bbox=dict(boxstyle="square", fc="w",alpha=.7),va="top",ha="left",fontsize=label_fontsize)


        print("Mean Rate of Motion: ",np.mean(rates))
        print("Total Distance: ", tdist)
        if "GMHRF" in pc_path and hs!="Rurutu": print("Total Displacement: ", dists[-1]+hi_data["dis"][-1])
        else: print("Total Displacement: ", dists[-1])

        if hs=="Hawaii":
            m.scatter(hs_lon,hs_lat,color="tab:purple",edgecolors="k",marker="D",transform=ccrs.PlateCarree(),s=markersize,zorder=5)
            txt = m.text(hs_lon+.4,hs_lat," Kilauea",va="center",ha="left",transform=ccrs.PlateCarree(),color="k",fontsize=an_fontsize)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="grey", alpha=.7)])
        elif hs=="Rurutu":
            m.scatter(-150.730,-23.440,color="tab:brown",edgecolors="k",marker="s",transform=ccrs.PlateCarree(),s=markersize,zorder=5)
            txt = m.text(-150.73,-23.7,"Arago",va="top",ha="center",transform=ccrs.PlateCarree(),color="k",fontsize=an_fontsize)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="grey", alpha=.7)])
            m.scatter(hs_lon,hs_lat,color="tab:brown",edgecolors="k",marker="D",transform=ccrs.PlateCarree(),s=markersize,zorder=5)
            txt = m.text(hs_lon+.4,hs_lat,"Rurutu",va="center",ha="left",transform=ccrs.PlateCarree(),color="k",fontsize=an_fontsize)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="grey", alpha=.7)])
        elif "Louisville" in hs: #Work on getting all locations and citations
            #WK08 Location
            m.scatter(-137.2,-(52+24/60),color="tab:pink",edgecolors="k",marker="s",transform=ccrs.PlateCarree(),s=markersize,zorder=5)
            txt = m.text(-138.0+2.,-(52+24/60)-.5,"Wessel &     \nKronke (2008)",va="center",ha="right",transform=ccrs.PlateCarree(),color="k",fontsize=an_fontsize)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="grey", alpha=.7)])
            #Lonsdale Location
            m.scatter(-138.1,-50.9,color="tab:pink",edgecolors="k",marker="D",transform=ccrs.PlateCarree(),s=markersize,zorder=5)
            txt = m.text(-138.1-.5,-51.1-.25,"Lonsdale\n(1988)",va="center",ha="right",transform=ccrs.PlateCarree(),color="k",fontsize=an_fontsize)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="grey", alpha=.7)])
            #1 Ma seamount - Koppers et al. (2004)
            m.scatter(lou1_lon,lou1_lat,color="tab:pink",edgecolors="k",marker="X",transform=ccrs.PlateCarree(),s=markersize,zorder=5)
            txt = m.text(lou1_lon-.4,lou1_lat,"LOU-2",va="center",ha="right",transform=ccrs.PlateCarree(),color="k",fontsize=an_fontsize)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="grey", alpha=.7)])
        else: raise ValueError("hotspot %s not known"%str(hs))
        row = hs_data.iloc[0]
        m = psk.plot_pole(row["Lon"],row["Lat"],row["Azi"],2*row["MajSE"],2*row["MinSE"],m=m,color='tab:red',alpha=.4,s=markersize, zorder=1)

        if hs=="Hawaii":
            if pc_path=="PA_nhotspot_inversion.csv": title = "GWG20"
            elif pc_path=="Koiv2014.csv": title = "KAG14"
            elif pc_path=="GMHRF.csv": title = "GMHRF"
            else: title = pc_path.split(".")[0]
            m.set_title(title,fontsize=label_fontsize)
        if pc_path==pc_paths[0]: 
#            if hs=="Hawaii": hs = "Hawaiian-Emperor"
            if "Louisville" in hs: hs = "Louisville"
            m.set_ylabel(hs,fontsize=label_fontsize)

        print(extent)
        m.set_extent(extent, ccrs.PlateCarree())
        axpos += len(pc_paths)
        rax_pos += 1
        dax_pos += 1

fig.subplots_adjust(right=0.82)
cbar_ax = fig.add_axes([1-0.125, padding, 0.05, 0.8])
cbar = fig.colorbar(line, cax=cbar_ax, format="%.0f Ma")
cbar_ax.set_title("Age",fontsize=label_fontsize)
cbar_ax.tick_params(labelsize=20)
cbar.ax.tick_params(labelsize=24)

rax.set_xlabel("Age (Ma)",fontsize=fontsize)
dax.set_xlabel("Age (Ma)",fontsize=fontsize)


fig.savefig("results/%s_plumedrift.png"%os.path.basename(data_path).split(".")[0])
fig.savefig("results/%s_plumedrift.pdf"%os.path.basename(data_path).split(".")[0])
rfig.savefig("results/%s_plumedrift_rate.pdf"%os.path.basename(data_path).split(".")[0])
dfig.savefig("results/%s_plumedrift_dist.pdf"%os.path.basename(data_path).split(".")[0])
#plt.show()

