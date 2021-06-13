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
src_path = os.path.abspath("/home/kevin/Projects/MidCretaceousHSMotion/src/")
sys.path.append(src_path)
import nhotspot as nhs
from functools import cmp_to_key
from glob import glob
import pyskew.plot_gravity as pg
import cartopy.crs as ccrs


anom_df,hs_dfs = nhs.read_hsloc_excel(sys.argv[1])
cwd = os.path.abspath(os.getcwd())
outdir = os.path.abspath("./data/reconstructions/PaRecFiles")
geoid = Geodesic(6371.,0.)
force_chi2 = False

ages,nhs_files = nhs.write_NHSin(hs_dfs,outdir=outdir)
out_poles = []
outdir = os.path.dirname(nhs_files[0])
os.chdir(src_path)
os.system("./hcorrect_final.sh %s"%outdir)
os.chdir(cwd)

nhs_outfiles = glob(os.path.join(outdir,"*.out"))
nhs_mean_poles = {}
for nhs_outfile in nhs_outfiles:
    age = round(float(os.path.basename(nhs_outfile).split("_")[0])*.01,2)
    with open(nhs_outfile) as fin:
        final_pole = fin.readlines()[-1].split()
        nhs_mean_poles[age] = final_pole
print(nhs_mean_poles)

nhs_logfiles = glob(os.path.join(outdir,"*.log"))
nhs_error_poles = {}
for nhs_logfile in nhs_logfiles:
    age = round(float(os.path.basename(nhs_logfile).split("_")[0])*.01,2)
    with open(nhs_logfile) as fin:
        numhs = int(os.path.basename(nhs_logfile).split("_")[1].split('.')[0].strip('N'))
        idxp1 = 6+6*numhs+2
        idxp2 = idxp1+2
        idxp3 = idxp2+2
        lines = fin.readlines()
        p1 = list(map(float,lines[idxp1].split()))
        p2 = list(map(float,lines[idxp2].split()))
        p3 = list(map(float,lines[idxp3].split()))
        nhs_error_poles[age] = np.array(sorted([p1,p2,p3],key=cmp_to_key(lambda x,y: y[2]-x[2])))
print(nhs_error_poles)


cov_files,rots,red_chi2s = [],[],[]
for age in nhs_mean_poles.keys():
    mean_pole = nhs_mean_poles[age]
    error_poles = nhs_error_poles[age]
    with open("tmp_eig2cov_file.txt",'w+') as fout:
        fout.write("%s %s %s\n"%(mean_pole[0],mean_pole[1],mean_pole[2]))
        fout.write("%.2f %.2f %.2f\n"%(error_poles[0,0],error_poles[0,1],error_poles[0,2]))
        fout.write("%.2f %.2f %.2f\n"%(error_poles[1,0],error_poles[1,1],error_poles[1,2]))
        fout.write("%.2f %.2f %.2f\n"%(error_poles[2,0],error_poles[2,1],error_poles[2,2]))
    cov_file = "./data/reconstructions/PaRecFiles/%d.cov"%int(float(age)*100+.5)
    cov_files.append(cov_file)
    os.system("%s/eig2conreg.o < tmp_eig2cov_file.txt > %s"%(src_path,cov_file))
    with open(cov_file,'r') as fin:
        lines = fin.readlines()
        cov = np.array([list(map(float,lines[15].split())),list(map(float,lines[16].split())),list(map(float,lines[17].split()))])
    rot = Rot(lat=float(mean_pole[0]),lon=float(mean_pole[1]),w=float(mean_pole[2]),age_i=0,age_f=age)
    cov = vs_to_cov(*rot.to_cart(),[cov[0,0],cov[1,1],cov[2,2],cov[0,1],cov[0,2],cov[1,2]])
    rot = Rot(float(mean_pole[0]),float(mean_pole[1]),float(mean_pole[2]),0,age,cov=cov)
    chi2,dof = 0,0
    for hs_df in hs_dfs.values():
        row = hs_df[hs_df["Age"]==age]
        if row.empty: continue
        else: row = row.iloc[0]
        rlat,rlon,razi,_ = rot.rotate(row["Lat"],row["Lon"],azi=row["Azi"])
        present_loc = hs_df.iloc[0]
        geodict = geoid.Inverse(rlat, rlon, present_loc["Lat"],present_loc["Lon"]) #get distance and azimuth between points
        s1_2D_unc = ((row["MajSE"]*row["MinSE"])/np.sqrt((np.cos(np.deg2rad(razi-geodict["azi1"]))*row["MinSE"])**2 + (np.sin(np.deg2rad(razi-geodict["azi1"]))*row["MajSE"])**2))
#        s1_2D_unc = np.sqrt((np.cos(np.deg2rad(razi-geodict["azi1"]))*row["MajSE"])**2 + (np.sin(np.deg2rad(razi-geodict["azi1"]))*row["MinSE"])**2)#/np.sqrt(2)
        pd_s1_2D_unc = ((present_loc["MajSE"]*present_loc["MinSE"])/np.sqrt((np.cos(np.deg2rad(present_loc["Azi"]-geodict["azi2"]))*present_loc["MinSE"])**2 + (np.sin(np.deg2rad(present_loc["Azi"]-geodict["azi2"]))*present_loc["MajSE"])**2))
#        pd_s1_2D_unc = np.sqrt((np.cos(np.deg2rad(present_loc["Azi"]-geodict["azi2"]))*present_loc["MajSE"])**2 + (np.sin(np.deg2rad(present_loc["Azi"]-geodict["azi2"]))*present_loc["MinSE"])**2)#/np.sqrt(2)
        chi2 += (geodict["a12"]**2)/(s1_2D_unc**2+pd_s1_2D_unc**2)
        dof += 2
    red_chi2s.append(chi2/(dof-3))
    if force_chi2:
        rot = Rot(float(mean_pole[0]),float(mean_pole[1]),float(mean_pole[2]),0,age,cov=np.sqrt(red_chi2s[-1])*cov)
    print('------------------------------------------------------')
    print(rot)
    print(rot.to_cart_cov())
    print(red_chi2s[-1],(dof-3)*red_chi2s[-1],(dof-3))
#    if age==48.: import pdb; pdb.set_trace()
    rots.append(rot)
PaHsReconst = PlateReconstruction('PA','HS',rots)

print(PaHsReconst)
print(PaHsReconst.to_df())
PaHsReconst.to_csv("./data/reconstructions/PA_nhotspot_inversion.csv")

#Construct Paper Table
pahs_df = pd.read_csv("./data/reconstructions/PA_nhotspot_inversion.csv", header=1, sep="\t")
manuscript_df = pd.DataFrame(index=pahs_df.index)
manuscript_df["Age (Ma)"] = pahs_df["stop_age"]
manuscript_df["Lat (N$^\circ$)"] = pahs_df["lat"]
manuscript_df["Lon (E$^\circ$)"] = pahs_df["lon"]
manuscript_df["Angle (deg)"] = pahs_df["rot"]
manuscript_df["a"] = pahs_df["vxx"]*1e5
manuscript_df["b"] = pahs_df["vxy"]*1e5
manuscript_df["c"] = pahs_df["vxz"]*1e5
manuscript_df["d"] = pahs_df["vyy"]*1e5
manuscript_df["e"] = pahs_df["vyz"]*1e5
manuscript_df["f"] = pahs_df["vzz"]*1e5
manuscript_df["$\hat{\kappa}$"] = (1/np.array(red_chi2s))
manuscript_df.to_csv("./manuscript/PA_nhotspot_inversion.csv", sep=",", index=False)



for (i,row) in anom_df.iterrows(): #Plot the present day Location and Paleo-Location connected by GC passed in to inversion for each HS

    fig = plt.figure(figsize=(16,9),dpi=200)
    m = pgeo.create_basic_map(projection="moll",center_lon=180)

    if PaHsReconst!=None:
        rot = PaHsReconst[row["Age"]]
        lat,lon = rot.lat,rot.lon
        a,b,azi = cov_to_ellipse(lat,lon,rot.cov[:-1,:-1])
        if lat<0: lat,lon=-lat,(lon+180)%360
        print(lon,lat,azi,2*a,2*b)
        m = psk.plot_pole(lon,lat,azi,2*a,2*b,marker='o',color='purple',m=m,zorder=10)

    for (hs_name,hs_df) in hs_dfs.items():
        present_day = hs_df.iloc[0]
        print(present_day["Lon"],present_day["Lat"],present_day["Azi"],present_day["MajSE"],present_day["MinSE"])
        m = psk.plot_pole(present_day["Lon"],present_day["Lat"],present_day["Azi"],present_day["MajSE"],present_day["MinSE"],m=m,color='r',alpha=.5,s=10)
        try: hs_row = hs_df[hs_df["Age"]==row["Age"]].iloc[0]
        except IndexError: continue
#        print(row["Lon"],row["Lat"],row["Azi"],row["MajSE"],row["MinSE"])
        m = psk.plot_pole(hs_row["Lon"],hs_row["Lat"],hs_row["Azi"],hs_row["MajSE"],hs_row["MinSE"],m=m,color='c',alpha=.5,s=10)
        if PaHsReconst!=None:
            geo1 = Geodesic.WGS84.Inverse(lat,lon,present_day["Lat"],present_day["Lon"])
            geo2 = Geodesic.WGS84.Inverse(lat,lon,hs_row["Lat"],hs_row["Lon"])
            aprox_dis = (geo1["a12"]+geo2["a12"])/2
            azi_list = [(geo1["azi1"]+360)%360,(geo2["azi1"]+360)%360]
            psk.plot_small_circle(lon,lat,aprox_dis,range_azis=(min(azi_list),max(azi_list),.1),m=m,color='m')
    m.set_global()

    plt.title("%.1f Ma"%(row["Age"]))

    fig.savefig("./results/PAReconstruction/PAReconstructionStills/%d.png"%(int(row["Age"])))
    plt.close()

padding = .05
fig = plt.figure(figsize=(16,9),dpi=200)
fig.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.05,hspace=0.12)
ax_pos = 131
hs_names = ["Hawaii","Rurutu","Louisville (Heaton & Koppers 2019)"]
for hs_name in hs_names:
    hs_df = hs_dfs[hs_name]
    present_day = hs_df.iloc[0]
    half_diag = 5

    geo1 = Geodesic.WGS84.ArcDirect(present_day["Lat"],present_day["Lon"],45.0,half_diag)
    geo2 = Geodesic.WGS84.ArcDirect(present_day["Lat"],present_day["Lon"],225.0,half_diag)
    m = pgeo.create_basic_map(projection="merc",fig=fig,ax_pos=ax_pos)

    bend_extent = [geo2["lon2"],geo1["lon2"],geo2["lat2"],geo1["lat2"]]
    decimation = 10
    #Plot Bend Figures
    all_lons,all_lats,all_grav = pg.get_sandwell(bend_extent,decimation,sandwell_files_path="../PySkew/raw_data/gravity/Sandwell/*.tiff")

    print("Plotting Gravity")
    start_time = time()
    print(all_lons.shape,all_lats.shape,all_grav.shape)
#    potental cmaps: cividis
    fcm = m.contourf(all_lons, all_lats, all_grav, cmap="Blues_r", alpha=.75, transform=ccrs.PlateCarree(), zorder=0)
    print("Runtime: ",time()-start_time)

    print(bend_extent)
    m.set_extent(bend_extent, ccrs.PlateCarree())

    for i,row in hs_df.iterrows():
        if row["Age"]==0:
            m = psk.plot_pole(row["Lon"],row["Lat"],row["Azi"],row["MajSE"],row["MinSE"],m=m,color='tab:red',alpha=.5,pole_text=row["Age"],s=10)
        else:
            rlat,rlon,razi,(ra,rb,phi) = PaHsReconst[row["Age"]].rotate(row["Lat"],row["Lon"],row["Azi"],a=row["MajSE"],b=row["MinSE"],phi=row["Azi"])
            m = psk.plot_pole(rlon,rlat,razi,row["MajSE"],row["MinSE"],m=m,color='tab:orange',alpha=.1,pole_text=row["Age"],s=10)

    plt.title(hs_name)
    ax_pos += 1

fig.savefig("./results/PAReconstruction/CheckFit/Misfit.png")
fig.savefig("./results/PAReconstruction/CheckFit/Misfit.pdf")
plt.close()

fig = plt.figure(figsize=(16,9),dpi=200)
m = pgeo.create_basic_map(projection="npstere",stereo_bound_lat=0)

for rot in PaHsReconst.get_rots():
    lat,lon = rot.lat,rot.lon
    a,b,azi = cov_to_ellipse(lat,lon,rot.cov[:-1,:-1])
    if lat<0: lat,lon=-lat,(lon+180)%360
    print("Age:",rot.age_f)
    print(lat,lon,a,b,azi)
    print("----------------------------------------------------------")
    m = psk.plot_pole(lon,lat,azi,2*a,2*b,marker='o',color='c',m=m,label="Pacific Reconstruction Pole %.1f"%rot.age_f,zorder=10,alpha=.3)

fig.savefig("./results/PAReconstruction/poles.png")


fig = plt.figure(figsize=(16,9),dpi=200)
plt.hist(red_chi2s)
fig.savefig("./results/PAReconstruction/hist_red_chi2s.png")

fig = plt.figure(figsize=(16,9),dpi=200)
plt.scatter(anom_df["Age"],red_chi2s)
fig.savefig("./results/PAReconstruction/scatter_red_chi2s.png")

import pdb; pdb.set_trace()

