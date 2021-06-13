import os,sys
import pandas as pd
import numpy as np
import matplotlib.patheffects as PathEffects
import pyrot.reconstruction as pyrec
from pyrot.rot import cart2latlon,cart2latlonrot,latlon2cart
import matplotlib.pyplot as plt
import matplotlib as mpl
from geographiclib.geodesic import Geodesic
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


mpl.rcParams.update(mpl.rcParamsDefault)
#mpl.use("")
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

geoid = Geodesic(6371.,0.)
#xlims = [-20,20]
#ylims = [-50,70]
xlims = [-20,20]
ylims = [-90,60]
tick_step = 10.
fontsize = 18
#rec_path = "/home/kevin/Projects/DispursionOfHSAges/data/reconstructions/AGHJ06.csv"
#rec_path = "/home/kevin/Projects/DispursionOfHSAges/data/reconstructions/PA_nhotspot_inversion.csv"
timescale_file = "/home/kevin/Projects/PySkew/raw_data/timescale_gradstein2012_all.txt"
tdf = pd.read_csv(timescale_file,sep="\t")
projection_age = 11.0
#recs = [[[-14.37,51.27],"India-Somalia","tab:green","./data/reconstructions/relrec/INSO_EH_rec.csv"],[[-52.42,48.00],"India-East Antarctica","tab:pink","./data/reconstructions/relrec/INAN_rec.csv"],[[-60.68,12.17],"Somalia-East Antarctica","tab:purple","./data/reconstructions/relrec/ANSO_rec.csv"],[[-63.00,120.87],"Australia-East Antarctica","tab:orange","./data/reconstructions/relrec/ANAU_rec.csv"], [[-65.27,-19.96],"South America-East Antarctica","tab:cyan","./data/reconstructions/relrec/SAAN_rec.csv"]]#,[[],"Antarctica-Nubia","black","./data/reconstructions/relrec/ANNB_rec.csv"]]
#recs = [[[-14.37,51.27],"India-Somalia | Cande et al. 2010","#97FA97","./data/reconstructions/relrec/INSO_rec.csv"],[[-14.37,51.27],"India-Somalia | Copley et al. 2010","tab:green","./data/reconstructions/relrec/INSO_CPJV_rec.csv"],[[-14.37,51.27],"India-Somalia | Eagles and Hoang 2013","#004400","./data/reconstructions/relrec/INSO_EH_rec.csv"]]
recs = [[[42.09,177.77],"Vancouver-Pacific","tab:red","./data/reconstructions/relrec/PAVA_EByte_recs.csv"],[[3.18, -153.67],"Farallon-Pacific","tab:blue","./data/reconstructions/relrec/PAFR_EByte_recs.csv"],[[-69.82,-101.487],"West Antarctica-Pacific","white","./data/reconstructions/relrec/PAWAN_EByte_recs.csv"]]
#recs = [[[22.72,-156.63],"Farallon-Pacific Molokai FZ","#021828","./data/reconstructions/relrec/PAFR_EByte_recs.csv"],[[3.18, -153.67],"Farallon-Pacific Clipperton FZ","tab:blue","./data/reconstructions/relrec/PAFR_EByte_recs.csv"],[[-24.40,-151.14],"Farallon-Pacific Austral FZ","#9ACFF5","./data/reconstructions/relrec/PAFR_EByte_recs.csv"]]
#["Pacific-Kula","tab:cyan","./data/reconstructions/relrec/PAKU_EByte_recs_cut.csv"] #OLD PACIFIC KULA LINE
#recs = [[[19+25/60,-(155+17/60)],"Pacific-Hotspots","tab:blue","./data/reconstructions/PA_nhotspot_inversion.csv"]]

#ref_points_latlon = [[-0.134,-128.765],[15.16295753539516,-38.80131405142588],[-74.83642179968396,-38.27054899816777]] #Pacific Data (Azi = 74.837)
#v = np.array([[-0.62612591,  0.75219207,  0.20536172],
#              [-0.77971846, -0.60480635, -0.16201359],
#              [-0.00233874,  0.26156523, -0.96518297]]) #Galapagos FZ Location
#v = np.array([[-0.62612591,  0.660197  , -0.18767058],
#              [-0.77971846, -0.73054364, -0.38538909],
#              [-0.00233874,  0.17448758, -0.90346832]]) #Molokai FZ Location

v = np.eye(3) #Null Rotation
e = np.ones((3,))

#pahs_rec = pyrec.PlateReconstruction.read_csv(recs[0][-1])
#e,v = np.linalg.eig(pahs_rec[projection_age].to_cart_cov())
#print(pahs_rec[projection_age].to_cart_cov())
#idx = e.argsort()[::-1]
#e = e[idx]
#v = v[:,idx]

#    for j,vec in enumerate(v.T):
#        pole,_ = cart2latlon(*vec,np.zeros([3,3]))
#        if j==1 and "AGHJ06" in rec_path:
#            pole = [-pole[0],pole[1]-180]
#            cart,_ = latlon2cart(*pole,np.zeros([3,3]))
#            v[:,j] = cart
#        if j<2 and "PA_nhotspot_inversion" in rec_path:
#            pole = [-pole[0],pole[1]-180]
#            cart,_ = latlon2cart(*pole,np.zeros([3,3]))
#            v[:,j] = cart
#        print(pole)
    #import pdb; pdb.set_trace()
print(np.rad2deg(np.sqrt(e)))
print(v)

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(1, 1, 1)
plt.tick_params(labelsize=fontsize)

#import pdb; pdb.set_trace()

for ref_point,label,color,rec_path in recs:

    print("-----------------------------------------------------------------------")
    print(rec_path)

    try:
        pahs_rec = pyrec.PlateReconstruction.read_csv(rec_path)
    except ValueError:
        pahs_rec = pyrec.PlateReconstruction.read_csv(rec_path,sep=",")

    ################################Eigen Correct output for AGHJ06
    #for pole in [[17.03,218.14],[-51.66,285.35],[-33.15,139.68]]:
    #    cart,_ = latlon2cart(*pole,np.zeros([2,2]))
    #    if pole[0]==17.03: v = cart
    #    else: v = np.vstack([v,cart])
    #v = v.T
    #print(v)

    ################################Eigen Correct output for GWG20
    #for pole in [[19.44,-149.96],[-50.20,145.10],[33.15,-253.29]]:
    #    cart,_ = latlon2cart(*pole,np.zeros([2,2]))
    #    if pole[0]==19.44: v = cart
    #    else: v = np.vstack([v,cart])
    #v = v.T
    #print(v)

    comps = np.array([])
    for i,rot in enumerate(pahs_rec.get_rrots()):
#        if int(rot.age_f%5): continue
#        if rot.age_f<30. or rot.age_f>55.: continue
        rv = np.rad2deg(v @ rot.to_cart())
        print(rot)
        print(rv)
        print(np.sqrt(rv[0]**2 + rv[1]**2 + rv[2]**2))
#        print(rot.age_f,np.rad2deg(rv))
        comp = rv
#        comp[1] = rot.age_f
        try:
            idx = tdf[tdf["base"]==rot.age_f]["chron"].index
            name = tdf.loc[idx]["chron"].iloc[0]
            if "r" in name: name = tdf.loc[idx+1]["chron"].iloc[0].lstrip("C").rstrip("n")+"y"
            else: name = name.lstrip("C").rstrip("n")+"o"
#            txt = ax.text(comp[1], comp[0], "%s"%name, color=color, ha="right", va="bottom", fontsize=fontsize-6)
#    #        txt = ax.text(comp[1], comp[0], r"%.1f Ma"%rot.age_f, color=color, ha="left", va="bottom", fontsize=fontsize-6)
#            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k", alpha=.7)])

            txt = ax.text(comp[1]+.3, comp[2], "%s"%name, color=color, ha="left", va="center", fontsize=fontsize-6)
    #        txt = ax.text(comp[1], comp[2], r"%.1f Ma"%rot.age_f, color=color, ha="left", va="bottom", fontsize=fontsize-6)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k", alpha=.7)])
        except IndexError: pass
        if len(comps)==0: comps = comp
        else: comps = np.vstack([comps,comp])
#        if rot.age_f==47.349:
#            ax.plot(comps[-1,1], comps[-1,0], color="k", marker="o", linestyle="-",zorder=100,mec="k")
#            ax.plot(comps[-1,1], comps[-1,2], color="k", marker="s", linestyle="-",zorder=100,mec="k")

#    comps = comps/np.max(np.abs(comps),axis=0)
#    if "EByte" in rec_path: comps = comps[1:]
    print(comps)

    ##########################################################Plotting

    # Actual plotting
    ax.plot(comps[:,1], comps[:,0], color=color, marker="o", linestyle="-",zorder=2,label=label+" X-Y",mec="k")
    ax.plot(comps[:,1], comps[:,0], color="k", marker=None, linestyle="-",zorder=1,linewidth=3)
    ax.plot(comps[:,1], comps[:,2], color=color, marker="s", linestyle="-",zorder=3,label=label+" Z-Y",mec="k")
    ax.plot(comps[:,1], comps[:,2], color="k", marker=None, linestyle="-",zorder=1,linewidth=3)

#ax.axvspan(47.4-1.,47.4+1.,color="tab:red",alpha=.3,zorder=0)
#ax.axvline(47.4,color="tab:red",zorder=1)

ax.legend(facecolor="grey",framealpha=.3,fontsize=14)

# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position(["data",0.])
ax.spines['bottom'].set_position(["data",0.])

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.set_xticks(list(np.arange(xlims[0],0,tick_step))+list(np.arange(tick_step,xlims[1]+tick_step,tick_step)))
ax.set_yticks(list(np.arange(ylims[0],0,tick_step))+list(np.arange(tick_step,ylims[1]+tick_step,tick_step)))
#ax.set_xlim(xlims)
#ax.set_ylim(ylims)

xlabel = ax.set_xlabel(r"Y",fontsize=fontsize)
#ylabel = ax.set_ylabel(r"$\hat{w}_{max},\hat{w}_{min}$",rotation=0.,fontsize=fontsize)
ticklab = ax.yaxis.get_ticklabels()[0]
trans = ticklab.get_transform()
ax.yaxis.set_label_coords(0., ylims[1] + 1., transform=trans)
ax.text(0.1, ylims[1] + 4.5, r"Z", color="black", transform=trans, ha="left", fontsize=fontsize)
ax.text(0., ylims[1] + 4.5, r",", color="black", transform=trans, ha="center", fontsize=fontsize)
ax.text(-0.1, ylims[1] + 4.5, r"X", color="black", transform=trans, ha="right", fontsize=fontsize)

ticklab = ax.xaxis.get_ticklabels()[0]
trans = ticklab.get_transform()
ax.xaxis.set_label_coords(xlims[1] + 2., 1., transform=trans)
#ax.set_title("Pacific Basin Finite Rotation Vector Endpoints",fontsize=fontsize+4)

    #ticklab = ax.yaxis.get_ticklabels()[0]
    #trans = ticklab.get_transform()
    #ax.yaxis.set_label_coords(0., ylims[1] + 1., transform=trans)

fig.savefig("./results/vector_endpoint_%.1f.png"%projection_age,transparent=True)
fig.savefig("./results/vector_endpoint_%.1f.pdf"%projection_age,transparent=True)
fig.savefig("./results/vector_endpoint_%.1f.svg"%projection_age,transparent=True)



####################################################################RATES

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(1, 1, 1)
plt.tick_params(which="major",labelsize=fontsize, length=8)

for ref_point,label,color,rec_path in recs:

    try:
        pahs_rec = pyrec.PlateReconstruction.read_csv(rec_path)
    except ValueError:
        pahs_rec = pyrec.PlateReconstruction.read_csv(rec_path,sep=",")

    rot_rate,age = [],[]
    for i,rot in enumerate(pahs_rec.get_srots()):
        geodict = geoid.Inverse(*ref_point,rot.lat,rot.lon)
        rad_dis = np.deg2rad(geodict["a12"])
        if "PAVA" in rec_path and rot.age_i==52.62: rot_rate += [np.nan,np.nan]#rot_rate += [abs(83.7895*np.sin(rad_dis)),abs(83.7895*np.sin(rad_dis))]
#        elif "rec" in os.path.basename(rec_path): rot_rate += [abs(111.113*rot.wr*np.sin(rad_dis)),abs(111.113*rot.wr*np.sin(rad_dis))]
#        else: rot_rate += [abs(2*111.113*rot.wr*np.sin(rad_dis)),abs(2*111.113*rot.wr*np.sin(rad_dis))]
        else: rot_rate += [abs(111.113*rot.wr*np.sin(rad_dis)),abs(111.113*rot.wr*np.sin(rad_dis))]
        age += [rot.age_i,rot.age_f]
#        if geodict["azi1"]+90>180 or geodict["azi1"]+90<0: geodict["azi1"] = (180+geodict["azi1"])%360
#        new_ref_geo = geoid.ArcDirect(*ref_point, geodict["azi1"]+90, abs(rot.wr*np.sin(rad_dis)))
#        ref_point = [new_ref_geo["lat2"],new_ref_geo["lon2"]]

        print(rot.age_i,rot.age_f,111.113*rot.wr,rot_rate[-1],geodict["azi1"]+90,ref_point)

        nlat,nlon,_,_ = rot.rotate(*ref_point)
        ref_point = (nlat,nlon)

        print("NEW REF:", ref_point)

    if age[0]==0.: age,rot_rate = age[2:],rot_rate[2:]
    if rot_rate[0]==0.: age,rot_rate = age[1:],rot_rate[1:]

    print(np.array([age,rot_rate]).T)

    ax.plot(age,rot_rate,color=color,label=label,zorder=3)
    ax.plot(age,rot_rate,color="k",linewidth=3,zorder=2)

ylim = ax.get_ylim()
ax.axvspan(47.4-1.,47.4+1.,10/(ylim[1]+10),1.,color="tab:red",alpha=.3,zorder=1,label="Coeval Bend Age")
ax.axvline(47.4,10/(ylim[1]+10),1.,color="tab:red",zorder=5,linewidth=2)

for i,row in tdf.iterrows():
    name = row["chron"]
#    if "r" in name: name = tdf.loc[idx+1]["chron"].iloc[0].lstrip("C").rstrip("n")+"y"
#    else: name = name.lstrip("C").rstrip("n")+"o"
#    if row["top"]<10 or row["top"]>84: continue
    if "n" in row["chron"]: color,oth_color="k","w"
    else: color,oth_color="w","k"
    ax.bar(row["top"],-10.,row["base"]-row["top"],align="edge",color=color,zorder=0)

    if name in ["C33r","C33n","C32n.2n","C31r","C30n","C26r","C24r","C21n","C20r","C12r"]:
        txt = ax.text((row["base"]+row["top"])/2, -5, "%s"%(name).strip("C").split(".")[0], color=oth_color, ha="center", va="center", fontsize=fontsize-7)
#        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k", alpha=.7)])

ax.plot([11.056,83.64],[0.,0.],color="k")

ax.set_xlim(83.64,11.056)
ax.set_ylim(-10,ylim[1])
ax.set_xlabel("Age (Ma)",fontsize=fontsize)
ax.set_ylabel("Spreading Rate (mm/a)",fontsize=fontsize)
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.tick_params(which='minor', color='k', length=4)

ax.legend(facecolor="grey",framealpha=.3,fontsize=14)
ax.set_title("Pacific Ocean Basin Spreading Rates",fontsize=fontsize+4)

fig.savefig("./results/vector_rates_%.1f.png"%projection_age,transparent=True)
fig.savefig("./results/vector_rates_%.1f.pdf"%projection_age,transparent=True)

plt.show()
