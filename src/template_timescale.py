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
mpl.rcParams['hatch.linewidth'] = 0.4

geoid = Geodesic(6371.,0.)
#xlims = [-20,20]
#ylims = [-50,70]
xlims = [-20,20]
ylims = [-90,60]
tick_step = 10.
fontsize = 18
padding = 0.05
height = 100

events_file = "/home/kevin/Events_Near_HEB_manuscript.csv"
edf = pd.read_csv(events_file,sep="\t")

timescale_file = "/home/kevin/Projects/PySkew/raw_data/timescale_gradstein2012_all.txt"
tdf = pd.read_csv(timescale_file,sep="\t")

timescale_ages_file = "/home/kevin/Projects/PySkew/raw_data/gradstein_ages.txt"
adf = pd.read_csv(timescale_ages_file,sep="\t")

fig = plt.figure(figsize=(16,9))
fig.subplots_adjust(left=padding,bottom=padding,right=1-padding,top=1-padding,wspace=0.,hspace=0.)
#axs = fig.subplot_mosaic([['bar1', 'patches'], ['bar2', 'patches']])
#ax = axs["bar1"]
ax = fig.add_subplot(1, 1, 1)
plt.tick_params(which="major",labelsize=fontsize, length=8, bottom=True, top=True, labeltop=True)

ylim = ax.get_ylim()
ylim = (ylim[0],height)
ax.axvspan(47.4-1.,47.4+1.,10/(ylim[1]+10),1.,color="tab:red",alpha=.15,zorder=0,label="Coeval Bend Age")
ax.axvline(47.4,10/(ylim[1]+10),.91,color="tab:red",zorder=6,linewidth=4,alpha=.8)

C23_done,C24_done,C18_done = False,False,False
for i,row in tdf.iterrows():
    name = row["chron"]
#    if "r" in name: name = tdf.loc[idx+1]["chron"].iloc[0].lstrip("C").rstrip("n")+"y"
#    else: name = name.lstrip("C").rstrip("n")+"o"
#    if row["top"]<10 or row["top"]>84: continue
    if "n" in row["chron"]: color,oth_color="k","w"
    else: color,oth_color="w","k"
    ax.bar(row["top"],-10.,row["base"]-row["top"],align="edge",color=color,zorder=1000)

#    if name in ["C33r","C33n","C32n.2n","C31r","C30n","C26r","C24r","C21n","C20r","C12r"]:
    if row["base"]>59: continue
    elif row["top"]<39: continue
    if name=="C19n": continue
    elif name.startswith("C24n"):
        if not C24_done:
            C24_done = True
            txt = ax.text((52.62+53.983)/2, -5, "%s"%("24n").strip("C").split(".")[0], color=oth_color, ha="center", va="center", fontsize=fontsize-5,zorder=1001)
    elif name.startswith("C23n"):
        if not C23_done:
            C23_done = True
            txt = ax.text((50.628+51.833)/2, -5, "%s"%("23n").strip("C").split(".")[0], color=oth_color, ha="center", va="center", fontsize=fontsize-5,zorder=1001)
    elif name.startswith("C18n"):
        if not C18_done:
            C18_done = True
            txt = ax.text((38.615+40.145)/2, -5, "%s"%("18n").strip("C").split(".")[0], color=oth_color, ha="center", va="center", fontsize=fontsize-5,zorder=1001)
    else: txt = ax.text((row["base"]+row["top"])/2, -5, "%s"%(name).strip("C").split(".")[0], color=oth_color, ha="center", va="center", fontsize=fontsize-5,zorder=1001)
#        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k", alpha=.7)])

for i,row in adf.iterrows():
    name = row["age"]
#    if "r" in name: name = tdf.loc[idx+1]["chron"].iloc[0].lstrip("C").rstrip("n")+"y"
#    else: name = name.lstrip("C").rstrip("n")+"o"
#    if row["top"]<10 or row["top"]>84: continue
    if name=="Selandian": color = "#FABE6B"
    elif name=="Thanetian": color="#FABE76"
    elif name=="Ypresian": color="#F9A67C"
    elif name=="Lutetian": color="#F9B289"
    elif name=="Bartonian": color="#FABE96"
    elif name=="Priabonian": color="#FBCBA5"
    else: color = "#F9A668"
    oth_color="k"
    ax.bar(row["top"],-10.,row["base"]-row["top"],height,align="edge", edgecolor="k", linewidth=1,color=color,zorder=1000)

#    if name in ["C33r","C33n","C32n.2n","C31r","C30n","C26r","C24r","C21n","C20r","C12r"]:
#    if name=="Thanetian":
#        txt = ax.text(row["base"]+.25, height-5, "%s"%(name).strip("C").split(".")[0], color=oth_color, ha="right", va="center", fontsize=fontsize,zorder=1001)
    if name=="Bartonian":
        txt = ax.text(row["top"]-.75, height-5, "%s"%(name).strip("C").split(".")[0], color=oth_color, ha="left", va="center", fontsize=fontsize,zorder=1001)
    elif name=="Selandian": continue
    else: txt = ax.text((row["base"]+row["top"])/2, height-5, "%s"%(name).strip("C").split(".")[0], color=oth_color, ha="center", va="center", fontsize=fontsize,zorder=1001)

space,space_interval,width,pad=4,5.5,-2.5,.15
for i,row in edf.iterrows():
    name = row["Event"]
#    if name=="Full India-Eurasia Collision": space-=2
    if name=="Bend in Pacific Hotspot Tracks": zorder = 1
    else: zorder = 9
#    if name=="Eastern Pacific Seafloor Spreading Rotates West": hoffset = .29
#    elif name=="Antarctic-Pacific change in plate motion direction": hoffset = .75
#    elif name=="Farallon- and Vancouver-Pacific Seafloor Spreading Doubles": hoffset = .25
#    elif name=="India-Africa and -Antarctica Sea Floor Spreading Rates halved": hoffset = .05
#    elif name=="Smooth-Rough transition across the Carlsberg Ridge": hoffset = .7
#    elif name=="Uplift of New Caladonia and New Zealand": hoffset = .03
#    elif name=="Initial Collision of India and Eurasia": hoffset = .1
#    elif name=="South America-Antarctica Diverge Faster": hoffset = -.3
#    else: hoffset = 0.
    hoffset = 0.
    edge,ha = row["top"],"right"
#    if name=="Hypothesized True Polar Wander": edge,ha = row["base"],"left"
#    else: edge,ha = row["top"],"right"
    color,oth_color,hatch=row["color"],"k",row["hatch"]
    ax.bar(row["top"],width,row["base"]-row["top"],height-10-space,align="edge", edgecolor=oth_color, linewidth=1, color=color, zorder=zorder,hatch=hatch)
#    txt = ax.text((row["base"]+row["top"])/2 + hoffset, height-10-space, "%s"%(name).strip("C").split(".")[0], color=oth_color, ha="center", va="bottom", fontsize=fontsize,zorder=9)
    if name=="Bend in Pacific Hotspot Tracks" or "India-Eurasia Collision" in name:
        txt = ax.text(edge + hoffset + pad, height-10-space+width/2, "%s"%(name).strip("C").split(".")[0], color=color, ha=ha, va="center", fontsize=fontsize,zorder=9)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="k")])
    else:
        txt = ax.text(edge + hoffset + pad, height-10-space+width/2, "%s"%(name).strip("C").split(".")[0], color=oth_color, ha=ha, va="center", fontsize=fontsize,zorder=9)
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="k")])
    space += space_interval


ax.plot([11.056,83.64],[0.,0.],color="k")

ax.set_xlim(60,38)
ax.set_ylim(-10,ylim[1])
ax.set_xlabel("Age (Ma)",fontsize=fontsize)
#ax.set_ylabel("Spreading Rate (mm/a)",fontsize=fontsize)
#ax.yaxis.set_minor_locator(MultipleLocator(5))

ax.yaxis.set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_minor_locator(MultipleLocator(.5))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.tick_params(which='minor', color='k', length=4, top=True)

#ax.legend(facecolor="grey",framealpha=.3,fontsize=14)
#ax.set_title("Pacific Ocean Basin Spreading Rates",fontsize=fontsize+4)

fig.savefig("HEB_timeline_manuscript.pdf",transparent=True,layout="tight", dpi=200)
plt.show()
