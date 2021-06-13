import os,sys
import pandas as pd
import numpy as np
import pyrot.reconstruction as pyrec
from pyrot.rot import cart2latlon,cart2latlonrot,latlon2cart
import pmagpy.ipmag as ipmag
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

xlims = [-10,40]
ylims = [-10,30]
tick_step = 10
fontsize = 24
#rec_path = "/home/kevin/Projects/DispursionOfHSAges/data/reconstructions/AGHJ06.csv"
rec_path = "/home/kevin/Projects/DispursionOfHSAges/data/reconstructions/PA_nhotspot_inversion.csv"
projection_age = 11.0

pahs_rec = pyrec.PlateReconstruction.read_csv(rec_path)

e,v = np.linalg.eig(pahs_rec[projection_age].to_cart_cov())
print(pahs_rec[projection_age].to_cart_cov())
#e, v = np.ones((3,)), np.eye(3)
idx = e.argsort()[::-1]
e = e[idx]
v = v[:,idx]

for j,vec in enumerate(v.T):
    pole,_ = cart2latlon(*vec,np.zeros([3,3]))
    if j==1 and "AGHJ06" in rec_path:
        pole = [-pole[0],pole[1]-180]
        cart,_ = latlon2cart(*pole,np.zeros([3,3]))
        v[:,j] = cart
    if j<2 and "PA_nhotspot_inversion" in rec_path:
        pole = [-pole[0],pole[1]-180]
        cart,_ = latlon2cart(*pole,np.zeros([3,3]))
        v[:,j] = cart
    print(pole)
#import pdb; pdb.set_trace()
print(np.rad2deg(np.sqrt(e)))
print(v)

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

comps = np.array([0,0,0])
for i,rot in enumerate(pahs_rec.get_rots()):
#    if int(rot.age_f%5): continue
    rv = v @ rot.to_cart()
    print(rot.age_f,np.rad2deg(rv))
    comp = np.rad2deg(rv)
    comps = np.vstack([comps,comp])


##########################################################Plotting
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(1, 1, 1)
plt.tick_params(labelsize=fontsize)

# Actual plotting
ax.plot(comps[:,1], comps[:,0], color="tab:blue", marker="o", linestyle="-")
ax.plot(comps[:,1], comps[:,2], color="tab:orange", marker="s", linestyle="-")

for age in np.arange(0,90,10):
    #lrot = pahs_rec.get_rots()[0]
    lrot = pahs_rec[age]
    lcomp = np.rad2deg(v @ lrot.to_cart())
    lcov = v.T @ lrot.to_cart_cov() @ v

    #x_vec,y_vec = lcov[:,0],lcov[:,1]
    le,lv = np.linalg.eig(lcov[0:2,0:2])
    idx = le.argsort()[::-1]
    a,b = ((180/np.pi)**2)*le[idx]
    maj_ax = lv[:,idx][:,0]
    azi = np.rad2deg(np.arctan2(maj_ax[1],maj_ax[0]))
    print(lcomp[1], lcomp[0],a,b,azi)
    ell = mpl.patches.Ellipse((lcomp[1], lcomp[0]), width=a, height=b, angle=azi, fill=True, facecolor="tab:blue", alpha=.3, edgecolor='#000000', zorder=0)
    ax.add_artist(ell)

    le,lv = np.linalg.eig(lcov[1:3,1:3])
    idx = le.argsort()[::-1]
    a,b = ((180/np.pi)**2)*le[idx]
    maj_ax = lv[:,idx][:,0]
    azi = np.rad2deg(np.arctan2(maj_ax[1],maj_ax[0]))
    print(lcomp[1],lcomp[2],a,b,azi)
    ell = mpl.patches.Ellipse((lcomp[1], lcomp[2]), width=a, height=b, angle=azi, fill=True, facecolor="tab:orange", alpha=.3, edgecolor='#000000', zorder=0)
    ax.add_artist(ell)


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

xlabel = ax.set_xlabel(r"$\hat{w}_{int}$",fontsize=fontsize)
#ylabel = ax.set_ylabel(r"$\hat{w}_{max},\hat{w}_{min}$",rotation=0.,fontsize=fontsize)
ticklab = ax.yaxis.get_ticklabels()[0]
trans = ticklab.get_transform()
ax.yaxis.set_label_coords(0., ylims[1] + 1., transform=trans)
ax.text(0.1, ylims[1] + 1.5, r"$\hat{w}_{max}$", color="tab:blue", transform=trans, ha="left", fontsize=fontsize)
ax.text(0., ylims[1] + 1.5, r",", color="black", transform=trans, ha="center", fontsize=fontsize)
ax.text(-0.1, ylims[1] + 1.5, r"$\hat{w}_{min}$", color="tab:orange", transform=trans, ha="right", fontsize=fontsize)

ticklab = ax.xaxis.get_ticklabels()[0]
trans = ticklab.get_transform()
ax.xaxis.set_label_coords(xlims[1] + 2., 1., transform=trans)

#ticklab = ax.yaxis.get_ticklabels()[0]
#trans = ticklab.get_transform()
#ax.yaxis.set_label_coords(0., ylims[1] + 1., transform=trans)

fig.savefig("./results/vector_endpoint_%.1f.png"%projection_age,transparent=True)
fig.savefig("./results/vector_endpoint_%.1f.pdf"%projection_age,transparent=True)
plt.show()
