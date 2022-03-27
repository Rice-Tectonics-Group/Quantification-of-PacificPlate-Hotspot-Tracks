import os
import subprocess
import pandas as pd
import numpy as np
import pmagpy.ipmag as ipmag
from pyrot.rot import Rot,ellipse_to_cov,cov_to_ellipse,latlon2cart,cart2latlon
from pyrot.reconstruction import PlateReconstruction
import pyskew.utilities as utl
from geographiclib.geodesic import Geodesic
from scipy.optimize import brute,fmin,least_squares
import multiprocessing
from functools import reduce

def read_hsloc_excel(path):
    hs_xls = pd.ExcelFile(path)
    anom_df,hs_dfs = None,{}
    for sheet in hs_xls.sheet_names:
        if "anom" in sheet.lower(): anom_df = pd.read_excel(path,sheet)
        else:
            new_hs_df = pd.read_excel(path,sheet,keep_default_na=False,na_values=['#N/A', '#N/A N/A', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null'])
            new_hs_df = new_hs_df[~new_hs_df["Loc"].str.contains("#")]
            new_hs_df.sort_values("Age",inplace=True)
            hs_dfs[sheet] = new_hs_df
    return anom_df,hs_dfs

def to_hsloc_excel(anom_df,hs_dfs,out_path):
    hs_xls = pd.ExcelWriter(out_path,engine='xlsxwriter')
    anom_df.to_excel(hs_xls,sheet_name="Magnetic Anomalies",float_format="%.3f",index=False)
    for key in hs_dfs.keys():
        if not isinstance((hs_dfs[key]),pd.DataFrame): break
        (hs_dfs[key]).to_excel(hs_xls,sheet_name=key,float_format="%.2f",index=False)
    hs_xls.save()

def calculate_along_track_uncertainty(hs_dfs,reconst=None):
    for hs_name,hs_df in hs_dfs.items():
        for plate in hs_df["Plate"].drop_duplicates().tolist():
            hs_plate_data = hs_df[hs_df["Plate"]==plate]
            prev_i = hs_plate_data.index[0]
            for j,(i,row) in enumerate(hs_plate_data.iterrows()):
                if i==prev_i: continue
                prev_geodict = Geodesic.WGS84.Inverse(row["Lat"],row["Lon"],hs_plate_data["Lat"][prev_i],hs_plate_data["Lon"][prev_i])
                if reconst==None: prev_rate = prev_geodict["a12"]/(row["Age"]-hs_plate_data["Age"][prev_i])
                else: prev_rate = (reconst[row["Age"]:hs_plate_data["Age"][prev_i]].w)/(row["Age"]-hs_plate_data["Age"][prev_i])
#                vrate = ((geodict["a12"]/(row["Age"]-hs_plate_data["Age"][prev_i])**2)**2)*(row["dAge"]**2 + hs_plate_data["dAge"][prev_i]**2)
#                new_se = np.sqrt(float(rate*row["dAge"])**2 + vrate)
                prev_new_se = np.sqrt(float(prev_rate*row["dAge"])**2 + row["MajSE"]**2)

                if i!=hs_plate_data.index[-1]:
                    next_i = hs_plate_data.index[j+1]
                    next_geodict = Geodesic.WGS84.Inverse(row["Lat"],row["Lon"],hs_plate_data["Lat"][next_i],hs_plate_data["Lon"][next_i])
                    if reconst==None: next_rate = next_geodict["a12"]/(hs_plate_data["Age"][next_i]-row["Age"])
                    else: next_rate = (reconst[hs_plate_data["Age"][next_i]:row["Age"]].w)/(hs_plate_data["Age"][next_i]-row["Age"])
                    next_new_se = np.sqrt(float(next_rate*row["dAge"])**2 + row["MajSE"]**2)

                    a = max([next_new_se,prev_new_se])
                    if next_new_se>prev_new_se:
                        maj_b = abs(prev_new_se*np.cos(np.deg2rad(90-(prev_geodict["azi1"]-next_geodict["azi1"]))))
                        b = np.sqrt((maj_b**2 + row["MinSE"]**2)/2)
                        phi = next_geodict["azi1"]
                    else:
                        maj_b = abs(next_new_se*np.cos(np.deg2rad(90-(next_geodict["azi1"]-prev_geodict["azi1"]))))
                        b = np.sqrt((maj_b**2 + row["MinSE"]**2))
                        phi = prev_geodict["azi1"]

                else: a,b,phi = prev_new_se,row["MinSE"],prev_geodict["azi1"]

                hs_df.set_value(i,"Azi",phi)
                hs_df.set_value(i,"MajSE",a)
                hs_df.set_value(i,"MinSE",b)

                prev_i = i

    return hs_dfs

def avg_contemperanious_points(hs_dfs):
    new_hs_dfs = {}
    for hs_name,hs_df in hs_dfs.items():
        new_hs_list = []
        for plate in hs_df["Plate"].drop_duplicates().tolist():
            hs_plate_data = hs_df[hs_df["Plate"]==plate]
            for i,row in hs_plate_data.iterrows():
                if i==0:
                    new_hs_dict = hs_plate_data.loc[i].to_dict()
                    new_hs_list.append(new_hs_dict)
                    continue

                skip=False
                for prev_hs_dict in new_hs_list:
                    if row["Loc"] in prev_hs_dict["Loc"]: skip=True; break
                if skip: continue

                new_hs_dict={}

                avg_age,new_dage = row["Age"],row["dAge"]
                all_points_in_age = hs_plate_data[((hs_plate_data["Age"]-hs_plate_data["dAge"])<=(avg_age+new_dage)) & ((hs_plate_data["Age"]+hs_plate_data["dAge"])>=(avg_age-new_dage))]

#                new_len,prev_len,avg_age,new_dage = 1,0,row["Age"],row["dAge"]
#                while prev_len != new_len: #Recursivly add data for averaging
#                    prev_len = new_len
#                    all_points_in_age = hs_plate_data[((hs_plate_data["Age"]-hs_plate_data["dAge"])<=(avg_age+new_dage)) & ((hs_plate_data["Age"]+hs_plate_data["dAge"])>=(avg_age-new_dage))]
#                    avg_age = sum(all_points_in_age["Age"])/len(all_points_in_age)
#                    age_range = max(all_points_in_age["Age"])-min(all_points_in_age["Age"])
#                    sum_dage = np.sqrt(sum(np.array(all_points_in_age["dAge"].tolist())**2))
#                    new_dage = max(age_range,sum_dage)
#                    new_len = len(all_points_in_age)

                if len(all_points_in_age.index)>1:

                    fdict = ipmag.fisher_mean(inc=all_points_in_age["Lat"].tolist(),dec=all_points_in_age["Lon"].tolist())
                    avg_lat,avg_lon = fdict["inc"],fdict["dec"]
                    age_var = (np.array(all_points_in_age["dAge"].tolist())**2)
                    avg_age = sum((1/age_var)*all_points_in_age["Age"])/sum(1/age_var)

                    tot_cov = np.zeros([3,3])
                    for j,point in all_points_in_age.iterrows():
                        cart,cov = latlon2cart(point["Lat"],point["Lon"],ellipse_to_cov(point["Lat"],point["Lon"],point["MajSE"],point["MinSE"],point["Azi"]))
                        tot_cov += cov
                    avg_cart,_ = latlon2cart(avg_lat,avg_lon,np.eye(3))
                    [nlat,nlon],ncov = cart2latlon(*avg_cart,cov/len(all_points_in_age))
                    a,b,azi = cov_to_ellipse(nlat,nlon,ncov)

                    age_range = max(all_points_in_age["Age"])-min(all_points_in_age["Age"])
                    sum_dage = np.sqrt(sum(age_var)/len(age_var))
                    new_dage = max(age_range,sum_dage)

                    new_name = reduce(lambda x,y: x+', '+y, all_points_in_age["Loc"].tolist())

                    new_hs_dict["Lat"] = nlat
                    new_hs_dict["Lon"] = nlon
                    new_hs_dict["Azi"] = azi
                    new_hs_dict["MajSE"] = a
                    new_hs_dict["MinSE"] = b
                    new_hs_dict["Age"] = avg_age
                    new_hs_dict["dAge"] = new_dage
                    new_hs_dict["Loc"] = new_name
                    new_hs_dict["Plate"] = plate
                    new_hs_dict["Method"] = "Average"
                    new_hs_dict["Ref"] = "This Study"

                else:

                    new_hs_dict = all_points_in_age.to_dict('records')[0]

                new_hs_list.append(new_hs_dict)

            new_hs_dfs[hs_name] = pd.DataFrame(new_hs_list)

    return new_hs_dfs

def rel_reconstruct_hs_points(hs_dfs):
    for hs_name,hs_df in hs_dfs.items():
        for i,row in hs_df.iterrows():
            if row["Plate"]=="AF": continue
            elif row["Plate"]=="NA":
                NaAfReconst = PlateReconstruction.read_csv('../Data/Rotations/Mathews2016/naaf_mathews2016.tsv') #TODO (give directory with reconstructions so it doesn't need to be hardcoded)
                rlat,rlon,_,(ra,rb,rphi) = NaAfReconst[row["Age"]].rotate(row["Lat"],row["Lon"],a=row["MajSE"],b=row["MinSE"],phi=row["Azi"])
            elif row["Plate"]=="SA":
                SaAfReconst = PlateReconstruction.read_csv('../Data/Rotations/Mathews2016/saaf_mathews2016.tsv') #TODO (give directory with reconstructions so it doesn't need to be hardcoded)
                rlat,rlon,_,(ra,rb,rphi) = SaAfReconst[row["Age"]].rotate(row["Lat"],row["Lon"],a=row["MajSE"],b=row["MinSE"],phi=row["Azi"])
            elif row["Plate"]=="AN":
                AnAfReconst = PlateReconstruction.read_csv('../Data/Rotations/Mathews2016/anaf_mathews2016.tsv') #TODO (give directory with reconstructions so it doesn't need to be hardcoded)
                rlat,rlon,_,(ra,rb,rphi) = AnAfReconst[row["Age"]].rotate(row["Lat"],row["Lon"],a=row["MajSE"],b=row["MinSE"],phi=row["Azi"])
            elif row["Plate"]=="IP":
                IpAfReconst = PlateReconstruction.read_csv('../Data/Rotations/Torsvik2012/ipnu_torsvik2012.tsv') #TODO (give directory with reconstructions so it doesn't need to be hardcoded)
                rlat,rlon,_,(ra,rb,rphi) = IpAfReconst[row["Age"]].rotate(row["Lat"],row["Lon"],a=row["MajSE"],b=row["MinSE"],phi=row["Azi"])
            else: raise ValueError("Plate %s for hotspot %s not found in set of reconstructions and thus cannot be mapped to AF Frame"%(row["Plate"],hs_name))
            hs_df.set_value(i,"Lat",rlat)
            hs_df.set_value(i,"Lon",rlon)
            hs_df.set_value(i,"Azi",rphi)
            hs_df.set_value(i,"MajSE",ra)
            hs_df.set_value(i,"MinSE",rb)
            hs_df.set_value(i,"Plate","AF")
    return hs_dfs

def gc_interp_hs_points(anom_df,hs_dfs):
    interp_hs_dfs = {}
    for hs_name,hs_df in hs_dfs.items():
        if 0.0 in hs_df["Age"]: interp_hs_list = [hs_df[hs_df["Age"]==0.0].to_dict('index')[0]]
        else: interp_hs_list = []
        for plate in hs_df["Plate"].drop_duplicates().tolist():
            hs_plate_data = hs_df[hs_df["Plate"]==plate]
            for i,row in anom_df.iterrows():
                if all(row["Age"]<hs_plate_data["Age"]-hs_plate_data["dAge"]) or all(row["Age"]>hs_plate_data["Age"]+hs_plate_data["dAge"]): continue

                interp_hs_dict={}

                dless = hs_plate_data[((hs_plate_data["Age"]-hs_plate_data["dAge"])<=row["Age"])]
                dplus = hs_plate_data[((hs_plate_data["Age"]+hs_plate_data["dAge"])>=row["Age"])]
                dless1 = dless.iloc[-1]
                dplus1 = dplus.iloc[0]
                if dless1["Loc"]==dplus1["Loc"]:
                    try:
                        dless1 = hs_plate_data[((hs_plate_data["Age"])<=row["Age"])].iloc[-1]
                        dplus1 = hs_plate_data[((hs_plate_data["Age"])>=row["Age"])].iloc[0]
                    except IndexError:
                        if len(dless)<len(dplus): dplus1=dplus.iloc[1]
                        else: dless1=dless.iloc[-2]

                geodict = Geodesic.WGS84.Inverse(dless1["Lat"],dless1["Lon"],dplus1["Lat"],dplus1["Lon"])
                weight = ((row["Age"]-dless1["Age"])/(dplus1["Age"]-dless1["Age"]))
                interp_dis = geodict["a12"]*weight
                interp_geodict = Geodesic.WGS84.ArcDirect(geodict["lat1"],geodict["lon1"],geodict["azi1"],interp_dis)

                cartless,covless = latlon2cart(dless1["Lat"],dless1["Lon"],ellipse_to_cov(dless1["Lat"],dless1["Lon"],dless1["MajSE"],dless1["MinSE"],dless1["Azi"]))
                cartplus,covplus = latlon2cart(dplus1["Lat"],dplus1["Lon"],ellipse_to_cov(dplus1["Lat"],dplus1["Lon"],dplus1["MajSE"],dplus1["MinSE"],dplus1["Azi"]))
                cartinterp,_ = latlon2cart(interp_geodict["lat2"],interp_geodict["lon2"],np.eye(3))
#                [Eless,Vless] = np.linalg.eig(covless)
#                [Eplus,Vplus] = np.linalg.eig(covplus)
#                Einterp = (weight*Eless + (1-weight)*Eplus)
#                Vinterp = weight*Vless + (1-weight)*Vplus
#                covinterp = Vinterp.T @ np.diag(Einterp) @ Vinterp
                covinterp = covless+covplus
                [lat,lon],cov = cart2latlon(*cartinterp,covinterp)
                try: a,b,azi = cov_to_ellipse(lat,lon,cov)
                except: import pdb; pdb.set_trace()

                interp_hs_dict["Lat"] = interp_geodict["lat2"]
                interp_hs_dict["Lon"] = interp_geodict["lon2"]
                interp_hs_dict["Azi"] = azi
                interp_hs_dict["MajSE"] = a
                interp_hs_dict["MinSE"] = b
                interp_hs_dict["Age"] = row["Age"]
                interp_hs_dict["dAge"] = np.sqrt(dless1["dAge"]**2+dplus1["dAge"]**2)
                interp_hs_dict["Loc"] = row["Name"]
                interp_hs_dict["Plate"] = plate
                interp_hs_dict["Method"] = "Interp"
                interp_hs_dict["Ref"] = "This Study"

                interp_hs_list.append(interp_hs_dict)

            interp_hs_dfs[hs_name] = pd.DataFrame(interp_hs_list)

    return interp_hs_dfs

def grid_search_nhspot(pres_data,paleo_data,age,nhs_queue,step=None,max_rot_rate=1,max_rot=50):

    def nhotspot_error(*args):
        (tlat,tlon,tw),pres_data,paleo_data,age = args
        trot = Rot(tlat,tlon,tw,age,0)
        tot_error = 0
        for pres_datum,paleo_datum in zip(pres_data,paleo_data):
            ptlat,ptlon,ptazi,_ = trot.rotate(paleo_datum["Lat"],paleo_datum["Lon"],paleo_datum["Azi"])
            geodict = Geodesic.WGS84.Inverse(pres_datum["Lat"],pres_datum["Lon"],ptlat,ptlon)
            gc_azi,gc_dis2 = geodict["azi1"],geodict["a12"]**2
            var = (pres_datum["MinSE"]*np.sin(np.deg2rad(pres_datum["Azi"]-gc_azi)))**2 + (pres_datum["MajSE"]*np.cos(np.deg2rad(pres_datum["Azi"]-gc_azi)))**2 + (paleo_datum["MinSE"]*np.sin(np.deg2rad(ptazi-gc_azi)))**2 + (paleo_datum["MajSE"]*np.cos(np.deg2rad(ptazi-gc_azi)))**2
            tot_error += gc_dis2/var
        return tot_error

    if not isinstance(step,type(None)):
        (mlat,mlon,mw),merror,_,_ = brute(nhotspot_error,((-90,90+step,step),(0,360,step),(step,(max_rot_rate*age)+step,step)),args=(pres_data,paleo_data,age),full_output=True,finish=fmin)
    else:

        step = 5
        mrot = (max_rot_rate*age)+step
        if mrot>(max_rot+step): mrot = max_rot
        grid = ((-90,90+step,step),(0,360,step),(step,mrot,step))
        (mlat,mlon,mw),merror,_,_ = brute(nhotspot_error,grid,args=(pres_data,paleo_data,age),full_output=True,finish=None)
        print("Age: %.2f, Grid: %d, %.2f %.2f %.2f"%(age,step,mlat,mlon,mw))

        step = 1
        mrot = mw+20+step
        if mrot>(max_rot+step): mrot = max_rot
        grid = ((mlat-30,mlat+30+step,step),(mlon-30,mlon+30+step,step),(mw-20,mrot,step))
        (mlat,mlon,mw),merror,_,_ = brute(nhotspot_error,grid,args=(pres_data,paleo_data,age),full_output=True,finish=None)
        print("Age: %.2f, Grid: %d, %.2f %.2f %.2f"%(age,step,mlat,mlon,mw))

        step = .1
        mrot = mw+3+step
        if mrot>(max_rot+step): mrot = max_rot
        grid = ((mlat-3,mlat+3+step,step),(mlon-5,mlon+5+step,step),(mw-3,mrot,step))
        (mlat,mlon,mw),merror,_,_ = brute(nhotspot_error,grid,args=(pres_data,paleo_data,age),full_output=True,finish=None)
        print("Age: %.2f, Grid: %d, %.2f %.2f %.2f"%(age,step,mlat,mlon,mw))

        step = .01
        mrot = mw+1+step
        if mrot>(max_rot+step): mrot = max_rot
        grid = ((mlat-1,mlat+1+step,step),(mlon-1,mlon+1+step,step),(mw-1,mrot,step))
        (mlat,mlon,mw),merror,_,_ = brute(nhotspot_error,grid,args=(pres_data,paleo_data,age),full_output=True,finish=fmin)

    mrot = Rot(mlat,mlon,mw,0,age)
#    opr = least_squares(nhotspot_error,[90,0,.1],args=(pres_data,paleo_data,age),bounds=((-90,0,0),(90,360,(max_rot_rate*age))))
#    mrot = Rot(*opr.x,0,age)
    print("Age: %.2f\nTot_Error: %.2f"%(age,merror))
    print(mrot)
    print("--------------------------------------------")
    nhs_queue[age] = mrot
    return

def old_grid_search_nhspot(pres_data,paleo_data,age,nhs_queue,step=1,max_rot_rate=1):
    min_error,mrot = 1e9,Rot()
    for tlon in np.arange(0,360,step):
        for tlat in np.arange(-90,90+step,step):
            for tw in np.arange(step,(max_rot_rate*age)+step,step):
                trot = Rot(tlat,tlon,tw,age,0)
                tot_error = 0
                for pres_datum,paleo_datum in zip(pres_data,paleo_data):
                    ptlat,ptlon,ptazi,_ = trot.rotate(paleo_datum["Lat"],paleo_datum["Lon"],paleo_datum["Azi"])
                    geodict = Geodesic.WGS84.Inverse(pres_datum["Lat"],pres_datum["Lon"],ptlat,ptlon)
                    gc_azi,gc_dis2 = geodict["azi1"],geodict["a12"]**2
                    s1error = (pres_datum["MinSE"]*np.sin(np.deg2rad(pres_datum["Azi"]-gc_azi)))**2 + (pres_datum["MajSE"]*np.cos(np.deg2rad(pres_datum["Azi"]-gc_azi)))**2 + (paleo_datum["MinSE"]*np.sin(np.deg2rad(ptazi-gc_azi)))**2 + (paleo_datum["MajSE"]*np.cos(np.deg2rad(ptazi-gc_azi)))**2
                    tot_error += gc_dis2/s1error
                if tot_error<min_error: min_error,mrot = tot_error,trot
    print("Age: %.2f\nTot_Error: %.2f"%(age,min_error))
    print(mrot.reverse_time())
    print("--------------------------------------------")
    nhs_queue[age] = mrot
    return

def nhotspot(ages,hs_dfs,plate_to_reconstruct="AF",step=1,max_rot_rate=1):
    pres_data = []
    for key in hs_dfs.keys():
        hs_df = hs_dfs[key]
        pres_data += hs_df[hs_df["Age"]==0.0].to_dict('records')

    nhs_manager = multiprocessing.Manager()
    nhs_processes,nhs_queue = [],nhs_manager.dict()
    for age in ages:
        paleo_data = []
        for key in hs_dfs.keys():
            hs_df = hs_dfs[key]
            paleo_data += hs_df[hs_df["Age"]==age].to_dict('records')
        process = multiprocessing.Process(target=grid_search_nhspot,args=[pres_data,paleo_data,age,nhs_queue],kwargs={"step":step,"max_rot_rate":max_rot_rate})
        nhs_processes.append(process)
        print("Starting: %.2f"%age)
        process.start()

    for age,process in zip(ages,nhs_processes):
        print("Joining: %.2f"%age)
        process.join()
    print("Getting Results")
#    nhs_rots = [nhs_queue.get() for e in nhs_processes]
    nhs_rots = list(nhs_queue.values())
    print("All Rots Found")

    return PlateReconstruction(plate_to_reconstruct,"HS",nhs_rots)

#----------------------------------------Write and Run Andrew's Code----------------------------------------------#

def write_NHSin(hs_dfs,outdir="../Data/NHSReconstruction/"):
    ages,nhs_files,nhs_data = [],[],{}
    utl.check_dir(outdir)
    for hs_name,hs_df in hs_dfs.items():
        for i,row in hs_df.iterrows():
            if i==hs_df.index[0]: continue
            if row["Age"] not in ages: ages.append(row["Age"]); nhs_data[row["Age"]] = ""
            pres_datum = hs_df.iloc[0]
            nhs_data[row["Age"]] += "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n"%(pres_datum["Lat"],pres_datum["Lon"],(360+pres_datum["Azi"])%360,pres_datum["MajSE"],pres_datum["MinSE"])#,pres_datum["MajSE"],pres_datum["MinSE"])
            nhs_data[row["Age"]] += "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n"%(row["Lat"],row["Lon"],(360+row["Azi"])%360,row["MajSE"],row["MinSE"])#,row["MajSE"],row["MinSE"])
    for age,nhs_datum in nhs_data.items():
        num_hs = int(nhs_datum.count('\n')/2)
        outfile = os.path.join(outdir,"%d_N%d.txt"%(int(float(age)*100+.5),num_hs))
        nhs_files.append(outfile)
        with open(outfile,'w+') as fout:
            fout.write(str(num_hs)+'\n'+nhs_datum)
    return ages,nhs_files





