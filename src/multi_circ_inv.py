from numpy import *
from scipy.optimize import brute,fmin
from functools import reduce,cmp_to_key
from geographiclib.geodesic import Geodesic
from tqdm import tqdm

geoid = Geodesic(6731.,0.)#Geodesic.WGS84

def pole_arc_fitfunc(tpole,ppoles,bend,sml_circ,l1_dis):
    (tlat,tlon,tdis),_,dis,s1_errors = get_pole_arc_misfit_uncertainty(tpole,ppoles,bend,sml_circ=sml_circ)
#    if round(geoid.Inverse(*bend,*tpole)["a12"],1)!=tdis: return inf
    if l1_dis: tot_error = sum((abs(dis-tdis)/s1_errors))
    else: tot_error = sum(((dis-tdis)/s1_errors)**2)
    return tot_error

def get_pole_arc_misfit_uncertainty(tpole,ppoles,bend,sml_circ=False):
    tlat,tlon = tpole
    dis,azis,s1_errors=[],[],[]
    for plat,plon,a,b,phi in ppoles:
        geo_dict = geoid.Inverse(tlat,tlon,plat,plon)
        dis.append(geo_dict['a12'])
        azi = geo_dict['azi1']
        azis.append(azi)
        s1_errors.append(((a*b)/sqrt(((cos(phi-azi)*b)**2 + (sin(phi-azi)*a)**2)))/sqrt(2))
    if sml_circ: tdis = geoid.Inverse(tlat,tlon,*bend)["a12"]
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

#ADD THIS TO MAIN RECONSTRUCTION OBJ SO YOU CAN ESTIMATE MOTION ARCS FROM POLES
def fit_circ(ppoles1,ppoles2,gridspace=1,bend_gridspace=1,pvalue=.05,bend_window=[30.,35.,170.,175.],sml_circ1=True,sml_circ2=True,north_hemisphere=True,finish=True,l1_dis=False): #Follows method laid out in Gordon and Cox 1984 Appendix A
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
    min_bend_err,ep1_lat,ep1_lon,ep1_dis,ep2_lat,ep2_lon,ep2_dis = inf,0,0,0,0,0,0
#    if pvalue<1 and pvalue>0: chi2_cutoff = chi2.ppf(1-pvalue,2*len(ppoles)-3)
#    else: raise ValueError("pvalue must be less than 1 and greater than 0")

    #find euler pole and all poles that are within confidence
    blat_range = bend_window[:2]
    blat_range[1] += bend_gridspace
    blon_range = bend_window[2:]
    blon_range[1] += bend_gridspace
    X = arange(*blon_range,bend_gridspace)
    Y = arange(*blat_range,bend_gridspace)
    lon_mesh, lat_mesh = meshgrid(X, Y)
    chi2_surf = zeros(lon_mesh.shape)
    for i,blon in tqdm(enumerate(arange(*blon_range,bend_gridspace))):
        for j,blat in tqdm(enumerate(arange(*blat_range,bend_gridspace))):
#            print(blon,blat)
            min_tot_error1,min_tot_error2 = inf,inf
            lon_range,lat_range = [0.,360.],[-90.,90.]
            for grd in [gridspace,gridspace/2,gridspace/10]:

                if grd==gridspace:
#                    if not sml_circ2: #restrict lon and lat range to only gcs passing through this point
#                        lats,lons = [],[]
#                        for azi in arange(0.,360.,grd):
#                            geodict = geoid.ArcDirect(blat,blon,azi,90.)
#                            lats.append(geodict["lat2"]),lons.append(geodict["lon2"])
                    lons,lats = arange(*lon_range,grd),arange(*lat_range,grd)
                    for tlon in lons:
                        for tlat in lats:

                            #First Inv.
                            (tlat1,tlon1,tdis1),(tmin_azi1,tmax_azi1),dis1,s1_errors1 = get_pole_arc_misfit_uncertainty((tlat,tlon),ppoles1,(blat,blon),sml_circ=sml_circ1)
                            if l1_dis: tot_error1 = sum((abs(dis1-tdis1)/s1_errors1))
                            else: tot_error1 = sum(((dis1-tdis1)/s1_errors1)**2)
                            if tot_error1<min_tot_error1:
                                min_tot_error1,tmp_ep1_lat,tmp_ep1_lon,tmp_ep1_dis,tmp_min_azi1,tmp_max_azi1 = tot_error1,tlat1,tlon1,tdis1,tmin_azi1,tmax_azi1 #check new best

                            #Second Inv.
                            (tlat2,tlon2,tdis2),(tmin_azi2,tmax_azi2),dis2,s2_errors2 = get_pole_arc_misfit_uncertainty((tlat,tlon),ppoles2,(blat,blon),sml_circ=sml_circ2)
                            if l1_dis: tot_error2 = sum((abs(dis2-tdis2)/s2_errors2))
                            else: tot_error2 = sum(((dis2-tdis2)/s2_errors2)**2)
                            if tot_error2<min_tot_error2:
                                min_tot_error2,tmp_ep2_lat,tmp_ep2_lon,tmp_ep2_dis,tmp_min_azi2,tmp_max_azi2 = tot_error2,tlat2,tlon2,tdis2,tmin_azi2,tmax_azi2 #check new best

                    if (min_tot_error1+min_tot_error2)<min_bend_err:
                        min_bend_err = (min_tot_error1+min_tot_error2)
                        bend_lat,bend_lon = blat,blon
                        ep1_lat,ep1_lon,ep1_dis,min_azi1,max_azi1 = tmp_ep1_lat,tmp_ep1_lon,tmp_ep1_dis,tmp_min_azi1,tmp_max_azi1
                        ep2_lat,ep2_lon,ep2_dis,min_azi2,max_azi2 = tmp_ep2_lat,tmp_ep2_lon,tmp_ep2_dis,tmp_min_azi2,tmp_max_azi2
                    lon1_range,lat1_range = [(360.+(tmp_ep1_lon-30.))%360.,(360.+(tmp_ep1_lon+30.))%360.],[tmp_ep1_lat-15.,tmp_ep1_lat+15.]
                    lon2_range,lat2_range = [(360.+(tmp_ep2_lon-30.))%360.,(360.+(tmp_ep2_lon+30.))%360.],[tmp_ep2_lat-15.,tmp_ep2_lat+15.]
                    if lat1_range[1]>90.: lat1_range[1]=90.
                    if lat2_range[1]>90.: lat2_range[1]=90.
                    if lat1_range[1]>90.: lat1_range[1]=90.
                    if lat2_range[1]>90.: lat2_range[1]=90.
                else:
                    lons1,lats1 = arange(*lon1_range,grd),arange(*lat1_range,grd)
                    lons2,lats2 = arange(*lon2_range,grd),arange(*lat2_range,grd)

                    for tlon in lons1:
                        for tlat in lats1:

                            #First Inv.
                            (tlat1,tlon1,tdis1),(tmin_azi1,tmax_azi1),dis1,s1_errors1 = get_pole_arc_misfit_uncertainty((tlat,tlon),ppoles1,(blat,blon),sml_circ=sml_circ1)
                            if l1_dis: tot_error1 = sum((abs(dis1-tdis1)/s1_errors1))
                            else: tot_error1 = sum(((dis1-tdis1)/s1_errors1)**2)
                            if tot_error1<min_tot_error1:
                                min_tot_error1,tmp_ep1_lat,tmp_ep1_lon,tmp_ep1_dis,tmp_min_azi1,tmp_max_azi1 = tot_error1,tlat1,tlon1,tdis1,tmin_azi1,tmax_azi1 #check new best

                    for tlon in lons2:
                        for tlat in lats2:

                            #Second Inv.
                            (tlat2,tlon2,tdis2),(tmin_azi2,tmax_azi2),dis2,s2_errors2 = get_pole_arc_misfit_uncertainty((tlat,tlon),ppoles2,(blat,blon),sml_circ=sml_circ2)
                            if l1_dis: tot_error2 = sum((abs(dis2-tdis2)/s2_errors2))
                            else: tot_error2 = sum(((dis2-tdis2)/s2_errors2)**2)
                            if tot_error2<min_tot_error2:
                                min_tot_error2,tmp_ep2_lat,tmp_ep2_lon,tmp_ep2_dis,tmp_min_azi2,tmp_max_azi2 = tot_error2,tlat2,tlon2,tdis2,tmin_azi2,tmax_azi2 #check new best

                    chi2_surf[j,i] = (min_tot_error1+min_tot_error2)
                    if (min_tot_error1+min_tot_error2)<min_bend_err:
                        min_bend_err = (min_tot_error1+min_tot_error2)
                        bend_lat,bend_lon = blat,blon
                        ep1_lat,ep1_lon,ep1_dis,min_azi1,max_azi1 = tmp_ep1_lat,tmp_ep1_lon,tmp_ep1_dis,tmp_min_azi1,tmp_max_azi1
                        ep2_lat,ep2_lon,ep2_dis,min_azi2,max_azi2 = tmp_ep2_lat,tmp_ep2_lon,tmp_ep2_dis,tmp_min_azi2,tmp_max_azi2

                    lon1_range,lat1_range = [(360.+(tmp_ep1_lon-15.))%360.,(360.+(tmp_ep1_lon+15.))%360.],[tmp_ep1_lat-10.,tmp_ep1_lat+10.]
                    lon2_range,lat2_range = [(360.+(tmp_ep2_lon-15.))%360.,(360.+(tmp_ep2_lon+15.))%360.],[tmp_ep2_lat-10.,tmp_ep2_lat+10.]
                    if lat1_range[1]>90.: lat1_range[1]=90.
                    if lat2_range[1]>90.: lat2_range[1]=90.
                    if lat1_range[1]>90.: lat1_range[1]=90.
                    if lat2_range[1]>90.: lat2_range[1]=90.


#                    if tot_error<=chi2_cutoff: error_points.append([tlat,tlon,tdis]) #check in confidence limits

#    ppoles1 = list(filter(lambda x: x[1]<=blon and x[0]>=blat,ppoles)) #split into pre-bend
#    ppoles2 = list(filter(lambda x: x[1]>blon and x[0]<blat,ppoles)) #split into post-bend

    if finish:
        (ep1_lat,ep1_lon),min_tot_error1,_,_,_ = fmin(pole_arc_fitfunc,(ep1_lat,ep1_lon),args=(ppoles1,(bend_lat,bend_lon),sml_circ1,l1_dis),full_output=True)
        (ep1_lat,ep1_lon,ep1_dis),(min_azi1,max_azi1),_,_ = get_pole_arc_misfit_uncertainty((ep1_lat,ep1_lon),ppoles1,(bend_lat,bend_lon),sml_circ=sml_circ1)

        (ep2_lat,ep2_lon),min_tot_error2,_,_,_ = fmin(pole_arc_fitfunc,(ep2_lat,ep2_lon),args=(ppoles2,(bend_lat,bend_lon),sml_circ2,l1_dis),full_output=True)
        (ep2_lat,ep2_lon,ep2_dis),(min_azi2,max_azi2),_,_ = get_pole_arc_misfit_uncertainty((ep2_lat,ep2_lon),ppoles2,(bend_lat,bend_lon),sml_circ=sml_circ2)
        min_bend_err = (min_tot_error1+min_tot_error2)

    #ensure proper antipode is returned
    if north_hemisphere and ep1_lat<0: ep1_lat,ep1_lon,ep1_dis = -ep1_lat,(180+ep1_lon)%360,180-ep1_dis
    elif not north_hemisphere and ep1_lat>0: ep1_lat,ep1_lon,ep1_dis = -ep1_lat,(180+ep1_lon)%360,180-ep1_dis

    if north_hemisphere and ep2_lat<0: ep2_lat,ep2_lon,ep2_dis = -ep2_lat,(180+ep2_lon)%360,180-ep2_dis
    elif not north_hemisphere and ep2_lat>0: ep2_lat,ep2_lon,ep2_dis = -ep2_lat,(180+ep2_lon)%360,180-ep2_dis

    #Return within confidence poles to determine error ellipse
    #exclude current euler pole and antipodal points without messing up hemispheres
#    corrected_error_points = list(filter(lambda x: sign(x[2]-90)==sign(ep_dis-90), error_points))
#    if [ep_lat,ep_lon,ep_dis] in corrected_error_points: corrected_error_points.remove([ep_lat,ep_lon,ep_dis])

    #Use data and ep to determine the rotation angle
#    sml_circ_points = []
#    for azi in arange(min_azi,max_azi,gridspace):
#        geo_dict = Geodesic.WGS84.ArcDirect(ep_lat,ep_lon,azi%360,ep_dis)
#        sml_circ_points.append([geo_dict["lat2"],geo_dict["lon2"]])
#    ep_rot = sum([Geodesic.WGS84.Inverse(*sml_circ_points[i-1],*sml_circ_points[i])['a12'] for i in range(1,len(sml_circ_points))])

    return (bend_lat,bend_lon),(ep1_lat,ep1_lon,ep1_dis),(ep2_lat,ep2_lon,ep2_dis),min_bend_err,ppoles1,ppoles2,(lon_mesh,lat_mesh,chi2_surf)

def get_max_likelyhood_small_circ_radius(dis,s1_errors): #From Gordon and Cox 1984 (eqn. A13)
    return sum(dis/(s1_errors**2))/sum(1/(s1_errors**2))
