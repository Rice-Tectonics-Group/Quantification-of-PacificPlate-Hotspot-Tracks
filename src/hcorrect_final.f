	program dla

	double precision tp(3), round, best(4)
	double precision hsset(5, 2, 5), angle, worst(4)
	double precision con
	integer i, numsets, filec, iter
	double precision xmin,xmax,ymin,ymax,zmin,zmax
	double precision error, r, small, step, x, y, z
	double precision flat, flon, smajfix, sminfix, azfix
	double precision rlat, rlon, smajrot, sminrot, azrot
	double precision lat1, lat2, lon1, lon2, smaj1, smaj2
	double precision smin1, smin2
	double precision az1, az2
	character filnam*80, filnam2*80
C	implicit none
	
C	Input: 
C	  Data Input filename and Data output filename
C	DataInput File:
C	  Hotspot Pair (dated and current locations) with uncertainties
C	    hssets are the data sets, parameters (a, b, c), a=number of 
C	    track pair, b:1 for present, 2 for dated, c: 1 for lat, 2 for 
C	    lon 3-smajoraxis, 4-sminoraxis, 5-azimuth of smajor
C	  First number in HS datafile should contain the total number of 
C	  HS sets to be used.	
C	Program function:
C	  Performs gridsearch to find the best pole and angle of 
C	    rotation for dated hotspot to current location.  Calculates 
C	    and characterizes misfits for each pole location tried.
C	 Output:
C	  File containing all poles and best angle found based on the 
C	    misfit calculation which is also included in output.     
    	    
	con = 3.14159265358979323846d0 / 180.d0
	write (6,*), 'Enter datafile name: '
	read (5,*) filnam
	
	open (1, file=filnam)
	read (1,*) numsets
	do 100 i=1,numsets
          read (1,*) flat, flon, azfix, smajfix, sminfix,rlat,rlon, 
     &      azrot,smajrot, sminrot
          hsset(i, 1, 1) = flat
          hsset(i, 1, 2) = flon
          hsset(i, 1, 3) = smajfix
          hsset(i, 1, 4) = sminfix
          hsset(i, 1, 5) = azfix
          hsset(i, 2, 1) = rlat
          hsset(i, 2, 2) = rlon
          hsset(i, 2, 3) = smajrot
          hsset(i, 2, 4) = sminrot
          hsset(i, 2, 5) = azrot
          write (6,2091) hsset(i,1,1),hsset(i,1,2),hsset(i,1,3),
     &      hsset(i,1,4),hsset(i, 1, 5),hsset(i, 2, 1),hsset(i, 2, 2),
     &	    hsset(i, 2, 3),hsset(i, 2, 4),hsset(i, 2, 5)
100     continue
        close (1)

C performing gridsearch for best pole and angle of rotation  	
C 	
	write (6,*), 'Create output file? (1=yes, 2=no)'
	read (5,*) filec
	if (filec.eq.1) then
	  write (6,*), 'Output File Name? '
	  read (5,*) filnam2
  	  open (1, file=filnam2)
  	  endif
  	write (6,*), 'Least-Squares best fit pole finder'
  	  	
  	do 2000 iter=0,3
  	if (iter.eq.0) then
  	  xmin=-90
  	  xmax=90
  	  ymin=-180
  	  ymax=180
  	  zmin=0
  	  zmax=50
  	  step=1
  	  endif
  	if (iter.eq.1) then
  	  xmin=(best(1)-30)
  	  xmax=(best(1)+30)
  	  if (best(1).lt.-60) then
  	    xmin=-90
  	    endif
  	  if (best(1).gt.60) then
  	    xmax=90
  	    endif
  	  ymin=(best(2)-40)
  	  ymax=(best(2)+40)
  	  zmin=(best(3)-10)
  	  zmax=(best(3)+10)
  	  step=1
  	  endif
  	  
  	if (iter.eq.2) then
  	  xmin=(best(1)-3)
  	  xmax=(best(1)+3)
  	  if (best(1).lt.-87) then
  	    xmin=-90
  	    endif
  	  if (best(1).gt.87) then
  	    xmax=90
  	    endif
  	  ymin=(best(2)-3)
  	  ymax=(best(2)+3)
  	  zmin=(best(3)-3)
  	  zmax=(best(3)+3)
  	  step=0.1
  	  endif
  	if (iter.eq.3) then  
  	  xmin=(best(1)-0.5)
  	  xmax=(best(1)+0.5)
  	  if (best(1).lt.-89.50) then
  	    xmin=-90
  	    endif
  	  if (best(1).gt.89.50) then
  	    xmax=90
  	    endif
  	  ymin=(best(2)-0.5)
  	  ymax=(best(2)+0.5)
  	  zmin=(best(3)-0.5)
  	  zmax=(best(3)+0.5)
  	  step=0.01
  	  endif	  
  	write (6,*), '************'
  	write (6,*), 'Iteration #', iter 
  	write (6,*), 'Trial Pole Starting With= '
  	write (6,2090) best
  	best(4)=10000.d0
	worst(4)=0.d0
	if (filec.eq.1) then
  	open (1, file=filnam2)
  	endif
          do 500 x=xmin,xmax,step
c            write (6,*), x
            do 600 y=ymin,ymax,step
              round=0.d0
1000          small=10000.d0
              do 700 z=zmin,zmax,step
		r=0.d0     	  
		tp(1)=x
		tp(2)=y
		tp(3)=z      
	        do 200 i=1,numsets
	          lat1 = hsset(i, 1, 1)
   	          lat2 = hsset(i, 2, 1)
   	          lon1 = hsset(i, 1, 2)
       	          lon2 = hsset(i, 2, 2)
   	          smaj1 = hsset(i, 1, 3)
   	          smaj2 = hsset(i, 2, 3)
   	          smin1 = hsset(i, 1, 4)
      	          smin2 = hsset(i, 2, 4)
      	          az1 = hsset(i, 1, 5)
   	          az2 = hsset(i, 2, 5)
	          call sumerr(tp, lat1, lon1, smaj1, smin1, az1,
     &	           lat2, lon2,smaj2, smin2, az2, error)
	          r=r+error
200             continue
		if (r.lt.small) then
		  small=r
		  angle=tp(3)
		  endif
700		continue
c                 write (6,*), 'trial= ',tp,r
     	      if (small.ge.worst(4)) then
	        worst(1)=tp(1)
	        worst(2)=tp(2)
	        worst(3)=angle
	        worst(4)=small
	        endif
	      if (small.le.best(4)) then
     	        best(1)=tp(1)
     	        best(2)=tp(2)
     	        best(3)=angle
     	        best(4) = small   	        
     	        endif
600     continue 
	if (filec.eq.1) then
	  write(1,'(T1,F7.2,T12,F7.2,T25,F12.5,T40,F12.9)') 
     &	  best(1),best(2),best(3),best(4)
     	endif
500 	continue
	write (6,*), 'best ='
	write (6,2090) best
	write (6,*), 'worst='
	write (6,2090) worst
	close (1)
2000    continue
	stop
	
2090    format(T10,4F10.2)
2091    format(10F8.2)
	end      	



        subroutine sumerr(trpole, flat, flon, smajfix, sminfix, azfix, 
     &   rlat, rlon, smajrot, sminrot, azrot, misfit)

C       fixed and rotat = lat, lon in degrees
	double precision misfit
        double precision flat, flon, smajfix, sminfix, azfix
        double precision rlat, rlon, smajrot, sminrot, azrot
        double precision delta, azfr, azrf, angdis, con
        double precision razfix, razrot, sigfix, sigrot, sigtot
        double precision trpole(3)
	double precision nlat, nlon, rlatr, rlonr, naz, nplat
	double precision nplon        
C	implicit none
        con = 3.14159265358979323846d0 / 180.d0

        call locate(rlat*con, rlon*con, 90.d0*con, azrot*con,
     &    rlatr, rlonr)
        call rotp(trpole(1)*con,trpole(2)*con,trpole(3)*con,rlat*con,
     &   rlon*con,nlat,nlon)
	call rotp(trpole(1)*con,trpole(2)*con,trpole(3)*con,rlatr,
     &   rlonr,nplat,nplon)
        
        nplat=nplat/con
        nplon=nplon/con
        nlat = nlat/con
        nlon = nlon/con

        call aztran(nplat*con, nplon*con, nlat*con, nlon*con, naz)
        if (nlon .ge. nplon) then
             naz = naz/con + 90.d0
            else
             naz = naz/con - 90.d0
        endif     
        
        delta = angdis(con*flat,con*flon,con*nlat,con*nlon)
        delta = delta / con
        
        call aztran(flat*con,flon*con,nlat*con,nlon*con,azfr)
        call aztran(nlat*con,nlon*con,flat*con,flon*con,azrf)
        
        if (nlon .ge. flon) then
             azfr = azfr/con + 90.d0
             azrf = azrf/con + 90.d0
            else
             azfr = azfr/con - 90.d0
             azrf = azrf/con - 90.d0
        endif     
         
        write (6,*), 'Azimuth from Rot to Fix= ',azfr
        write (6,*), 'Azimuth from Fix to Rot= ',azrf
        
        razfix = azfr - azfix
        razrot = azrf - naz

        sigfix =((smajfix**2) * (sminfix**2))/
     &     (((sin(razfix*con))**2)*((smajfix)**2)+
     &     ((cos(razfix*con))**2)*((sminfix)**2))
        sigrot =((smajrot**2) * (sminrot**2))/
     &     (((sin(razrot*con))**2)*((smajrot)**2)+
     &     ((cos(razrot*con))**2)*((sminrot)**2))
C        sigfix = ((dcos(razfix*con))**2)*((smajfix)**2)+
C      &    ((dsin(razfix*con))**2)* ((sminfix)**2)
C        sigrot = ((dcos(razrot*con))**2)*((smajrot)**2)+
C      &    ((dsin(razrot*con))**2)*((sminrot)**2)
        write (6,*), 'Fixed: A, B, Err= ',smajfix,sminfix,sqrt(sigfix)
        write (6,*), 'Rotate: A, B, Err= ',smajrot,sminrot,sqrt(sigrot)
        write (6,*), 'Fixed: ang =', razfix
        write (6,*), 'Rotate: ang =', razrot
   	sigtot = sigfix + sigrot

   	misfit=(delta**2) / sigtot

        return
        end   
                    
C       find point at given distance and azimuth
C
	subroutine locate(oldlat,oldlon,angd,az,newlat,newlon)
C	all angles are assumed to be in radians
C       az is in radians cw from north
	double precision oldlat,oldlon,angd,az,newlat,newlon,pi
	double precision tlat1,tlon1
	data pi/ 3.14159265358979323846d0 /
	call rot2( pi / 2.d0-angd,pi-az, pi / 2.d0-oldlat,tlat1,tlon1)
	call rot3(tlat1,tlon1,oldlon,newlat,newlon)
	return
	end
        
        real function sq(x)
C	squares input
	real x
	sq = x * x
        return
        end
                
        double precision function angdis(lat1,lon1,lat2,lon2)
C	input assumed in radians; output is in radians
	double precision lat1,lon1,lat2,lon2,x1,x2,x3,y1,y2,y3
	call sphcar(lat1,lon1,x1,x2,x3)
	call sphcar(lat2,lon2,y1,y2,y3)
	angdis = dabs(acos(x1*y1 + x2*y2 + x3*y3))
	return
	end
     
       
        subroutine sphcar(lat,lon, x1,x2,x3)
C	change spherical coordinates to cartesian
C	lat and lon are assumed to be in radians
 	double precision lat, lon, x1, x2, x3
	x1 = dcos(lat) * dcos(lon)
	x2 = dcos(lat) * dsin(lon)
	x3 = dsin(lat)
	return
	end
         
	
        subroutine aztran(polat,polon,sitlat,sitlon,az)
C	all angles assumed to be in radians
C	azimuth is cw from north
	double precision polat,polon,sitlat,sitlon,az,tlat1
	double precision tlon1,tlat2,tlon2,pi
	data pi/ 3.14159265358979323846d0 /	
	call rot3(polat,polon,pi - sitlon,tlat1,tlon1)
	call rot2(tlat1,tlon1,pi/2.d0-sitlat,tlat2,tlon2)
	az = pi / 2.d0 - tlon2
	return
	end
       
C       rotate w radians about an arbitrary pole
	subroutine rotp(polat,polon,angle,oldlat,oldlon,newlat,newlon)
C	all angles assumed to be in radians
	double precision polat,polon,angle,oldlat,oldlon,newlat
	double precision newlon,pi
	double precision tlat,tlon,tlat2,tlon2,tlat3,tlon3
	double precision tlat4,tlon4
	data pi/ 3.14159265358979323846d0 /
	call rot3(oldlat,oldlon,-polon,tlat,tlon)
	call rot2(tlat,tlon,polat - pi / 2.,tlat2,tlon2)
	call rot3(tlat2,tlon2,angle,tlat3,tlon3)
	call rot2(tlat3,tlon3,pi / 2. - polat,tlat4,tlon4)
	call rot3(tlat4,tlon4,polon,newlat,newlon)
	return
	end
            
C	rotate 'angle' radians about axis #2
	subroutine rot2(oldlat,oldlon,angle,newlat,newlon)
C	all angles are assumed to be in radians
	double precision oldlat,oldlon,angle,newlat,newlon
	double precision temp,x1,x2,x3
	call sphcar(oldlat,oldlon,x1,x2,x3)
	temp = x1
	x1 = x1 * dcos(angle) + x3 * dsin(angle)
	x3 = x3 * dcos(angle) - temp * dsin(angle)
	call carsph(x1,x2,x3,newlat,newlon)
	return
	end


C       rotate 'angle' radians about axis #3	
	subroutine rot3(oldlat,oldlon,angle,newlat,newlon)
C	all angles are assumed to be in radians
	double precision oldlat,oldlon,angle,newlat,newlon,pi
	data pi/ 3.14159265358979323846d0 /
	newlat = oldlat
	newlon = oldlon + angle
	if (newlon .gt. 2.d0 * pi) then
             newlon = newlon - 2.d0 * pi
	else if (newlon .lt. -2.d0 * pi) then
             newlon = newlon + 2.d0 * pi
        endif
	return
     	end
     	
C       change cartesian coordinates to spherical
	subroutine carsph(x1,x2,x3, lat,lon)
C	output lat and lon are in radians
	double precision lat,lon,x1,x2,x3
	lat = dasin(x3)
	lon = datan2(x2 , x1)
	return
	end
