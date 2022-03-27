/* eig2conreg.c	convert rotation pole and eigenvector/eigenvalues to 
 *		conreg input format
 *
 * 2003-OCT-03	Benjamin C. Horner-Johnson [BCH-J]
 * 2003-NOV-06	[BCH-J] changed name from hs4toconreg to eig2conreg,
 *			expanded usage information, switched from 0.95
 *			to 0.888 for confidence level (3D->2D projection)
 *
 * gcc eig2conreg.c -lm -o eig2conreg 
 */

/* Include standard libraries */
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

/* Main Program */
main(int argc, char **argv)
{

  double deg2rad;
  double plat, plon, omega;
  double lat[3], lon[3], sigma[3];
  double qhat1, qhat2, qhat3, qhat4;
  double max1, max2, max3, int1, int2, int3, min1, min2, min3;
  double ahat[3][3], h11_2[3][3], mul[3][3], S[3][3], St[3][3], cov[3][3];
  double ahat11, ahat12, ahat13, ahat21, ahat22, ahat23, ahat31, ahat32, ahat33;
  double L11, L12, L13, L21, L22, L23, L31, L32, L33;
  char line[BUFSIZ], col[3][15];
  char file1[BUFSIZ], file2[BUFSIZ];
  int i, err = 0;
  FILE *fp1, *fp2;

  /* calculate conversion factor for degrees to radians (pi/180) */
  /* atan(1.0) = pi/4 */
  deg2rad = 4.0 * atan(1.0) / 180.0;

  /* open input/output files from command line */
  switch (argc) {
     case 3:
	     strcpy(file1,(*(++argv)));
             if ((fp1=fopen(file1,"r"))==NULL){
                fprintf (stderr, "Error opening input file: %s\n",(*(argv)));
                exit (-1);
             }
	     strcpy(file2,(*(++argv)));
             if ((fp2=fopen(file2,"w"))==NULL){
               fprintf (stderr, "Error opening output file: %s\n",(*(argv)));
               exit(-1);
             }
             break;
     case 2:
	     strcpy(file1,(*(++argv)));
	     if ( (strcmp(file1,"-help")==0) || (strcmp(file1,"-h")==0) ) {
	       err+=1;
	       break;
             } 
             if ((fp1=fopen(file1,"r"))==NULL){
               fprintf (stderr, "Error opening input file: %s\n",(*(argv)));
               exit (-1);
             }
	     strcpy(file2,"stdout");
             fp2=stdout;
             break;
     case 1:
	     strcpy(file1,"stdin");
             fp1=stdin;
	     strcpy(file2,"stdout");
             fp2=stdout;
             break;
     default:
	     err+=1;
             break;
  }

  if (err > 0) {
    fprintf(stderr,"Usage: eig2conreg [filename] [outfile]\n");
    fprintf(stderr,"\tExpected input:\n");
    fprintf(stderr,"\t\tLine 1: rotation pole (lat,lon,angle)\n");
    fprintf(stderr,"\t\tLine 2: max eigenvector and eigenvalue (lat,lon,sigma_max)\n");
    fprintf(stderr,"\t\tLine 3: int eigenvector and eigenvalue (lat,lon,sigma_int)\n");
    fprintf(stderr,"\t\tLine 4: min eigenvector and eigenvalue (lat,lon,sigma_min)\n");
    exit(0);    
  } 

  /* INPUT */

  /* pole of rotation; conversion to radians */
  fprintf(stderr,"Input rotation pole (lat, lon, omega)\n");
  fgets(line,BUFSIZ,fp1);
  sscanf(line,"%s %s %s",col[0],col[1],col[2]);
  plat = deg2rad * atof(col[0]);
  plon = deg2rad * atof(col[1]);
  omega = deg2rad * atof(col[2]);

  /* eigenvectors and eigenvalues; conversion to radians */
  fprintf(stderr,"Input 3x3 matrix of eigenvectors and eigenvalues.\n");
  fprintf(stderr,"Enter lat, lon, sigma_max\n");
  for(i = 0; i < 3; i++) {
    switch (i) {
    case 1:
      fprintf(stderr,"Enter lat, lon, sigma_int\n");
      break;
    case 2:
      fprintf(stderr,"Enter lat, lon, sigma_min\n");
      break;
    default:
      break;
    }
    fgets(line,BUFSIZ,fp1);
    sscanf(line,"%s %s %s",col[0],col[1],col[2]);
    lat[i] = deg2rad * atof(col[0]);
    lon[i] = deg2rad * atof(col[1]);
    sigma[i] = deg2rad * atof(col[2]);
  }
  fclose(fp1);

  /* calculation */
  /*  fprintf(stderr,"Calculating ahat and H11.2\n"); */

  /* convert plat,plon,omega to quaternion qhat */
  qhat1 = cos(omega/2.0);
  qhat2 = sin(omega/2.0) * cos(plat) * cos(plon);
  qhat3 = sin(omega/2.0) * cos(plat) * sin(plon);
  qhat4 = sin(omega/2.0) * sin(plat);

  /* convert quaternion quat to rotation matrix ahat */
  ahat11 = qhat1*qhat1 + qhat2*qhat2 - qhat3*qhat3 - qhat4*qhat4;
  ahat21 = 2.0 * (qhat1*qhat4 + qhat2*qhat3);
  ahat31 = 2.0 * (qhat2*qhat4 - qhat1*qhat3);
  ahat12 = 2.0 * (qhat2*qhat3 - qhat1*qhat4);
  ahat22 = qhat1*qhat1 - qhat2*qhat2 + qhat3*qhat3 - qhat4*qhat4;
  ahat32 = 2.0 * (qhat1*qhat2 + qhat3*qhat4);
  ahat13 = 2.0 * (qhat1*qhat3 + qhat2*qhat4);
  ahat23 = 2.0 * (qhat3*qhat4 - qhat1*qhat2);
  ahat33 = qhat1*qhat1 - qhat2*qhat2 - qhat3*qhat3 + qhat4*qhat4;

  /* convert eigenvectors to Cartesian coordinates */
  max1 = cos(lat[0]) * cos(lon[0]);
  max2 = cos(lat[0]) * sin(lon[0]);
  max3 = sin(lat[0]);
  int1 = cos(lat[1]) * cos(lon[1]);
  int2 = cos(lat[1]) * sin(lon[1]);
  int3 = sin(lat[1]);
  min1 = cos(lat[2]) * cos(lon[2]);
  min2 = cos(lat[2]) * sin(lon[2]);
  min3 = sin(lat[2]);

  /* set up eigenvector matrices */
  S[0][0] = St[0][0] = max1;
  S[1][0] = St[0][1] = max2;
  S[2][0] = St[0][2] = max3;
  S[0][1] = St[1][0] = int1;
  S[1][1] = St[1][1] = int2;
  S[2][1] = St[1][2] = int3;
  S[0][2] = St[2][0] = min1;
  S[1][2] = St[2][1] = min2;
  S[2][2] = St[2][2] = min3;

  /* set up eigenvalue matrix for cov, using sigma squared */
  L11 = sigma[0] * sigma[0]; 
  L22 = sigma[1] * sigma[1]; 
  L33 = sigma[2] * sigma[2]; 
  L12 = L21 = L13 = L31 = L23 = L32 = 0.0;

  /* multiply matrix Lambda by S */
  for (i = 0; i < 3; i++) {
    mul[i][0] = L11*S[i][0] + L21*S[i][1] + L31*S[i][2];
    mul[i][1] = L12*S[i][0] + L22*S[i][1] + L32*S[i][2];
    mul[i][2] = L13*S[i][0] + L23*S[i][1] + L33*S[i][2];
  }
  /* multiply matrix SLambda by St */
  for (i = 0; i < 3; i++) {
    cov[i][0] = mul[i][0]*St[0][0] + mul[i][1]*St[1][0] + mul[i][2]*St[2][0];
    cov[i][1] = mul[i][0]*St[0][1] + mul[i][1]*St[1][1] + mul[i][2]*St[2][1];
    cov[i][2] = mul[i][0]*St[0][2] + mul[i][1]*St[1][2] + mul[i][2]*St[2][2];
  }

  /* set up inverse eigenvalue matrix for H11.2 */
  L11 = 1.0 / L11;
  L22 = 1.0 / L22;
  L33 = 1.0 / L33;
  L12 = L21 = L13 = L31 = L23 = L32 = 0.0;

  /* multiply matrix Lambda^{-1} by S */
  for (i = 0; i < 3; i++) {
    mul[i][0] = L11*S[i][0] + L21*S[i][1] + L31*S[i][2];
    mul[i][1] = L12*S[i][0] + L22*S[i][1] + L32*S[i][2];
    mul[i][2] = L13*S[i][0] + L23*S[i][1] + L33*S[i][2];
  }
  /* multiply matrix SLambda^{-1} by St */
  for (i = 0; i < 3; i++) {
    h11_2[i][0] = mul[i][0]*St[0][0] + mul[i][1]*St[1][0] + mul[i][2]*St[2][0];
    h11_2[i][1] = mul[i][0]*St[0][1] + mul[i][1]*St[1][1] + mul[i][2]*St[2][1];
    h11_2[i][2] = mul[i][0]*St[0][2] + mul[i][1]*St[1][2] + mul[i][2]*St[2][2];
  }


  /* OUTPUT */
  /* 2 comment lines */
  fprintf(fp2,"eig2conreg conversion from input file: %s\n",file1);
  fprintf(fp2,"                       to output file: %s\n",file2);
  /* label for pole and pole */
  fprintf(fp2,"rotation pole: alat, alon, omega\n");
  fprintf(fp2,"%f\t%f\t%f\n", plat/deg2rad, plon/deg2rad, omega/deg2rad);
  /* label for confidence level, level */
  fprintf(fp2,"confidence level (0.888 for 3D->2D 95%%)\n 0.888\n");
  /* label for kappahat and degrees of freedom, values */
  fprintf(fp2,"kappahat, degrees of freedom\n 1.0\t10000.0\n");
  /* 2 blank lines 
     would be number of points and sections from finite rotation */
  fprintf(fp2,"dummy value\n0\n");
  /* 3x3 rotation matrix ahat */
  fprintf(fp2,"rotation matrix ahat:\n");
  fprintf(fp2,"%g   %g   %g\n", ahat11, ahat12, ahat13);
  fprintf(fp2,"%g   %g   %g\n", ahat21, ahat22, ahat23);
  fprintf(fp2,"%g   %g   %g\n", ahat31, ahat32, ahat33);
  /* 4 blank lines, would be label and 3x3 covariance matrix */
  fprintf(fp2,"covariance matrix:\n");
  for (i = 0; i < 3; i++)
    fprintf(fp2," %e   %e   %e\n",cov[0][i], cov[1][i], cov[2][i]);
  /* label and 3x3 inverse covariance matrix (H11.2) */
  fprintf(fp2,"H11.2 matrix:\n");
  for (i = 0; i < 3; i++)
    fprintf(fp2,"%g   %g   %g\n",h11_2[0][i], h11_2[1][i], h11_2[2][i]);
  fclose(fp2);
}


