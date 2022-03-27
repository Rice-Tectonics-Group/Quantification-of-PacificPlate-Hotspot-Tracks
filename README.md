# Quantification of Pacific Plate Hotspot Tracks

This repository contains the data and results of the Tectonics manuscript "Quantification of Pacific Plate Hotspot Tracks 
Since 80 Ma and the Relative Timing of Eocene Plate Tectonic Events" by Kevin Gaastra, Richard Gordon, and Daniel Woodworth

## Software

The src folder contains primarily python 3.7 based software in the src folder though there are also 
a Fortran 77 and a C script for running the Nhotspot method (Andrews et al. 2006), as well as a 
bash scripts which run the previous two and interact with the nhotspot.py python library.

The software can reproduce all analysis figures in the manuscript and some basic structure is as follows:  
- multi_circ_inv.py: library that contains functions for geographic inversion method of Gaastra et al. 2021  
- multi_fit_all_circs.py: script that runs the majority of the age progression analysis of Gaastra et al. 2021 for non-coeval bends  
- multi_fit_all_circs_3const.py: script that runs the majority of the age progression analysis of Gaastra et al. 2021 for coeval bends  
- multi_fit_all_dis.py: script that does the interhotspot distance calculation, rates, and uncertainties  
- multi_fit_plumedrift.py: script that renders misfit to pacific hotspot age models (Figure 5 Gaastra et al. 2021)  
- nhotspot.py: library that runs and wraps bash and low level scripts for the nhotspot method of Andrews et al. 2006  
- pa_rec.py: script to run the nhotspot method  
- make_hs_locs.py: script to take the simply parameterized geographic and age models and turn them into the spreadsheet for the pa_rec script  
- pole_plot.py: script to generate plot of rotation poles and also the small circle pole uncertainties for the geographic models  
- template_timescale.py: script to create a timescale visual of events in the Eocene (Figure 6 Gaastra et al. 2021)  
- vector_endpoint_plot.py: script to create a vector endpoint diagram of reconstructions in the Pacific (Figure 4 Gaastra et al. 2021)  
- vector_endpoint_plot_all_recs.py: script to create a vector endpoint diagram of global recsontructions (Figure 7 Gaastra et al. 2021)  

## Data

The data contains input and output files for the software in the src directory. Some basic details are:  
- The reconstructions subdirectory contains csv files of all reconstructions used in the study of this data  
- The subtrack_pole_grid subdirectory contains text files of the misfit matrix for the small circle pole error surfaces of the geographic models  
- Bend_Inversion_*_5.00_0.10_chi2surf.txt: the misfit matrix for the bend of the respective hotspots  
- Bend_Inversion_*_5.00_0.10_chi2surf_subair.txt: the misfit matrix for the bend of the respective hotspots including subarial age data  
- Events_Near_HEB_manuscript.csv: list of references and events as well as color and time of such events for rendering  
- GMHRFPlumeMotions.txt: the GMHRF plume motions file required to run multi_fit_plumedrift.py  
- pa_seamount_ages_subareal_included.xlsx: A spreadsheet of seamount ages  


## References

1) Andrews, D. L., Gordon, R. G., & Horner-Johnson, B. C. (2006). Uncertainties in plate reconstructions relative to the hotspots; Pacific-hotspot rotations and uncertainties for the past 68 million years. Geophysical Journal International, 166(2), 939-951.
