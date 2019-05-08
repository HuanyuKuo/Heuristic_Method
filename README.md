# Heuristic_Method
# Create Date: May 07, 2019 by Huanyu Kuo
# Python code made for SV's coevolution experiment data

Heuristic Method: 
Use heuristic method to generate (1) Group of Neutral & Adaptives (2) Mean Fitness Estimates (3) Selection coefficient of each lineage on each time point

Mean Fitness Alignment:
The extinction of neutrals on later time point ruins the estimates mean fitness estimates (and selection coefficient as well) in Heuristic Method. I “fixed” the mean fitness by taking the Large Lineage on middle time point and alignment their selection coefficients (assume the selection coefficient is a constant for those lineages, i.e. no secondary mutation). The alignment will bends the mean fitness x  -> x - dx. This program will generate txt file of dx.

Plot_Grant:
Some functions to make plots like frequency trajectory, DFE, mean fitness. etc.
