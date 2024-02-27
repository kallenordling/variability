import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import glob
import os
import regionmask
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

seasons =['DJF','MAM','JJA','SON']
seasons=['JJA']
exps = ["0","1","2","3","4"] # Names of warming levels
#exps=["2"]
models=[sys.argv[1]]#,'CanESM5','ACCESS-ESM1-5']
var_list=[sys.argv[2]]
#var_list=['tasmax']
#models=['ACCESS-ESM1-5']
#change in mean sift

for model in models:
	for var in var_list:
		for season in seasons:
			ds_base = xr.open_dataset('map_data_new/'+model+"_data_"+season+"_0_"+var+"_total.nc")#.isel(time=slice(0,33000))
			ds_base =ds_base[list(ds_base.keys())[0]]
			treshold_high = ds_base.quantile(0.999,dim='time')

			for exp in exps:
				print(exp,var,'map_data_new/'+model+"_data_"+season+"_"+exp+"_"+var+"_mean_sift.nc")
				ds = xr.open_dataset('map_data_new/'+model+"_data_"+season+"_"+exp+"_"+var+"_mean_sift.nc")#.isel(time=slice(0,33000))
				ds =ds[list(ds.keys())[0]]
				ds_h = xr.where(ds >= treshold_high,1,0).sum('time')/(len(ds.time)/(31+31+30))	
				ds_h.to_dataset(name=var).to_netcdf('map_data_treshold_new/'+model+'_'+season+"_"+var+'_'+exp+"_high_abs_mean_sift.nc",format="NETCDF3_64bit")


				ds = xr.open_dataset('map_data_new/'+model+"_data_"+season+"_"+exp+"_"+var+"_var.nc")#.isel(time=slice(0,33000))
				ds =ds[list(ds.keys())[0]]
				print(len(ds.time),exp,var)
				ds_h = xr.where(ds >= treshold_high,1,0).sum('time')/(len(ds.time)/(31+31+30))				
				ds_h.to_dataset(name=var).to_netcdf('map_data_treshold_new/'+model+'_'+season+"_"+var+'_'+exp+"_high_abs_var.nc",format="NETCDF3_64bit")

				ds = xr.open_dataset('map_data_new/'+model+"_data_"+season+"_"+exp+"_"+var+"_total.nc")#.isel(time=slice(0,33000))
				ds =ds[list(ds.keys())[0]]
				print(len(ds.time),exp,var)
				ds_h = xr.where(ds >= treshold_high,1,0).sum('time')/(len(ds.time)/(31+31+30))				
				ds_h.to_dataset(name=var).to_netcdf('map_data_treshold_new/'+model+'_'+season+"_"+var+'_'+exp+"_high_abs_total.nc",format="NETCDF3_64bit")


