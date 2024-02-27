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
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import json
import cftime
from sklearn.neighbors import KernelDensity
from scipy import stats
import scipy as sp
import scipy.interpolate   
from shapely.geometry import LineString
#import scipy.integrate as integrate
from scipy.stats import ks_2samp
from scipy import integrate
import warnings
import json
import os
from scipy import stats
import regionmask
import warnings
import matplotlib.patheffects as pe
from mycolorpy import colorlist as mcp
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

def maskRegion(ds,region):
	ar6_land=regionmask.defined_regions.ar6.land.mask(ds)
	reg_num =regionmask.defined_regions.ar6.land.map_keys(region)
	return ds.where(ar6_land != reg_num)

def make_cmap(cmap,num):
	list_color=mcp.gen_color(cmap=cmap,n=num)
	list_color.insert(int(num/2),'whitesmoke')
	list_color.insert(int(num/2),'whitesmoke')
	return mpl.colors.ListedColormap(list_color)

warnings.filterwarnings("ignore")
seasons =['DJF','MAM','JJA','SON']
seasons =['JJA']
path="/div/pdo/pdrmip/renamed/renamed/"

exps = ["1","2","3","4"] # Names of warming levels
#exps=['sulx10asia']
#print(glob(path))
#year_lin=50

models=['MPI-ESM1-2-LR','CanESM5','ACCESS-ESM1-5']
#models=[sys.argv[1]]
extends = {'Global':[-180,180,-90,90],'Asia':[43,150,2,60]}
limits2={'skew':[-1.2,1.2],'kurtosis':[-1,1]}
var_list={'tasmax':{'skew':[-0.1,0.1],'kurtosis':[-0.1,0.1]},'pr':{'skew':[-0.5,0.5],'kurtosis':[-8,8]}}
var_list={'tasmax':{'skew':[-5,5]},'pr':{'skew':[-1,1]}}
var_list={'pr':{'skew':[-2,2]}}
ssps = ['126','245','370','585']
cmap=make_cmap('RdBu_r',20)
cmap_t=make_cmap('RdBu_r',20)
cmap_pr=make_cmap("BrBG",20)
cmap_t='RdBu_r'
cmap_pr="BrBG"

for var,tmp in var_list.items():
	for diff,lims in tmp.items():#,'pr','rx5day','rx1day']:
			for region_name,limits in extends.items():
				for season_ind,season in enumerate(seasons):

					#axs = axs.flatten()
					for model in models:
						fig, axs = plt.subplots(3, 4, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
						ifile_base = xr.open_dataset('map_data_treshold_new/'+model+'_'+season+"_"+var+'_0_high_abs_mean_sift.nc')[var]
						for model_ind,times in enumerate([2025,2035,2045]):
							for exp_ind,ssp in enumerate(ssps):
								ifile = xr.open_dataset('map_data_treshold_new/'+model+'_'+season+"_"+var+'_'+ssp+"_"+str(times)+"_high_abs_mean_sift.nc")[var]							
								print(model,ssp)
								print(ifile)
								print(ifile_base)
								data = (ifile-ifile_base)*10
								land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(data)
								data=data.where(land_mask==0)

								#data = maskRegion(data,"SAH")
								#data = maskRegion(data,"GIC")
								#data = maskRegion(data,"NEN")

								fname='_'.join([model,ssp,str(times),season,var,'high','mean_sift'])
								data.to_dataset(name=var).to_netcdf('map_data/times/'+fname,format="NETCDF3_64bit")
								#print(data.max(),data.min())
								if var == "pr":
									cmap=cmap_pr
								else:
									cmap=cmap_t
								im = data.plot(ax=axs[model_ind,exp_ind],transform=ccrs.PlateCarree(),vmin=lims[0],vmax=lims[1],cmap=cmap,add_colorbar=False)
								axs[model_ind,exp_ind].coastlines()
								if model_ind==0:
									axs[model_ind,exp_ind].set_title(ssp,fontsize=18)
								else:
									axs[model_ind,exp_ind].set_title(ssp+" "+str(times))
								axs[model_ind,exp_ind].set_title(ssp+" "+str(times),fontsize=18)
								axs[model_ind,exp_ind].set_extent(limits, crs=ccrs.PlateCarree())
								#regionmask.defined_regions.ar6.land.plot(ax=axs[season_ind],label='abbrev', add_ocean=False, text_kws=text_kws);
								if exp_ind==0:
										axs[model_ind,exp_ind].text(-0.07, 0.55, str(times)+"-"+str(times+10), va='bottom', ha='center',
												rotation='vertical', rotation_mode='anchor',
												transform=axs[model_ind,exp_ind].transAxes,fontsize=18)
						cb_ax = fig.add_axes([0.03, 0.08, 0.9, 0.05])
						cbar = fig.colorbar(im, cax=cb_ax,orientation='horizontal',extend="both")
						cbar.ax.tick_params(labelsize=18)
						cbar.set_label('Change (%) in likelihood of pre-industial extreme-event(10yr)', fontsize=18)
						plt.subplots_adjust(top=0.93,
						bottom=0.11,
						left=0.05,
						right=0.95,
						hspace=0.01,
						wspace=0.04)
						plt.suptitle("Change due to change in mean",fontsize=18)
						plt.savefig('figures_new/threshold/'+var+"_"+model+"_"+region_name+"_multimodelens_"+season+"_threshold_high_ssp2_mean_sift.png")
						plt.close()

var_list={'tasmax':{'skew':[-5,5]},'pr':{'skew':[-30,30]}}
var_list={'pr':{'skew':[-1,5]}}
for var,tmp in var_list.items():
	for diff,lims in tmp.items():#,'pr','rx5day','rx1day']:
			for region_name,limits in extends.items():
				for season_ind,season in enumerate(seasons):

					#axs = axs.flatten()
					for model in models:
						fig, axs = plt.subplots(3, 4, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
						ifile_base = xr.open_dataset('map_data_treshold_new/'+model+'_'+season+"_"+var+'_0_high_abs_mean_sift.nc')[var]
						for model_ind,times in enumerate([2025,2035,2045]):
							for exp_ind,ssp in enumerate(ssps):
								ifile = xr.open_dataset('map_data_treshold_new/'+model+'_'+season+"_"+var+'_'+ssp+"_"+str(times)+"_high_abs_var.nc")[var]							
								print(model,ssp)
								print(ifile)
								print(ifile_base)
								data = (ifile-ifile_base)*10
								land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(data)
								data=data.where(land_mask==0)

								#data = maskRegion(data,"SAH")
								#data = maskRegion(data,"GIC")
								#data = maskRegion(data,"NEN")

								fname='_'.join([model,ssp,str(times),season,var,'high'])
								data.to_dataset(name=var).to_netcdf('map_data/times/'+fname,format="NETCDF3_64bit")
								#print(data.max(),data.min())
								if var == "pr":
									cmap=cmap_pr
								else:
									cmap=cmap_t
								norm = TwoSlopeNorm(vmin=lims[0], vcenter=0, vmax=lims[1])
								im = data.plot(ax=axs[model_ind,exp_ind],transform=ccrs.PlateCarree(),vmin=lims[0],vmax=lims[1],norm=norm,cmap=cmap,add_colorbar=False)
								axs[model_ind,exp_ind].coastlines()
								if model_ind==0:
									axs[model_ind,exp_ind].set_title(ssp,fontsize=18)
								else:
									axs[model_ind,exp_ind].set_title(ssp+" "+str(times))
								axs[model_ind,exp_ind].set_title(ssp+" "+str(times),fontsize=18)
								axs[model_ind,exp_ind].set_extent(limits, crs=ccrs.PlateCarree())
								#regionmask.defined_regions.ar6.land.plot(ax=axs[season_ind],label='abbrev', add_ocean=False, text_kws=text_kws);
								if exp_ind==0:
										axs[model_ind,exp_ind].text(-0.07, 0.55, str(times)+"-"+str(times+10), va='bottom', ha='center',
												rotation='vertical', rotation_mode='anchor',
												transform=axs[model_ind,exp_ind].transAxes,fontsize=18)
						cb_ax = fig.add_axes([0.03, 0.08, 0.9, 0.05])
						cbar = fig.colorbar(im, cax=cb_ax,orientation='horizontal',extend="both")
						cbar.ax.tick_params(labelsize=18)
						cbar.set_label('Change (%) in likelihood of pre-industial extreme-event(10yr)', fontsize=18)
						plt.subplots_adjust(top=0.93,
						bottom=0.11,
						left=0.05,
						right=0.95,
						hspace=0.01,
						wspace=0.04)
						plt.suptitle("Change due to change in variability",fontsize=18)
						plt.savefig('figures_new/threshold/'+var+"_"+model+"_"+region_name+"_multimodelens_"+season+"_threshold_high_ssp2.png")
						plt.close()


var_list={'pr':{'skew':[-1,5]}}
for var,tmp in var_list.items():
	for diff,lims in tmp.items():#,'pr','rx5day','rx1day']:
			for region_name,limits in extends.items():
				for season_ind,season in enumerate(seasons):

					#axs = axs.flatten()
					for model in models:
						fig, axs = plt.subplots(3, 4, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
						ifile_base = xr.open_dataset('map_data_treshold_new/'+model+'_'+season+"_"+var+'_0_high_abs_total.nc')[var]
						for model_ind,times in enumerate([2025,2035,2045]):
							for exp_ind,ssp in enumerate(ssps):
								ifile = xr.open_dataset('map_data_treshold_new/'+model+'_'+season+"_"+var+'_'+ssp+"_"+str(times)+"_high_abs_total.nc")[var]							
								print(model,ssp)
								print(ifile)
								print(ifile_base)
								data = (ifile-ifile_base)*10
								land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(data)
								data=data.where(land_mask==0)

								#data = maskRegion(data,"SAH")
								#data = maskRegion(data,"GIC")
								#data = maskRegion(data,"NEN")

								fname='_'.join([model,ssp,str(times),season,var,'high','total'])
								data.to_dataset(name=var).to_netcdf('map_data/times/'+fname,format="NETCDF3_64bit")
								#print(data.max(),data.min())
								if var == "pr":
									cmap=cmap_pr
								else:
									cmap=cmap_t
								norm = TwoSlopeNorm(vmin=lims[0], vcenter=0, vmax=lims[1])
								im = data.plot(ax=axs[model_ind,exp_ind],transform=ccrs.PlateCarree(),vmin=lims[0],vmax=lims[1],norm=norm,cmap=cmap,add_colorbar=False)
								axs[model_ind,exp_ind].coastlines()
								if model_ind==0:
									axs[model_ind,exp_ind].set_title(ssp,fontsize=18)
								else:
									axs[model_ind,exp_ind].set_title(ssp+" "+str(times))
								axs[model_ind,exp_ind].set_title(ssp+" "+str(times),fontsize=18)
								axs[model_ind,exp_ind].set_extent(limits, crs=ccrs.PlateCarree())
								#regionmask.defined_regions.ar6.land.plot(ax=axs[season_ind],label='abbrev', add_ocean=False, text_kws=text_kws);
								if exp_ind==0:
										axs[model_ind,exp_ind].text(-0.07, 0.55, str(times)+"-"+str(times+10), va='bottom', ha='center',
												rotation='vertical', rotation_mode='anchor',
												transform=axs[model_ind,exp_ind].transAxes,fontsize=18)
						cb_ax = fig.add_axes([0.03, 0.08, 0.9, 0.05])
						cbar = fig.colorbar(im, cax=cb_ax,orientation='horizontal',extend="both")
						cbar.ax.tick_params(labelsize=18)
						cbar.set_label('Change (%) in likelihood of pre-industial extreme-event(10yr)', fontsize=18)
						plt.subplots_adjust(top=0.93,
						bottom=0.11,
						left=0.05,
						right=0.95,
						hspace=0.01,
						wspace=0.04)
						plt.suptitle("Change due to change in variability",fontsize=18)
						plt.savefig('figures_new/threshold/'+var+"_"+model+"_"+region_name+"_multimodelens_"+season+"_threshold_high_ssp2_total.png")
						plt.close()


