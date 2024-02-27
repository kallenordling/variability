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
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import json
import cftime
import cdo
from matplotlib.ticker import PercentFormatter
import re

access=['r11i1p1f1',
'r12i1p1f1',
'r13i1p1f1',
'r14i1p1f1',
'r15i1p1f1',
'r16i1p1f1',
'r17i1p1f1',
'r18i1p1f1',
'r19i1p1f1',
'r20i1p1f1',
'r21i1p1f1',
'r22i1p1f1',
'r23i1p1f1',
'r24i1p1f1',
'r25i1p1f1',
'r26i1p1f1',
'r27i1p1f1',
'r28i1p1f1',
'r29i1p1f1',
'r30i1p1f1',
'r31i1p1f1',
'r32i1p1f1',
'r33i1p1f1',
'r34i1p1f1',
'r35i1p1f1',
'r36i1p1f1',
'r37i1p1f1',
'r38i1p1f1',
'r39i1p1f1',
'r40i1p1f1']

canesm=[
'r10i1p1f1',
'r11i1p1f1',
'r12i1p1f1',
'r13i1p1f1',
'r14i1p1f1',
'r15i1p1f1',
'r16i1p1f1',
'r17i1p1f1',
'r18i1p1f1',
'r19i1p1f1',
'r20i1p1f1',
'r21i1p1f1',
'r22i1p1f1',
'r23i1p1f1',
'r24i1p1f1',
'r25i1p1f1',
'r2i1p1f1',
'r3i1p1f1',
'r4i1p1f1',
'r5i1p1f1',
'r6i1p1f1',
'r7i1p1f1',
'r8i1p1f1',
'r9i1p1f1'
]

mpi=[
'r11i1p1f1',
'r12i1p1f1',
'r13i1p1f1',
'r14i1p1f1',
'r15i1p1f1',
'r16i1p1f1',
'r18i1p1f1',
'r19i1p1f1',
'r20i1p1f1',
'r21i1p1f1',
'r22i1p1f1',
'r23i1p1f1',
]

def glob_re(pattern, strings):
	return list(filter(re.compile(pattern).match, strings))

def find_nearest_index(array, value):

	array = np.asarray(array)
	tmp = np.where(np.abs(array - value) < 0.05)[0]
	#print(tmp)
	idx = np.nanargmin(np.abs(array - value))
	if tmp.size == 0:
		idx = np.nan

	
	return idx

def getFiles(model,ssp,var):
	path='/div/no-backup/CMIP6/CMIP6_downloads/ssp'+ssp+'/'+var+'/'
	#output=glob_re(r'tas_Amon_'+model+'_ssp'+ssp+'_r[1-9][1-9]i1p1f1_gr_201501_210012.nc', os.listdir(path))
	output=[] #glob_re(var+r'_day_'+model+'_ssp'+ssp+'_r([0-9][0-9]?|50)i1p1f1_(.*)_(.*)', os.listdir(path))
	if model == "MPI-ESM1-2-LR":
		ens=mpi
	if model == "CanESM5":
		ens=canesm
	if model == "ACCESS-ESM1-5":
		ens=access
	
	for i in ens:
		ifile=path+var+'_Amon_'+model+'_ssp'+ssp+'_'+i+'_gn_201501-210012.nc'
		if model == "CanESM5" and ens == "r11i1p1f1":
			ifile=path+var+'_Amon_'+model+'_ssp'+ssp+'_'+i+'_gn_201501-230012.nc'
		#print(ifile)
		if os.path.isfile(ifile):
			output.append(ifile)
	return output

def getFilesHist(model,ssp,var):
	path='/div/no-backup/CMIP6/CMIP6_downloads/'+ssp+'/'+var+'/'
	#output=glob_re(r'tas_Amon_'+model+'_ssp'+ssp+'_r[1-9][1-9]i1p1f1_gr_201501_210012.nc', os.listdir(path))
	output=[] #glob_re(var+r'_day_'+model+'_ssp'+ssp+'_r([0-9][0-9]?|50)i1p1f1_(.*)_(.*)', os.listdir(path))
	if model == "MPI-ESM1-2-LR":
		ens=mpi
	if model == "CanESM5":
		ens=canesm
	if model == "ACCESS-ESM1-5":
		ens=access
	
	for i in ens:
		ifile=path+var+'_Amon_'+model+'_'+ssp+'_'+i+'_gn_185001-201412.nc'
		if os.path.isfile(ifile):
			output.append(ifile)
	return output

def combineFiles(flist):
	dss=[]
	coords=[]
	for i in flist:
		coords.append(i.split('_')[5])
		dss.append(xr.open_dataset(i)['tas'])
	return xr.concat(dss,pd.Index(coords, name="memb"))

def calcMean(ds):
	weights = np.cos(np.deg2rad(ds.lat))
	ds_weighted = ds.weighted(weights)
	return ds_weighted.mean(("lon", "lat"))

plt.rcParams.update({'font.size': 22})

colors = sns.color_palette("Set1")
colors_ssp = sns.color_palette("Set2")
models=['CanESM5','MPI-ESM1-2-LR','ACCESS-ESM1-5']#,'EC-Earth3']
#models=['EC-Earth3']
#models=['MPI-ESM1-2-LR']
ssps=['585']
gwls={'0':[colors[0],0],'1':[colors[1],1],'1.5':[colors[2],1.5],'2':[colors[3],2],'3':[colors[4],3],'4':[colors[5],4]}
gwls={'0':[colors[0],0],'1':[colors[1],1],'2':[colors[3],2],'3':[colors[4],3],'4':[colors[5],4]}
#get CanESM time
model="CanESM5"
ssp='126'
flist=getFiles(model, ssp,'tas')
ssp_ds=combineFiles(flist)
flist=getFilesHist(model, 'historical','tas')
hist_ds=combineFiles(flist)
print(hist_ds)
dset_canesm = xr.concat([hist_ds,ssp_ds],dim='time').isel(time=slice(0,3012))
can_time=dset_canesm['time']

result_list = {}
for model in models:
	fig = plt.figure()
	result_list[model] = {}
	for ssp_i,ssp in enumerate(['585']):
		result_list[model][ssp] = {}
		limits={}
		for gwl,__ in gwls.items():
			result_list[model][ssp][gwl] = {}
			limits[gwl] = []
		flist=getFiles(model, ssp,'tas')
		print(flist,len(flist))
		if len(flist) < 10:
			continue
		if len(flist) > 25:
			flist=flist[0:25]
		print(ssp,model,len(flist))			
		ssp_ds=combineFiles(flist)
		print(ssp_ds)
		print("get hist  files")
		flist=getFilesHist(model, 'historical','tas')
		if len(flist) > 25:
			flist=flist[0:25]
		print("combine files")
		hist_ds=combineFiles(flist)
		print(hist_ds)
		print('calc anomaly')
		print(hist_ds.memb,ssp_ds.memb)
		dset = calcMean(xr.concat([hist_ds,ssp_ds],dim='time').isel(time=slice(0,3012)))
		dset['time'] = can_time
		dset = dset.groupby('time.year').mean()
		dset_anom = dset - dset.isel(year=slice(0,50)).mean('year')
		print(dset_anom)
		twentyyrmean = dset_anom.mean('memb').rolling(year=20,center=True).mean().values
		print(twentyyrmean)

		for i in range(0,dset_anom.values.shape[0]):
			print(i,dset_anom.values.shape[0])
			if i==0:
				plt.scatter(dset_anom.year,dset_anom.values[i,:],color='k',facecolor='none',zorder=-10)
			else:
				plt.scatter(dset_anom.year,dset_anom.values[i,:],color='k',facecolor='none',zorder=-10)
			for l,(gwl,color) in enumerate(gwls.items()):
				#print(twentyyrmean[:,:])
				idx = find_nearest_index(twentyyrmean[:],color[1])
				if np.isnan(idx):
					continue	
				idx_min = idx - 10
				idx_max = idx + 10
				if i==0:
					plt.scatter(dset_anom.year[idx_min:idx_max], dset_anom[i,idx_min:idx_max], color=color[0],zorder=10,label=gwl+r"$^\circ$ GWL")
				else:
					plt.scatter(dset_anom.year[idx_min:idx_max], dset_anom[i,idx_min:idx_max], color=color[0],zorder=10)
				limits[gwl].append(dset_anom.year[idx])
				print(np.squeeze(dset_anom[i,idx_min:idx_max].values).shape)
				result_list[model][ssp][gwl][str(dset_anom.memb[i].values)] = {'year':list(map(int,list(dset_anom.year[idx_min:idx_max].values))),'means':list(map(float,list(dset_anom[i,idx_min:idx_max].values))),}
		#for gwl,values in limits.items():
		#	if len(values) > 0:
		#		if ssp_i==0:
		#			plt.axvspan(np.min(values), np.max(values),-0.1,8, color=gwls[gwl][0], alpha=.5,label=gwl)
		#		else:
		#			plt.axvspan(np.min(values), np.max(values),-0.1,8, color=gwls[gwl][0], alpha=.5)
	plt.plot([1850,2100],[1,1],'--k',linewidth=3)
	plt.plot([1850,2100],[2,2],'--k',linewidth=3)
	plt.plot([1850,2100],[3,3],'--k',linewidth=3)
	plt.plot([1850,2100],[4,4],'--k',linewidth=3)
	plt.ylabel('Global mean \ntemperature change [$^\circ$ C]',fontsize=18)
	plt.legend(markerscale=2,ncol=2,fontsize=18)
	plt.xlim(1850,2100)
	plt.ylim(-0.1,8)
	print("save figure",model)
	plt.tight_layout()
	plt.savefig('/div/no-backup-nac/users/nordling/PDRMIP_variability/LESN/'+model+'.png')
	#plt.show()
	plt.close()
print(result_list)
with open("/div/no-backup-nac/users/nordling/PDRMIP_variability/LESN/gwl_data2.json", "w") as outfile:
    json.dump(result_list, outfile)

