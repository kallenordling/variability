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
import json
import cftime
import cdo
import re
import scipy
import datetime

def is_DJF(month):
    return (month >= 12) | (month <= 2)
    
def is_MAM(month):
    return (month >= 3) & (month <= 5)
    

def is_JJA(month):
    return (month >= 6) & (month <= 8)

def is_JJAS(month):
    return (month >= 6) & (month <= 9) 

def is_SON(month):
    return (month >= 10) & (month <= 11)
    
def is_JAN(month):
   return (month > 0) & (month <= 31)
   
def is_FEB(month):
   return (month > 31) & (month <= 31+28)
   
def is_MAR(month):
   return (month > 31+28) & (month <= 31+28+31)
   
def is_APR(month):
   return (month > 31+28+31) & (month <= 31+28+31+30)
   
def is_MAY(month):
   return (month > 31+28+31+30) & (month <= 31+28+31+30+31)
   
def is_JUN(month):
   return (month > 31+28+31+30+31) & (month <= 31+28+31+30+31+30)
   
def is_JUL(month):
   return (month > 31+28+31+30+31+30) & (month <= 31+28+31+30+31+30+31)  

def is_AUG(month):
   return (month > 31+28+31+30+31+30+31) & (month <= 31+28+31+30+31+30+31+31)   
   
def is_SEP(month):
   return (month > 31+28+31+30+31+30+31+31) & (month <= 31+28+31+30+31+30+31+31+30) 
   
def is_OCT(month):
   return (month > 31+28+31+30+31+30+31+31+30) & (month <= 31+28+31+30+31+30+31+31+30+31)
   
def is_NOV(month):
   return (month > 31+28+31+30+31+30+31+31+30+31) & (month <= 31+28+31+30+31+30+31+31+30+31+30)
   
def is_DEC(month):
   return (month > 31+28+31+30+31+30+31+31+30+31+30) & (month <= 31+28+31+30+31+30+31+31+30+31+30+31)


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

def rx1day(data):
	return data.max('doy')

def rx5day(data):
	return data.rolling({"doy": 5}, min_periods=1, center=True).sum(skipna=True).max('doy',skipna=True)
 

def to_monthly(ds,years_id):

	year = ds.time.dt.year.data
	doy = ds.time.dt.dayofyear.data
	#print(year,doy)
	# assign new coords
	ds = ds.assign_coords(year=("time", year), doy=("time", doy))

    # reshape the array to (..., "month", "year")

	return ds.set_index(time=("year", "doy")).unstack("time") 

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
		ifile=path+var+'_day_'+model+'_ssp'+ssp+'_'+i+'_gn_20150101-21001231.nc'
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
		ifile=path+var+'_day_'+model+'_'+ssp+'_'+i+'_gn_18500101-20141231.nc'
		if os.path.isfile(ifile):
			output.append(ifile)
	return output

def combineFiles(flist,var):
	dss=[]
	coords=[]
	for i in flist:
		coords.append(i.split('_')[5])
		ds=xr.open_dataset(i, chunks={"time": 10})[var]
		dss.append(ds)
	if len(dss) > 0:
		return xr.concat(dss,pd.Index(coords, name="memb"))
	else:
		return None

def constructDS2(ssp_ds,gwl_info):
	output_ds=[]
	input_id=1700
	ds_tmp=[]
	reference_time = pd.Timestamp("1700-01-01")
	print(ssp_ds)
	for memb,years in gwl_info.items():
		if memb.strip() in list(ssp_ds.memb.values):
			tmp_ds = ssp_ds.sel(memb=memb).sel(time=ssp_ds.time.dt.year.isin(years))
			#print(tmp_ds)
			years_id = np.arange(input_id,input_id+10,1)	

			start_y=str(input_id)+"-01-01"
			time_new = pd.date_range(start =start_y, freq="D", periods=len(tmp_ds.time.data))
			data = tmp_ds.data

			da = xr.DataArray(
					data=data,
					dims=["time", "lat", "lon"],
					coords=dict(
						lon=(["lon"], tmp_ds.lon.data),
						lat=(["lat"], tmp_ds.lat.data),
						time=time_new,
						reference_time=reference_time,
					),
				)
			#da.time.attrs['units'] = "days since 1700-01-01"
			ds_tmp.append(da)
			tmp_ds = xr.concat(ds_tmp,dim='time')	
			output_ds.append(tmp_ds)
			input_id+=11


	if len(output_ds) > 0:
		final_ds = xr.concat(output_ds,dim='time')
		#print(final_ds,'final_ds')
		return final_ds
	else:
		return None


def constructDS(ssp_ds):
	output_ds=[]
	input_id=1700
	ds_tmp=[]
	reference_time = pd.Timestamp("1700-01-01")

	for memb in range(0,ssp_ds.memb.size):
			tmp_ds = tmp_ds = ssp_ds.isel(memb=memb).sel(time=ssp_ds.time.dt.year.isin([np.unique(ssp_ds.time.dt.year)]))
			years_id = np.arange(input_id,input_id+10,1)	

			start_y=str(input_id)+"-01-01"
			time_new = pd.date_range(start =start_y, freq="D", periods=len(tmp_ds.time.data))
			data = tmp_ds.data

			da = xr.DataArray(
					data=data,
					dims=["time", "lat", "lon"],
					coords=dict(
						lon=(["lon"], tmp_ds.lon.data),
						lat=(["lat"], tmp_ds.lat.data),
						time=time_new,
						reference_time=reference_time,
					),
				)
			#da.time.attrs['units'] = "days since 1700-01-01"
			ds_tmp.append(da)
			tmp_ds = xr.concat(ds_tmp,dim='time')	
			output_ds.append(tmp_ds)
			input_id+=11


	if len(output_ds) > 0:
		final_ds = xr.concat(output_ds,dim='time')
		print(final_ds,'final_ds')
		return final_ds
	else:
		return None

def glob_re(pattern, strings):
	return list(filter(re.compile(pattern).match, strings))
   
def calcMean(data):
	#data = data.where(data > 1e-5)
	weights = np.cos(np.deg2rad(data.lat))
	data_weighted = data.weighted(weights)
	return data_weighted.mean(("lon", "lat"),skipna=True)
    

def maskRegion(ds,region):
	ar6_land=regionmask.defined_regions.ar6.land.mask(ds)
	reg_num =regionmask.defined_regions.ar6.land.map_keys(region)
	return ds.where(ar6_land != reg_num)

def plotting(ds):
	proj = ccrs.PlateCarree()
	ax = plt.subplot(111, projection=proj)
	ds.isel(time=1).plot(ax=ax, transform=ccrs.PlateCarree())
	ax.coastlines()
	plt.savefig("test.png")
	input('finish:')

#plt.rcParams.update({'font.size': 30})



#path="/div/pdo/pdrmip/renamed/renamed/"


models=['MPI-ESM1-2-LR','MPI-ESM1-2-LR']#,'CanESM5']
models=['CanESM5','ACCESS-ESM1-5']
models=[sys.argv[1]]
#gwls={'1.5':1.5,'2':2,'3':3,'4':4}
varlist=['tasmax']
 
f = open('gwl_data.json')


gwl_info = json.load(f)
seasons ={'JJA':is_JJA,'DJF':is_DJF,'MAM':is_MAM,'SON':is_SON}

seasons ={'JJA':is_JJA}#,'DJF':is_DJF}

#gwls={'1.5':1.5,'2':2,'3':3,'4':4}

#for var in ['pr']:
models=["ACCESS-ESM1-5"]
models=[sys.argv[1]]#,"ACCESS-ESM1-5",'CanESM5']
attrs = {"units": "days since 1700-01-01"}
for var in [sys.argv[2]]:#['pr','tasmax']:
	var_orig=var
	for model in models:
		for ssp in ['126','245',"370",'585']:
		#for ssp in ["370",'585']:#[2025,2035,2045]:
			#print(model,ssp,name,season,var)
			flist=getFiles(model, ssp,var_orig)
			#input('wait')
			ssp_ds=combineFiles(flist,var_orig)
			print(ssp_ds)
			print("ssp_ds")
			flist=getFilesHist(model, 'historical',var_orig)
			print("\n".join(flist))
			historical_ds=combineFiles(flist,var_orig)
			ssp_ds2=xr.concat([historical_ds,ssp_ds],dim='time')
			tmp_gwl_info_base =gwl_info[model]['585']['0']
			tmp_ds_base = constructDS2(ssp_ds2,tmp_gwl_info_base)
			tmp_ds_base2 =tmp_ds_base.copy(deep=True)
			tmp_ds_base = tmp_ds_base.groupby(tmp_ds_base.time.dt.dayofyear) - tmp_ds_base.groupby(tmp_ds_base.time.dt.dayofyear).mean() 

			for times in [2025,2035,2045]:#[2025,2035,2045]:
				tmp_ds = ssp_ds.sel(time=ssp_ds.time.dt.year.isin(np.arange(times,times+10,1)))#[list(ssp_ds.keys())[0]]
				tmp_ds = constructDS(tmp_ds)

				ds_annual2 = tmp_ds.groupby(tmp_ds.time.dt.dayofyear) -  tmp_ds_base2.groupby(tmp_ds_base2.time.dt.dayofyear).mean()

				#diff=tmp_ds.groupby(tmp_ds.time.dt.dayofyear).mean() - tmp_ds_pi.groupby(tmp_ds_pi.time.dt.dayofyear).mean()

				ds_annual =tmp_ds_base 


				####NEW entry####
				ds_annual =tmp_ds_base#.groupby(tmp_ds_base.time.dt.dayofyear) -  tmp_ds_base.groupby(tmp_ds_base.time.dt.dayofyear).mean() 
				ds_annual2 = tmp_ds.groupby(tmp_ds.time.dt.dayofyear) -  tmp_ds_base2.groupby(tmp_ds_base2.time.dt.dayofyear).mean() #total change
				ds_annual3 = tmp_ds.groupby(tmp_ds.time.dt.dayofyear) -  tmp_ds.groupby(tmp_ds.time.dt.dayofyear).mean() #change due variability
				ds_annual =	tmp_ds_base
				###END NEW ENTRY###

				#ds_annual = tmp_ds

				#if var == "pr":
				#	ds_annual *= 86400.0
				for season,fun in seasons.items():
					tmp1=ds_annual2.sel(time=fun(ds_annual2['time.month']))
					tmp2=ds_annual.sel(time=fun(ds_annual['time.month']))
					diff = tmp1.groupby(tmp1.time.dt.dayofyear).mean() -tmp2.groupby(tmp2.time.dt.dayofyear).mean()							
					ds_tmp=ds_annual.sel(time=fun(ds_annual['time.month']))+diff.mean('dayofyear',skipna=True)
					ds_tmp2=ds_annual2.sel(time=fun(ds_annual2['time.month']))
					ds_tmp3=ds_annual3.sel(time=fun(ds_annual3['time.month']))
					if var == "pr":
						ds_tmp *= 86400.0
						ds_tmp2 *= 86400.0
						ds_tmp3 *= 86400.0
					print(ds_tmp)
					#ds_tmp['time'] = ds_tmp['time'].astype(datetime.datetime)
					#ds_tmp = maskRegion(ds_tmp,"SAH")
					#ds_tmp = maskRegion(ds_tmp,"GIC")
					#ds_tmp = maskRegion(ds_tmp,"NEN")
					print('.......WRITING......',season,str(times),model,ssp,var)
					print(ds_tmp)
					ds_tmp['time'] = ds_tmp['time'].astype(datetime.datetime)
					ds_tmp2['time'] = ds_tmp2['time'].astype(datetime.datetime)
					ds_tmp3['time'] = ds_tmp3['time'].astype(datetime.datetime)
					ds_tmp.to_netcdf('/div/no-backup-nac/users/nordling/PDRMIP_variability/LESN/map_data_new/'+model+"_data_"+season+"_"+ssp+"_"+str(times)+"_"+var+"_mean_sift.nc",format="NETCDF4")
					ds_tmp2.to_netcdf('/div/no-backup-nac/users/nordling/PDRMIP_variability/LESN/map_data_new/'+model+"_data_"+season+"_"+ssp+"_"+str(times)+"_"+var+"_total.nc",format="NETCDF4")
					ds_tmp3.to_netcdf('/div/no-backup-nac/users/nordling/PDRMIP_variability/LESN/map_data_new/'+model+"_data_"+season+"_"+ssp+"_"+str(times)+"_"+var+"_var.nc",format="NETCDF4")
				

