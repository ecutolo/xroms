import xarray as xr
import numpy as np
from numba import njit, prange

@xr.register_dataarray_accessor('xroms')
class xROMSDataArrayAccessor(object):
    
    def __init__(self, xarray_obj):
        self._data = xarray_obj
        
    def sel_geographic_area(self,area_limits):
        lon_coords = [c for c in self._data.coords if "lon" in c]
        lat_coords = [c for c in self._data.coords if "lat" in c]
        ds_l = self._data.copy()
        ds_l = ds_l.where(ds_l[lon_coords[0]]>area_limits[0],drop=True)
        ds_l = ds_l.where(ds_l[lon_coords[0]]<area_limits[1],drop=True)
        ds_l = ds_l.where(ds_l[lat_coords[0]]>area_limits[2],drop=True)
        ds_l = ds_l.where(ds_l[lat_coords[0]]<area_limits[3],drop=True)
        return ds_l
    
    def move2grid(self,final_grid,init_grid=False):
        '''
        tempu = move2grid(temp, 'rho', 'u')

        Move var from init_grid to final_grid.
        '''
        varin = self._data
        if not(init_grid):
            init_grid = self._data.name
            
        if (init_grid == 'rho' and final_grid == 'u'):
            daout = varin.isel(xi_rho=slice(None,-1)).copy()
            varout = 0.5 * (varin.isel(xi_rho=slice(1,None)).values+
                            varin.isel(xi_rho=slice(None,-1)).values)
        elif (init_grid == 'rho' and final_grid == 'v'):
            daout = varin.isel(eta_rho=slice(None,-1)).copy()
            varout = 0.5 * (varin.isel(eta_rho=slice(1,None)).values+
                            varin.isel(eta_rho=slice(None,-1)).values)
        elif (init_grid == 'rho' and final_grid == 'psi'):
            daout = varin.isel(eta_rho=slice(None,-1),xi_rho=slice(None,-1)).copy()
            varout = 0.25 * (varin.isel(xi_rho=slice(1,None),eta_rho=slice(1,None)).values+
                             varin.isel(xi_rho=slice(None,-1),eta_rho=slice(None,-1)).values+
                             varin.isel(xi_rho=slice(None,-1),eta_rho=slice(1,None)).values+
                             varin.isel(xi_rho=slice(1,None),eta_rho=slice(None,-1)).values)
        elif (init_grid == 'u' and final_grid == 'psi'):
            daout = varin.isel(eta_u=slice(None,-1)).copy()
            varout = 0.5 * (varin.isel(eta_u=slice(1,None)).values+
                            varin.isel(eta_u=slice(None,-1)).values)
        elif (init_grid == 'v' and final_grid == 'psi'):
            daout = varin.isel(xi_v=slice(None,-1)).copy()
            varout = 0.5 * (varin.isel(xi_v=slice(1,None)).values+
                            varin.isel(xi_v=slice(None,-1)).values)
        else:
            raise ValueError('Undefined combination for init_grid and final_grid')
        daout.values = varout
        return daout
    
@xr.register_dataset_accessor('xroms')
class xROMSDataSetAccessor(object):
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._lon_coords = [c for c in self._obj.coords if "lon" in c]
        self._lat_coords = [c for c in self._obj.coords if "lat" in c]
        self._3d_vars = [c for c,d in self._obj.items() if 's_rho' in d.coords and len(d.coords)>=3]
        self._geo_vars = [c for c,d in self._obj.items() if any('lat' in x or 'lon' in x for x in d.coords)]
        self._z_r = None
        
    def load_grid(self,grid_file):
        self._grid = xr.open_dataset(grid_file)
        self._obj['h'] = self._grid.h
        self._obj['hc'] = self._grid.hc
        self._obj['s_rho'] = self._grid.s_rho
        self._obj['Cs_r'] = self._grid.Cs_r
        
        @njit(parallel=True)
        def compute_z_r(hc,s_rho,Cs_r,h,zeta):
            z_r = np.ones((zeta.shape[0],s_rho.shape[0],
                            h.shape[0],h.shape[1]))
            for tt in prange(zeta.shape[0]):
                for  kk in prange(s_rho.shape[0]):
                    z0 = (hc * s_rho[kk] + h * Cs_r[kk]) / (hc + h)        
                    z_r[tt,kk,:] = zeta[tt,:]+(zeta[tt,:] + h)*z0
            return(z_r)
        
        zeta = self._obj.zeta.values
        if len(zeta.shape) < 3:
            zeta = np.tile(zeta,(1,1,1))
        z_r = compute_z_r(float(self._obj.hc.values),
                                self._obj.s_rho.values,
                                self._obj.Cs_r.values,
                                self._obj.h.values,
                                zeta)
        
        self._obj['z_r'] = xr.DataArray(z_r, coords=[self._obj.ocean_time,
                                                     self._obj.s_rho,
                                                     self._obj.eta_rho,
                                                     self._obj.xi_rho])
        print('Grid Data Loaded')


    def sel_zslice(self,depth):
        @njit(parallel=True)
        def fast_get_zslice(data,depth,z_r):
            int_data = np.ones((z_r.shape[0]+1,z_r.shape[2],z_r.shape[3]))*np.nan
            for t in prange(z_r.shape[0]):
                for i in prange(data.shape[3]):
                    for j in prange(data.shape[2]):
                        if depth > z_r[t,0,j,i]:
                            idx = np.searchsorted(z_r[t,:,j,i],depth, side="right")
                            delta_d1 = np.abs(depth-z_r[t,idx-1,j,i])
                            delta_d2 = np.abs(depth-z_r[t,idx+1,j,i])
                            int_data[t,j,i] = (data[t,idx+1,j,i]*delta_d1+
                                              data[t,idx-1,j,i]*delta_d2)/(delta_d1+delta_d2)
            return(int_data[:-1,:,:])
        
        ds_slice = self._obj.isel(s_rho=0).copy()
        ds_slice = ds_slice.rename({'s_rho':'depth'})
        ds_slice['depth'] = depth
        for varname in self._3d_vars:
            if varname is 'z_r': 
                continue
            elif varname in ('u','v'):
                z_r = self._obj.z_r.xroms.move2grid(varname,'rho').values
            else:
                z_r = self._obj.z_r.values
            
            data = self._obj[varname].values

            if len(data.shape) is 4 and len(z_r.shape) is 4:
                data_slice = fast_get_zslice(data,depth,z_r)
                ds_slice[varname].values = data_slice
            elif len(data.shape) is 3 and len(z_r.shape) is 3:
                data_slice = fast_get_zslice(np.tile(data,(1,1,1,1)),depth,np.tile(z_r,(1,1,1,1)))
                ds_slice[varname].values = data_slice[0]
            else:
                raise Exception("You need at least 3 dimensions to take a slice!")

        return(ds_slice)