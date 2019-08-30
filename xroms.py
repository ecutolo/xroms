from typing import List, Tuple

import xarray as xr
import numpy as np
from numba import njit, prange


@xr.register_dataarray_accessor('xroms')
class xROMSDataArrayAccessor(object):
    '''
    Xarray DataArray extension for dealing with ROMS output files.

    Parameters
    ----------
    xarray_obj : xarray.DataArray

    Attributes
    ----------
    _data : xarray.DataArray
         The parameter xarray_obj is stored at this attribute.
         It represents the ROMS data.
    '''

    def __init__(self, xarray_obj):
        self._data = xarray_obj

    def sel_geographic_area(self, area_limits=[None, None, None, None]):
        ''' Function to select a geographic area inside the grid limits.

        Parameters
        ----------
        area_limits : tuple of float
             Tuple of flots with the values in the following order:
               (mininal logitude, maximum longitude,
                mininal latitude, maximum latitude)
             It is not necessary to pass all limits.
             If some of the limits of area_limits are None,
             the algorithm gets the limits from the grid original.
        '''

        lon_coords = [c for c in self._data.coords if "lon" in c]
        lat_coords = [c for c in self._data.coords if "lat" in c]
        min_lon, max_lon, min_lat, max_lat = area_limits

        '''
        If some of the limits are None,
        the select area will use the rho points as limits.
        '''
        if min_lon is None:
            min_lon = self._data.coords['lon_rho'].min()
        if max_lon is None:
            max_lon = self._data.coords['lon_rho'].max()
        if min_lat is None:
            min_lat = self._data.coords['lon_rho'].min()
        if max_lat is None:
            max_lat = self._data.coords['lon_rho'].max()

        ds_l = self._data.where(ds_l[lon_coords[0]] > area_limits[0],
                                drop=True)
        ds_l = ds_l.where(ds_l[lon_coords[0]] < area_limits[1], drop=True)
        ds_l = ds_l.where(ds_l[lat_coords[0]] > area_limits[2], drop=True)
        ds_l = ds_l.where(ds_l[lat_coords[0]] < area_limits[3], drop=True)
        return ds_l

    def move2grid(self, final_grid, init_grid=False,
                  x_prefix='xi', y_prefix='eta'):
        '''
        tempu = move2grid(temp, 'rho', 'u')

        Move var from init_grid to final_grid.
        '''
        varin = self._data
        if not(init_grid):
            init_grid = self._data.name

        if init_grid is final_grid:
            return varin

        if (init_grid == 'rho' and final_grid == 'u'):
            daout = varin.isel(xi_rho=slice(None, -1)).copy()
            varout = 0.5 * (varin.isel(xi_rho=slice(1, None)).values +
                            varin.isel(xi_rho=slice(None, -1)).values)
        elif (init_grid == 'rho' and final_grid == 'v'):
            daout = varin.isel(eta_rho=slice(None, -1)).copy()
            varout = 0.5 * (varin.isel(eta_rho=slice(1, None)).values +
                            varin.isel(eta_rho=slice(None, -1)).values)
        elif (init_grid == 'rho' and final_grid == 'psi'):
            daout = varin.isel(eta_rho=slice(None, -1),
                               xi_rho=slice(None, -1)).copy()
            varout = 0.25 * (varin.isel(xi_rho=slice(1, None), eta_rho=slice(1, None)).values +
                             varin.isel(xi_rho=slice(None, -1), eta_rho=slice(None, -1)).values +
                             varin.isel(xi_rho=slice(None, -1), eta_rho=slice(1, None)).values +
                             varin.isel(xi_rho=slice(1, None), eta_rho=slice(None, -1)).values)
        elif (init_grid == 'u' and final_grid == 'psi'):
            daout = varin.isel(eta_u=slice(None, -1)).copy()
            varout = 0.5 * (varin.isel(eta_u=slice(1, None)).values +
                            varin.isel(eta_u=slice(None, -1)).values)
        elif (init_grid == 'v' and final_grid == 'psi'):
            daout = varin.isel(xi_v=slice(None, -1)).copy()
            varout = 0.5 * (varin.isel(xi_v=slice(1, None)).values +
                            varin.isel(xi_v=slice(None, -1)).values)
        else:
            raise ValueError('Undefined combination for:\
                              init_grid and final_grid')
        daout = daout.rename({'{0}_{1}'.format(x_prefix, init_grid):
                              '{0}_{1}'.format(x_prefix, final_grid),
                              '{0}_{1}'.format(y_prefix, init_grid):
                              '{0}_{1}'.format(y_prefix, final_grid)})
        daout.values = varout
        return daout


@xr.register_dataset_accessor('xroms')
class xROMSDataSetAccessor(object):
    '''
    Xarray DataSet extension for dealing with ROMS output files.

    Parameters
    ----------
    xarray_obj : xarray.DataSet

    Attributes
    ----------
    _data : xarray.DataSet
         The parameter xarray_obj is stored at this attribute.
         It represents the ROMS data.
    '''

    def __init__(self, xarray_obj):
        self._data = xarray_obj
        self._lon_coords = [c for c in self._data.coords if "lon" in c]
        self._lat_coords = [c for c in self._data.coords if "lat" in c]
        self._3d_vars = [c for c, d in self._data.items()
                         if 's_rho' in d.coords and len(d.coords) >= 3]
        self._geo_vars = [c for c, d in self._data.items()
                          if any('lat' in x or 'lon' in x for x in d.coords)]
        self._z_r = None

    def load_grid_variable(self, variable: str):
        if variable not in self._grid:
            print("Variable: '{}' not found in grid file. Ignoring it!".
                  format(variable))
            return
        self._data[variable] = self._grid[variable]

    def load_grid(self, grid_file: str, variables: List[str] = None):
        self._grid = xr.open_dataset(grid_file)

        if variables is None:
            variables = [k for k in self._grid.keys()]

        for variable in variables:
            self.load_grid_variable(variable)

        zeta = self._data.zeta.values
        if len(zeta.shape) < 3:
            zeta = np.tile(zeta, (1, 1, 1))

        ''' If s_rho is presented, the grid is for a 3D model '''
        if 's_rho' in self._grid:
            z_r = compute_z_r(float(self._data.hc.values),
                              self._data.s_rho.values,
                              self._data.Cs_r.values,
                              self._data.h.values,
                              zeta)

            self._data['z_r'] = xr.DataArray(z_r, coords=[self._data.ocean_time,
                                                          self._data.s_rho,
                                                          self._data.eta_rho,
                                                          self._data.xi_rho])
        print('Grid Data Loaded')

    @njit(parallel=True)
    def compute_z_r(hc, s_rho, Cs_r, h, zeta):
        z_r = np.ones((zeta.shape[0], s_rho.shape[0], h.shape[0], h.shape[1]))
        for tt in prange(zeta.shape[0]):
            for kk in prange(s_rho.shape[0]):
                z0 = (hc * s_rho[kk] + h * Cs_r[kk]) / (hc + h)
                z_r[tt, kk, :] = zeta[tt, :] + (zeta[tt, :] + h) * z0
        return(z_r)

    def sel_geographic_area(self, area_limits):
        lon_dims = [d for d in self._data.dims if "xi" in d]
        lat_dims = [d for d in self._data.dims if "eta" in d]
        mask = {}
        mask['xi_rho'] = np.logical_and(np.unique(self._data['lon_rho']) >
                                        area_limits[0],
                                        np.unique(self._data['lon_rho']) <
                                        area_limits[1])
        mask['eta_rho'] = np.logical_and(np.unique(self._data['lat_rho']) >
                                         area_limits[2],
                                         np.unique(self._data['lat_rho']) <
                                         area_limits[3])

        for d_lon, d_lat in zip(lon_dims, lat_dims):
            mask[d_lon] = self._data.sel(mask).xi_rho.values
            mask[d_lat] = self._data.sel(mask).eta_rho.values
            if 'u' in d_lon or 'psi' in d_lon:
                mask[d_lon] = mask[d_lon][:-1]
            if 'v' in d_lat or 'psi' in d_lat:
                mask[d_lat] = mask[d_lat][:-1]
        return self._data.sel(mask)

    @njit(parallel=True)
    def fast_get_zslice(self, data, depth, z_r):
        int_data = np.nan * np.ones((z_r.shape[0] + 1,
                                     z_r.shape[2],
                                     z_r.shape[3]))
        for t in prange(z_r.shape[0]):
            for i in prange(data.shape[3]):
                for j in prange(data.shape[2]):
                    if depth > z_r[t, 0, j, i]:
                        idx = np.searchsorted(z_r[t, :, j, i],
                                              depth, side="right")
                        delta_d1 = np.abs(depth - z_r[t, idx - 1, j, i])
                        delta_d2 = np.abs(depth - z_r[t, idx + 1, j, i])
                        int_data[t, j, i] = (data[t, idx + 1, j, i] *
                                             delta_d1 +
                                             data[t, idx - 1, j, i] *
                                             delta_d2) / (delta_d1 + delta_d2)
        return(int_data[:-1, :, :])

    def sel_zslice(self, depth):
        ds_slice = self._data.isel(s_rho=0).copy()
        ds_slice = ds_slice.rename({'s_rho': 'depth'})
        ds_slice['depth'] = depth
        for varname in self._3d_vars:
            if varname is 'z_r':
                continue
            elif varname in ('u', 'v'):
                z_r = self._data.z_r.xroms.move2grid(varname, 'rho').values
            else:
                z_r = self._data.z_r.values

            data = self._data[varname].values

            if len(data.shape) is 4 and len(z_r.shape) is 4:
                data_slice = self.fast_get_zslice(data, depth, z_r)
                ds_slice[varname].values = data_slice
            elif len(data.shape) is 3 and len(z_r.shape) is 3:
                data_slice = self.fast_get_zslice(np.tile(data, (1, 1, 1, 1)),
                                                  depth, np.tile(z_r, (1, 1, 1, 1)))
                ds_slice[varname].values = data_slice[0]
            else:
                raise Exception("You need at least 3 \
                                dimensions to take a slice!")

        return(ds_slice)

    def compute_vorticity(self):
        pm = self._data.pm.xroms.move2grid('psi', 'rho')
        pn = self._data.pn.xroms.move2grid('psi', 'rho')
        DDxdiff = (pm**-1)
        DDydiff = (pn**-1)

        if 'lon_uv' in self._data.u.dims:
            u = self._data.u.xroms.move2grid('psi', 'rho')
            v = self._data.v.xroms.move2grid('psi', 'rho')
            UUX, UUY = u.differentiate('lat_psi'), u.differentiate('lon_psi')
            VVX, VVY = v.differentiate('lat_psi'), v.differentiate('lon_psi')
        else:
            u = self._data.u.xroms.move2grid('psi')
            v = self._data.v.xroms.move2grid('psi')
            UUX, UUY = u.differentiate('eta_psi'), u.differentiate('xi_psi')
            VVX, VVY = v.differentiate('eta_psi'), v.differentiate('xi_psi')

        omega = 7.2921E-5
        lat_2d = self._data.lat_rho.xroms.move2grid('psi', 'rho')
        f_matrix = 2 * omega * np.sin(np.deg2rad(lat_2d))

        vorticity = (VVX / (DDxdiff) - UUY / (DDydiff)) / f_matrix
        return vorticity
