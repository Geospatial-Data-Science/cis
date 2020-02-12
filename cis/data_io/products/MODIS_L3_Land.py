import logging
from cis.data_io import hdf as hdf
from cis.data_io.Coord import CoordList, Coord
from cis.data_io.products import AProduct
from cis.data_io.ungridded_data import UngriddedCoordinates, UngriddedData


def _get_MODIS_SDS_data(sds):
    """
    Reads raw data from an SD instance.

    :param sds: The specific sds instance to read
    :return: A numpy array containing the raw data with missing data is replaced by NaN.
    """
    from cis.utils import create_masked_array_for_missing_data
    import numpy as np

    data = sds.get()
    attributes = sds.attributes()

    # Apply Fill Value
    missing_value = attributes.get('_FillValue', None)
    if missing_value is not None:
        data = create_masked_array_for_missing_data(data, missing_value)

    # Check for valid_range
    valid_range = attributes.get('valid_range', None)
    if valid_range is not None:
        logging.debug("Masking all values {} > v > {}.".format(*valid_range))
        data = np.ma.masked_outside(data, *valid_range)

    # Offsets and scaling.
    add_offset = attributes.get('add_offset', 0.0)
    scale_factor = attributes.get('scale_factor', 1/10000.0)       # might need to change?
    data = _apply_scaling_factor_MODIS(data, scale_factor, add_offset)

    return data


def _apply_scaling_factor_MODIS(data, scale_factor, offset):
    """
    Apply scaling factor (applicable to MODIS data) of the form:
    ``data = (data - offset) * scale_factor``

    Ref:
    MODIS Atmosphere L3 Gridded Product Algorithm Theoretical Basis Document,
    MODIS Algorithm Theoretical Basis Document No. ATBD-MOD-30 for
    Level-3 Global Gridded Atmosphere Products (08_D3, 08_E3, 08_M3)
    by PAUL A. HUBANKS, MICHAEL D. KING, STEVEN PLATNICK, AND ROBERT PINCUS
    (Collection 005 Version 1.1, 4 December 2008)

    :param data: A numpy array like object
    :param scale_factor:
    :param offset:
    :return: Scaled data
    """
    logging.debug("Applying 'science_data = (packed_data - {offset}) * {scale}' "
                  "transformation to data.".format(scale=scale_factor, offset=offset))
    return (data - offset) * scale_factor


class MODIS_L3_Land(AProduct):
    """
    Data product for MODIS Level 3 data
    """

    def _parse_datetime(self, metadata_dict, keyword):
        import re
        res = ""
        for s in metadata_dict.values():
            i_start = s.find(keyword)
            ssub = s[i_start:len(s)]
            i_end = ssub.find("END_OBJECT")
            ssubsub = s[i_start:i_start + i_end]
            matches = re.findall('".*"', ssubsub)
            if len(matches) > 0:
                res = matches[0].replace('\"', '')
                if res is not "":
                    break
        return res

    def _get_start_date(self, filename):
        from cis.parse_datetime import parse_datetimestr_to_std_time
        metadata_dict = hdf.get_hdf4_file_metadata(filename)
        date = self._parse_datetime(metadata_dict, 'RANGEBEGINNINGDATE')
        time = self._parse_datetime(metadata_dict, 'RANGEBEGINNINGTIME')
        datetime_str = date + " " + time
        return parse_datetimestr_to_std_time(datetime_str)

    def _get_end_date(self, filename):
        from cis.parse_datetime import parse_datetimestr_to_std_time
        metadata_dict = hdf.get_hdf4_file_metadata(filename)
        date = self._parse_datetime(metadata_dict, 'RANGEENDINGDATE')
        time = self._parse_datetime(metadata_dict, 'RANGEENDINGTIME')
        datetime_str = date + " " + time
        return parse_datetimestr_to_std_time(datetime_str)

    def get_file_signature(self):
        product_names = [ "MCD15A2"]
        regex_list = [r'.*' + product + '.*\.hdf' for product in product_names]
        return regex_list

    def get_variable_names(self, filenames, data_type=None):
        try:
            from pyhdf.SD import SD
        except ImportError:
            raise ImportError("HDF support was not installed, please reinstall with pyhdf to read HDF files.")

        variables = set([])
        for filename in filenames:
            sd = SD(filename)
            for var_name, var_info in sd.datasets().items():
                # Check that the dimensions are correct
                if var_info[0] == ('YDim_MOD_Grid_MOD15A2H', 'XDim_MOD_Grid_MOD15A2H'):
                    variables.add(var_name)

        return variables

    def _create_cube(self, filenames, variable):
        import numpy as np
        from cis.data_io.hdf import _read_hdf4
        from iris.cube import Cube, CubeList
        from iris.coords import DimCoord, AuxCoord
        from cis.time_util import calculate_mid_time, cis_standard_time_unit
        from cis.data_io.hdf_sd import get_metadata
        from cf_units import Unit

        variables = ['XDim', 'YDim', variable]
        logging.info("Listing coordinates: " + str(variables))

        cube_list = CubeList()
        # Read each file individually, let Iris do the merging at the end.
        for f in filenames:
            sdata, vdata = _read_hdf4(f, variables)

            lat_coord = DimCoord(_get_MODIS_SDS_data(sdata['YDim_MOD_Grid_MOD15A2H']), standard_name='latitude', units='degrees')
            lon_coord = DimCoord(_get_MODIS_SDS_data(sdata['XDim_MOD_Grid_MOD15A2H']), standard_name='longitude', units='degrees')

            # create time coordinate using the midpoint of the time delta between the start date and the end date
            start_datetime = self._get_start_date(f)
            end_datetime = self._get_end_date(f)
            mid_datetime = calculate_mid_time(start_datetime, end_datetime)
            logging.debug("Using {} as datetime for file {}".format(mid_datetime, f))
            time_coord = AuxCoord(mid_datetime, standard_name='time', units=cis_standard_time_unit,
                                  bounds=[start_datetime, end_datetime])

            var = sdata[variable]
            metadata = get_metadata(var)

            try:
                units = Unit(metadata.units)
            except ValueError:
                logging.warning("Unable to parse units '{}' in {} for {}.".format(metadata.units, f, variable))
                units = None

            cube = Cube(_get_MODIS_SDS_data(sdata[variable]),
                        dim_coords_and_dims=[(lon_coord, 1), (lat_coord, 0)],
                        aux_coords_and_dims=[(time_coord, None)],
                        var_name=metadata._name, long_name=metadata.long_name, units=units)

            cube_list.append(cube)

        # Merge the cube list across the scalar time coordinates before returning a single cube.
        return cube_list.merge_cube()

    def create_coords(self, filenames, variable=None):
        """Reads the coordinates on which a variable depends.
        Note: This calls create_data_object because the coordinates are returned as a Cube.
        :param filenames: list of names of files from which to read coordinates
        :param variable: name of variable for which the coordinates are required
        :return: iris.cube.Cube
        """
        if variable is None:
            variable_names = self.get_variable_names(filenames)
            variable_name = str(variable_names.pop())
            logging.debug("Reading an IRIS Cube for the coordinates based on the variable %s" % variable_names)
        else:
            variable_name = variable

        return self.create_data_object(filenames, variable_name)

    def create_data_object(self, filenames, variable):
        from cis.data_io.gridded_data import make_from_cube
        logging.debug("Creating data object for variable " + variable)

        cube = self._create_cube(filenames, variable)
        return make_from_cube(cube)

    def get_file_format(self, filename):
        return "HDF4/ModisL3"

 