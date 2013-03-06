"""
Utilities for converting time units
"""
import numpy as np
from iris.unit import Unit

cis_standard_time_unit = Unit('days since 1600-01-01 00:00:00',calendar='gregorian')

def parse_datetimestr_to_std_time(s):
    import dateutil.parser as du
    return cis_standard_time_unit.date2num(du.parse(s))


def parse_datetimestr_to_std_time_array(string_time_array):
    return convert_numpy_array(string_time_array, 'float64', parse_datetimestr_to_std_time)


def parse_datetimestr_delta_to_float_days(s):
    """
    parsing "1y2m3d4H5M6S" into a time delta represented as a relativedelta object
    @param s: string to be parsed
    @return: a relativedelta object
    """
    import re
    from datetime import timedelta

    tokens = re.findall('[0-9]*[ymdHMS]',s)
    
    days = hours = minutes = seconds = 0
    for token in tokens:

        val = int(token.replace('','')[:-1])
        if token[-1:] == "y":
            days += val*365.2425
            continue
        elif token[-1:] == "m":
            days += val*365.2425/12.0
            continue
        elif token[-1:] == "d":
            days += val
            continue
        elif token[-1:] == "H":
            hours = val
            continue
        elif token[-1:] == "M":
            minutes = val
            continue
        elif token[-1:] == "S":
            seconds = val
            continue
        else:
            raise ValueError("unvalid time delta format. Must be '1y2m3d4H5M6S'")

    td = timedelta(days=days,hours=hours,minutes=minutes,seconds=seconds)

    return td.days


def calculate_mid_time(t1, t2):
    import math
    from dateutil.relativedelta import relativedelta
    delta = relativedelta(t2,t1)
    total_seconds = delta.seconds + delta.minutes * 60 + delta.hours * 3600 + delta.days * 86400 + delta.months * 2592000 + delta.years * 31536000
    half = relativedelta(seconds=int(math.ceil(total_seconds / 2.0)))
    return t1 + half


def convert_time_since_to_std_time(time_array, units):
    from netcdftime import _dateparse
    import datetime
    units, utc_offset, dt = _dateparse(units)
    dt = datetime.datetime(*dt.timetuple()[0:6])
    if units == 'days':
        new_array = convert_numpy_array(time_array, 'float64', convert_days_since_to_std_time, dt)
    elif units == 'secs':
        new_array = convert_numpy_array(time_array, 'float64', convert_sec_since_to_std_time, dt)
    return new_array


def convert_sec_since_to_std_time_array(tai_time_array, ref):
    return convert_numpy_array(tai_time_array, 'float64', convert_sec_since_to_std_time, ref)


def convert_sec_since_to_std_time(seconds, ref):
    from datetime import timedelta
    return cis_standard_time_unit.date2num(timedelta(seconds=int(seconds)) + ref)


def convert_days_since_to_std_time(days, ref):
    from datetime import timedelta
    return cis_standard_time_unit.date2num(timedelta(days=float(days)) + ref)


def convert_std_time_to_datetime(std_time):
    return cis_standard_time_unit.num2date(std_time)


def convert_julian_date_to_std_time_array(julian_time_array, calender='standard'):
    return convert_numpy_array(julian_time_array, 'float64', convert_julian_date_to_std_time, calender)


def convert_julian_date_to_std_time(julian_date, calender='standard'):
    from iris.unit import julian_day2date
    return cis_standard_time_unit.date2num(julian_day2date(julian_date, calender))


def convert_obj_to_standard_date_array(time_array):
    return convert_numpy_array(time_array, 'float64', cis_standard_time_unit.date2num)


def convert_masked_array_type(masked_array, new_type, operation, *args, **kwargs):
    from numpy.ma import getmaskarray
    converted_time = np.ma.array(np.zeros(masked_array.shape, dtype=new_type),
                                 mask=masked_array.mask)

    masks = getmaskarray(masked_array)
    for i, val in np.ndenumerate(masked_array):
        if not masks[i]:
            converted_time[i] = operation(val, *args, **kwargs)
    return converted_time


def convert_array_type(array, new_type, operation, *args, **kwargs):
    converted_time = np.zeros(array.shape, dtype=new_type)
    for i, val in np.ndenumerate(array):
        converted_time[i] = operation(val, *args, **kwargs)
    return converted_time


def convert_numpy_array(array, new_type, operation, *args, **kwargs):
    if isinstance(array,np.ma.MaskedArray):
        new_array = convert_masked_array_type(array, new_type, operation, *args, **kwargs)
    else:
        new_array = convert_array_type(array, new_type, operation, *args, **kwargs)
    return new_array


def convert_cube_time_coord_to_standard_time(cube):
    # Get the current time coordinate and it's data dimension
    t_coord = cube.coord(standard_name='time')
    data_dim = cube.coord_dims(t_coord)

    # And remove it from the cube
    cube.remove_coord(t_coord)

    # Convert the raw time numbers to our 'standard' time
    new_datetimes = convert_numpy_array(t_coord.points,'O',t_coord.units.num2date)
    new_datetime_nums = convert_obj_to_standard_date_array(new_datetimes)

    # Create a new time coordinate by copying the old one, but using our new points and units
    new_time_coord = t_coord
    new_time_coord.points = new_datetime_nums
    new_time_coord.units = cis_standard_time_unit

    # And add the new coordinate back into the cube
    cube.add_dim_coord(new_time_coord, data_dim)

    return cube
