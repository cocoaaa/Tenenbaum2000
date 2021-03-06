import math
from geopy.geocoders import Nominatim
from typing import Tuple

def getGeoFromTile(x, y, zoom):
	lon_deg = x / (2.0 ** zoom) * 360.0 - 180.0
	lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / (2.0 ** zoom))))
	lat_deg = lat_rad * (180.0 / math.pi)
	return lat_deg, lon_deg

def getCountryFromTile(x,y,zoom) -> str:
	"""Given x,y,z tile coords, return cityname"""
	lat_deg, lng_deg = getGeoFromTile(x,y,zoom)
	geolocator = Nominatim(user_agent="temp")
	location = geolocator.reverse(f"{lat_deg}, {lng_deg}")
	city = location.address.split(" ")[-1]
	return city

def test_getCountryFromTile():
    print("shanghai: ", getCountryFromTile(13703, 6671, 14))
    print("paris: ", getCountryFromTile(8301, 5639, 14))


def coord2xyz(
	coord_str: str,
	delimiter:str = '-',
	z: int = 14) -> Tuple[int]:
    lat_deg, lng_deg = list(map(int, coord_str.split(delimiter)))
    return (lat_deg, lng_deg, z)

def coord2country(coord_str: str,
                 delimiter='-',
                 z: int = 14) -> str:
    tile_xyz = coord2xyz(coord_str, delimiter, z)
    return getCountryFromTile(*tile_xyz)