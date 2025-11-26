import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from astropy.time import Time
from astropy.coordinates import get_sun, get_body, EarthLocation, AltAz
import astropy.units as u
from astropy.utils import iers

iers.conf.auto_download = False

# --- Time & grid params ---
start_time = Time('2026-01-01 00:00:00')
end_time   = Time('2026-12-31 23:59:59')
time_step  = 6 * u.hour

latitudes  = np.linspace(-90, 90, 90)
longitudes = np.linspace(-180, 180, 180)
LAT, LON = np.meshgrid(latitudes, longitudes, indexing='ij')
eclipse_mask = np.zeros(LAT.shape)

# --- Cities to check (name, lat, lon) ---
cities = [
    ("Chicago, IL, USA", 41.8781, -87.6298),
    ("New York, NY, USA", 40.7128, -74.0060),
    ("Los Angeles, CA, USA", 34.0522, -118.2437),
    ("London, UK", 51.5074, -0.1278),
    ("Istanbul, Turkey", 41.0082, 28.9784),
    # add more as you want
]
city_eclipses = {name: [] for name, _, _ in cities}

times = start_time + np.arange(0, (end_time - start_time).sec, time_step.to(u.s).value)*u.s

print("Starting full‑year 2026 approximate eclipse calculation + city checks...")

for t in times:
    # Global grid (as before)
    locations = EarthLocation(lat=LAT.ravel()*u.deg, lon=LON.ravel()*u.deg)
    altaz_frame = AltAz(obstime=t, location=locations)
    sun_altaz  = get_sun(t).transform_to(altaz_frame)
    moon_altaz = get_body('moon', t).transform_to(altaz_frame)
    separation = sun_altaz.separation(moon_altaz).deg
    sun_alt = sun_altaz.alt.deg
    eclipse_indices = np.where((separation < 0.5) & (sun_alt > 0))[0]
    eclipse_mask.ravel()[eclipse_indices] = 1

    # Check each city
    for name, lat, lon in cities:
        loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg)
        frame = AltAz(obstime=t, location=loc)
        sun_a = get_sun(t).transform_to(frame)
        moon_a = get_body('moon', t).transform_to(frame)
        sep = sun_a.separation(moon_a).deg
        if (sep < 0.5) and (sun_a.alt.deg > 0):
            city_eclipses[name].append(t.iso)  # record ISO date/time

print("Calculation done.\n")

# Print summary for cities
for name in city_eclipses:
    times_list = city_eclipses[name]
    if times_list:
        print(f"> {name}: Eclipse(s) possible at approx. UTC times:")
        for ti in times_list:
            print("   ", ti)
    else:
        print(f"> {name}: No eclipse predicted (approximate) in 2026")

# --- Plotting global map + city markers ---
plt.figure(figsize=(16,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.gridlines(draw_labels=True, linewidth=0.5)
plt.contourf(LON, LAT, eclipse_mask, levels=[0,0.5,1], colors=['white','orange'], alpha=0.5)

# Plot cities
for name, lat, lon in cities:
    ax.plot(lon, lat, marker='o', color='red', markersize=5, transform=ccrs.PlateCarree())
    ax.text(lon + 2, lat + 2, name, transform=ccrs.PlateCarree(), fontsize=8)

plt.title("Approximate 2026 Solar Eclipse Regions + Example Cities")
plt.show()
