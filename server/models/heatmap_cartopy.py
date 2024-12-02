import matplotlib.pyplot as plt
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/AlexandriaSea/BikeRecoveryDataset/main/Bicycle_Thefts_Open_Data.csv')

# Extract longitude and latitude
longitude = data['LONG_WGS84']
latitude = data['LAT_WGS84']

# Define the specific latitude and longitude range for zooming
lon_min, lon_max = -79.62222578, -79.12204396
lat_min, lat_max = 43.83723693, 43.58737945

# Add a margin around the bounding box to ensure the map isn't too tightly cropped
margin = 0.02  # 2% margin
lat_margin = (lat_max - lat_min) * margin
lon_margin = (lon_max - lon_min) * margin

# Define the new map extent with margin
extent = [lon_min - lon_margin, lon_max + lon_margin, lat_min - lat_margin, lat_max + lat_margin]

# Calculate the center of the region for better map centering
central_lon = np.mean(extent[:2])
central_lat = np.mean(extent[2:])

plt.figure(figsize=(12, 8))

# Create map with Albers Equal Area projection
ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
ax.set_extent(extent)

# Data resolution for map features
resol = '50m'

# Define map features
country_bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
    name='admin_0_boundary_lines_land', scale=resol, facecolor='none', edgecolor='k')

provinc_bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
    name='admin_1_states_provinces_lines', scale=resol, facecolor='none', edgecolor='k')

land = cartopy.feature.NaturalEarthFeature('physical', 'land', \
    scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])

ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
    scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])

lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', \
    scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])

rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', \
    scale=resol, edgecolor='b', facecolor='none')

# Add features to map
ax.add_feature(land, facecolor='beige', zorder=4)
ax.add_feature(ocean, linewidth=0.2)
ax.add_feature(lakes, zorder=5)
ax.add_feature(rivers, linewidth=0.5, zorder=6)

ax.add_feature(country_bodr, linestyle='--', linewidth=0.8, edgecolor="k", zorder=10)
ax.add_feature(provinc_bodr, linestyle='--', linewidth=0.6, edgecolor="k", zorder=10)

# --- Begin: plot some raster-thematic layer (already implemented in your original code)
xlims = (-130, -55)
ylims = (40, 70)
resolution = 0.2
y, x = np.mgrid[slice(ylims[0], ylims[1] + resolution, resolution),
               slice(xlims[0], xlims[1] + resolution, resolution)]
z = -x + np.sin(x)**2 + np.cos(y)
im = ax.pcolormesh(x, y, z, cmap='viridis_r', zorder=7, alpha=0.2, transform=ccrs.PlateCarree())
plt.colorbar(im, ax=ax, shrink=0.5)

# --- End: thematic layer plotting.

# Heatmap: Plot scatter (or hexbin) based on your data's longitude and latitude
# Scatter plot for heatmap-like effect
ax.scatter(longitude, latitude, c='red', s=10, alpha=0.5, transform=ccrs.PlateCarree(), zorder=8)

# Optional: Use hexbin for a more concentrated heatmap
# ax.hexbin(longitude, latitude, gridsize=100, cmap='YlOrRd', mincnt=1, transform=ccrs.PlateCarree(), zorder=8)

# Optional: Add gridlines
ax.gridlines(draw_labels=True, lw=1.2, edgecolor="darkblue", zorder=12)

# Save plot
plt.title('Bicycle Thefts in Toronto')
plt.savefig('plot/heatmap.png')

# Show plot
plt.show()