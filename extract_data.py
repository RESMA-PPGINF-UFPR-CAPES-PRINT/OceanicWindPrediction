#!pip install copernicusmarine
import copernicusmarine as cm

#1st Dataset
cm.subset(
  dataset_id="cmems_obs-wind_glo_phy_my_l3-metopa-ascat-asc-0.125deg_P1D-i",
  #force_dataset_version="202311",
  variables=["eastward_wind", "northward_wind"],
  minimum_longitude= -48.3541,
  maximum_longitude= -47.3541,
  minimum_latitude= -26.589,
  maximum_latitude= -25.589,
  start_datetime="2016-01-01T00:00:00",
  end_datetime="2021-11-15T00:00:00",
)

#2nd Dataset
cm.subset(
  dataset_id="cmems_obs-wind_glo_phy_my_l3-metopa-ascat-des-0.125deg_P1D-i",
  #force_dataset_version="202311",
  variables=["eastward_wind", "northward_wind"],
  minimum_longitude= -48.3541,
  maximum_longitude= -47.3541,
  minimum_latitude= -26.589,
  maximum_latitude= -25.589,
  start_datetime="2016-01-01T00:00:00",
  end_datetime="2021-11-15T00:00:00",
)