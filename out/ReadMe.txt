Snow End Day (SED) and Snow Onset Day (SOD) from Sentinel-2 (S2) 2017-2022 climatology

Contact: bav@geus.dk
Resolution: 20 m
CRS: EPSG 4326

Cloud masking: S2cloudless (Zupanc et al., 2017; Skakun et al. 2022) 
Snow detection: Adapted Let-It-Snow algorithm from (Gascoin et al., 2019) based on NDSI and B04
Data query and download: SentinelHub
<<<<<<<<<<
Gap-free snow map, climatology and snow end/start day construction: 
The cloud masks scenes from 2017 to 2022 are downloaded and binary snow cover (BSC) is derived. Since in the following, climatological snow cover will be derived, we convert the BSC maps into Snow Cover Probability (SCP) maps (0 or 100% SCP at this point). The temporally sparse SCP data is then interpolated to daily resolution. Each winter, the days between one year's last S2 observation and the following year's first observation is filled with SCP=100. Then for each of the remaining gap, we run separately a forward filling (SCP_ff) and a backward filling (SCP_bf). If SCP_ff == SCP_bf, meaning that the SCP values on each side of the gap agree, then we fill the gap with that value. The remaining gaps, over which SCP are seen to change, will be gap-filled from the climatological SCP values. For each day of the year, we make the average of all available SCP observations. The result is a a  6 year SCP climatology SCP_clim. Eventually, the gaps that was left in the SCP dataset can be filled with the correponding value for that day of the year in SCP_clim.

From SCP_clim, we then  define the Snow End Day (SED) and Snow Onset Day (SOD) as the first and last day of the year when the SCP_clim is below 65%.

Limitations:
- S2 has 5 days revisit
- No observation under clouds and in polar night
- Consequently, uneasy to detect snow-free pixels in the winter
- The early/late "spikes" in snow melt day and snow onset day is currently note handle. We only look at first and last snow-free day in the climatology.

References:
Gascoin, S., Grizonnet, M., Bouchet, M., Salgues, G., and Hagolle, O.: Theia Snow collection: high-resolution operational snow cover maps from Sentinel-2 and Landsat-8 data, Earth Syst. Sci. Data, 11, 493–514, https://doi.org/10.5194/essd-11-493-2019, 2019.

Skakun, S., Wevers, J., Brockmann, C., Doxani, G., Aleksandrov, M., Batič, M., Frantz, D., Gascon, F., Gómez-Chova, L., Hagolle, O. and López-Puigdollers, D., 2022. Cloud Mask Intercomparison eXercise (CMIX): An evaluation of cloud masking algorithms for Landsat 8 and Sentinel-2. Remote Sensing of Environment, 274, p.112990.

Zupanc, A., 2017. Improving Cloud Detection with Machine Learning. https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13 (accessed 09 June 2021).