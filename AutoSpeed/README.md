# AutoSpeed

Maintaining the vehicle speed and keep safe distance from the vehicle in front is crucial for safe driving. In order to
determine the closest inpath object AutoSpeed network is used. This network is inspired by YOLOv11 architecture, with
substituted c3K2 by a new block ASC block to improve CIPO object detection.

## Performance Results

Network is trained on [OpenLane](https://github.com/OpenDriveLab/OpenLane) dataset.

Network detection success rate is measured by the percentage of frames where the max IoU between the predicted and
ground-truth (GT) bounding boxes is above a certain threshold. For different groups of sequences are selected to measure
network detection success rate for different weather conditions

* Daytime Fair Weather
* Daytime Adverse Weather
* Nighttime Fair Weather
* Nighttime Adverse Weather

<table style="border:none; border-collapse:collapse;">
  <tr>
    <td><img src="../Media/daytime_fair_weather_1.jpg" width="300" alt="alt"></td>
    <td><img src="../Media/daytime_fair_weather_2.jpg" width="300" alt="alt"></td>
    <td><img src="../Media/daytime_fair_weather_3.jpg" width="300" alt="alt"></td>
  </tr>
</table>

<table>
  <tr>
    <th style="text-align:left;">Daytime, Fair Weather</th>
    <th colspan="3" style="text-align:center;">% frames with CIPO detection where prediction/ground truth maxIoU > threshold</th>
  </tr>
  <tr>
    <th style="text-align:left;">Sequence</th>
    <th >maxIoU &gt; 0.50</th>
    <th>maxIoU &gt; 0.75</th>
    <th>maxIoU &gt; 0.90</th>
  </tr>
  <tr>
    <td style="font-size:10px;">segment-17065833287841703_2980_000_3000_000_with_camera_labels</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">98.99%</td>
  </tr>
  <tr>
    <td style="font-size:10px;">segment-346889320598157350_798_187_818_187_with_camera_labels</td>
    <td style="text-align:center;">95.98%</td>
    <td style="text-align:center;">88.44%</td>
    <td style="text-align:center;">70.35%</td>
  </tr>
 <tr>
    <td style="font-size:10px;">segment-902001779062034993_2880_000_2900_000_with_camera_labels</td>
    <td style="text-align:center;">93.94%</td>
    <td style="text-align:center;">92.42%</td>
    <td style="text-align:center;">89.90%</td>
  </tr>
</table>

<table style="border:none; border-collapse:collapse;">
  <tr>
    <td><img src="../Media/daytime_adverse_weather_1.jpg" width="300" alt="alt"></td>
    <td><img src="../Media/daytime_adverse_weather_2.jpg" width="300" alt="alt"></td>
    <td><img src="../Media/daytime_adverse_weather_3.jpg" width="300" alt="alt"></td>
  </tr>
</table>

<table>
  <tr>
    <th style="text-align:left;">Daytime, Adverse Weather</th>
    <th colspan="3" style="text-align:center;">% frames with CIPO detection where prediction/ground truth maxIoU > threshold</th>
  </tr>
  <tr>
    <th style="text-align:left;">Sequence</th>
    <th >maxIoU &gt; 0.50</th>
    <th>maxIoU &gt; 0.75</th>
    <th>maxIoU &gt; 0.90</th>
  </tr>
  <tr>
    <td style="font-size:10px;">segment-191862526745161106_1400_000_1420_000_with_camera_labels</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">100.00%</td>
  </tr>
  <tr>
    <td style="font-size:10px;">segment-447576862407975570_4360_000_4380_000_with_camera_labels</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">100.00%</td>
  </tr>
 <tr>
    <td style="font-size:10px;">segment-6183008573786657189_5414_000_5434_000_with_camera_labels</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">95.48%</td>
  </tr>
</table>


<table style="border:none; border-collapse:collapse;">
  <tr>
    <td><img src="../Media/nighttime_fair_weather_1.jpg" width="300" alt="alt"></td>
    <td><img src="../Media/nighttime_fair_weather_2.jpg" width="300" alt="alt"></td>
    <td><img src="../Media/nighttime_fair_weather_3.jpg" width="300" alt="alt"></td>
  </tr>
</table>

<table>
  <tr>
    <th style="text-align:left;">Nighttime, Fair Weather</th>
    <th colspan="3" style="text-align:center;">% frames with CIPO detection where prediction/ground truth maxIoU > threshold</th>
  </tr>
  <tr>
    <th style="text-align:left;">Sequence</th>
    <th >maxIoU &gt; 0.50</th>
    <th>maxIoU &gt; 0.75</th>
    <th>maxIoU &gt; 0.90</th>
  </tr>
  <tr>
    <td style="font-size:10px;">segment-4426410228514970291_1620_000_1640_000_with_camera_labels</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">100.00%</td>
  </tr>
  <tr>
    <td style="font-size:10px;">segment-5289247502039512990_2640_000_2660_000_with_camera_labels</td>
    <td style="text-align:center;">92.93%</td>
    <td style="text-align:center;">92.93%</td>
    <td style="text-align:center;">91.92%</td>
  </tr>
 <tr>
    <td style="font-size:10px;">segment-8679184381783013073_7740_000_7760_000_with_camera_labels</td>
    <td style="text-align:center;">87.37%</td>
    <td style="text-align:center;">65.15%</td>
    <td style="text-align:center;">0.00%</td>
  </tr>
</table>

<table style="border:none; border-collapse:collapse;">
  <tr>
    <td><img src="../Media/nighttime_adverse_weather_1.jpg" width="300" alt="alt"></td>
    <td><img src="../Media/nighttime_adverse_weather_2.jpg" width="300" alt="alt"></td>
    <td><img src="../Media/nighttime_adverse_weather_3.jpg" width="300" alt="alt"></td>
  </tr>
</table>

<table>
  <tr>
    <th style="text-align:left;">Nighttime, Adverse Weather</th>
    <th colspan="3" style="text-align:center;">% frames with CIPO detection where prediction/ground truth maxIoU > threshold</th>
  </tr>
  <tr>
    <th style="text-align:left;">Sequence</th>
    <th >maxIoU &gt; 0.50</th>
    <th>maxIoU &gt; 0.75</th>
    <th>maxIoU &gt; 0.90</th>
  </tr>
  <tr>
    <td style="font-size:10px;">segment-6491418762940479413_6520_000_6540_000_with_camera_labels</td>
    <td style="text-align:center;">55.28%</td>
    <td style="text-align:center;">42.71%</td>
    <td style="text-align:center;">22.61%</td>
  </tr>
  <tr>
    <td style="font-size:10px;">segment-11356601648124485814_409_000_429_000_with_camera_labels</td>
    <td style="text-align:center;">100.00%</td>
    <td style="text-align:center;">90.95%</td>
    <td style="text-align:center;">43.72%</td>
  </tr>
 <tr>
    <td style="font-size:10px;">segment-11901761444769610243_556_000_576_000_with_camera_labels</td>
    <td style="text-align:center;">99.49%</td>
    <td style="text-align:center;">99.49%</td>
    <td style="text-align:center;">97.96%</td>
  </tr>
</table>

***Note:*** In some sequences detection success rate is 100%, but the reason is that these sequences do not have CIPO
object, and result in some sequences are low because some of the far away objects are not detected.