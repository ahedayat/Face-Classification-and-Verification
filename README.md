# Face Recognition with Face Classification and Verification

## Face Detection with HAAR Features

<table style="margin-left: auto; margin-right: auto;">
  <tr>
    <td><strong>Original Image</strong></td>
    <td><strong>Detected Face in the Image</strong></td>
    <td><strong>Cropped Face</strong></td>
    <td><strong>Landmarked Face</strong></td>

   </tr> 
   <tr>
    <td> <img src="./plots/face_detection/me2.jpeg"  alt="original_image" width = 200px height = 300px ></td>
    <td> <img src="./plots/face_detection/me2_bbox.jpeg"  alt="Original image with a boundig box arroung detected face" width = 200px height = 300px ></td>
    <td> <img src="./plots/face_detection/me2_croped.jpeg"  alt="cropped face" width = 200px height = 200px ></td>
    <td> <img src="./plots/face_detection/me2_landmark.jpeg"  alt="landmarked face" width = 200px height = 300px ></td>

  </td>
  </tr>
</table>

## Face Classification

<ul>
  <li><strong>Input + Backbone Network:</strong>
    <ul>
      <li>Gray + ResNet-18</li>
      <li>RGB + ResNet-18</li>
      <li>RGB + ResNet-50</li>
    </ul>
  </li>
  <li><strong>Loss Function:</strong>
    <ul>
      <li>ArcFace</li>
    </ul>
  </li>
</ul>

<ul>
  <li><strong>Hyper-Parameters:</strong>
  </li>
</ul>
<center>

| **Hyper-Parameters** | **Value** |
| :------------------: | :-------: |
|          s           |    64     |
|          m           |    0.5    |
|        Epoch         |    100    |
|  Momentum (for SGD)  |    0.9    |
|     Weight Decay     |   5e-4    |

</center>

<ul>
  <li><strong>Learning Rate:</strong>
  </li>
</ul>

<center>

| **Epoch** | **Value** |
| :-------: | :-------: |
|  [1,20]   |   1e-1    |
|  [21,30]  |   1e-2    |
|  [31,40]  |   1e-3    |
|  [41,60]  |   1e-4    |
|  [61,90]  |   1e-5    |
| [91,100]  |   1e-6    |

</center>

## Face Verification

<ul>
  <li><strong>Similarity Metrics:</strong>
    <ul>
      <li>Cosine Similarity</li>
    </ul>
  </li>
</ul>

<center>

$( Cosine-Similarity (x_1, x_2)=\frac{ \vec{x_1} \cdot \vec{x_2} }{\| x_1 \| \times \| x_2 \|} )$

</center>

<ul>
  <li><strong>Face Verification Flow Chart:</strong>
  </li>
</ul>

<center>
<img src="./plots/face_verification.png"  alt="Face Verification Flow Chart" width = 600px height = 300px >

</center>

## Dataset

### Classification Dataset

<center>

|            | **Num. of Categories** | **Num. of Data per Category** |
| :--------: | :--------------------: | :---------------------------: |
|  Training  |          1000          |              16               |
| Validation |          1000          |               4               |
|    Test    |          1000          |               5               |

</center>

### Verification Dataset

<ul>
  <li><strong>166800 comparison between two images</strong>
  <ul>
      <li>The result of matching two images is in the <em>verification_dev.csv</em></li>
    </ul>
    <li>Distribution of Labels in Verification Dataset:</li>
  </li>
</ul>

<table  style="margin-left: auto; margin-right: auto;">
  <tr>
    <td><center> <strong>Number</strong> </center></td>
    <td><center> <strong>Percentage</strong> </center></td>

   </tr> 
   <tr>
    <td> <img src="./plots/EDA/verification_match_result_eng.png"  alt="Number of correctness or incorrectness of matching of two images in validation dataset." width = 300px height = 300px ></td>
    <td> <img src="./plots/EDA/verification_match_result_percent_eng.png"  alt="Ratio of correctness or incorrectness of matching of two images in validation dataset." width = 300px height = 300px ></td>

  </td>
  </tr>
</table>

## Results

### Classification Results

<ul>
  <li><strong>Classification Loss Function for Different Networks</strong>
  </li>
</ul>

<table  style="margin-left: auto; margin-right: auto;">
  <tr>
    <td><center> <strong>Gray + ResNet-18</strong> </center></td>
    <td><center> <strong>RGB + ResNet-18</strong> </center></td>
    <td><center> <strong>RGB + ResNet-50</strong> </center></td>

   </tr> 
   <tr>
    <td> <img src="./plots/results_classification/gray_train_val_losses.png"  alt='loss function plot for "Gray + ResNNet-18"' width = 300px height = 200px ></td>
    <td> <img src="./plots/results_classification/18_train_val_losses.png"  alt='loss function plot for "RGB + ResNNet-18"' width = 300px height = 200px ></td>
    <td> <img src="./plots/results_classification/50_train_val_losses.png"  alt='loss function plot for "RGB + ResNNet-50"' width = 300px height = 200px ></td>

  </td>
  </tr>
</table>

### Verificaiton Results

<ul>
  <li><strong>Verification ROC Curve</strong>
  </li>
</ul>

<table style="margin-left: auto; margin-right: auto;">
   <tr>
    <td> <img src="./plots/results_verification/rocs.png"  alt='loss function plot for "Gray + ResNNet-18"' width = 600px height = 500px ></td>
  </tr>
</table>

<ul>
  <li><strong>Other Metrics</strong>
  </li>
</ul>

<table  style="margin-left: auto; margin-right: auto;">
    <thead>
        <tr>
            <th>Network</th>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Threshold</th>
            <th>Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>Gray + ResNet-18</td>
            <td><center> 0 </center></td>
            <td><center> 0.62 </center></td>
            <td><center> 0.79 </center></td>
            <td><center> 0.70 </center></td>
            <td rowspan=2><center> 0.158 </center></td>
            <td rowspan=2><center> 66.9% </center></td>
        </tr>
        <tr>
            <td><center> 1 </center></td>
            <td><center> 0.75 </center></td>
            <td><center> 0.56 </center></td>
            <td><center> 0.64 </center></td>
        </tr>
        <tr>
            <td rowspan=2>RGB + ResNet-18</td>
            <td><center> 0 </center></td>
            <td><center> 0.65 </center></td>
            <td><center> 0.83 </center></td>
            <td><center> 0.73 </center></td>
            <td rowspan=2><center> 0.162 </center></td>
            <td rowspan=2><center> 70.3% </center></td>
        </tr>
        <tr>
            <td><center> 1 </center></td>
            <td><center> 0.79 </center></td>
            <td><center> 0.59 </center></td>
            <td><center> 0.68 </center></td>
        </tr>
        <tr>
            <td rowspan=2>RGB + ResNet-50</td>
            <td><center> 0 </center></td>
            <td><center> 0.64 </center></td>
            <td><center> 0.85 </center></td>
            <td><center> 0.73 </center></td>
            <td rowspan=2><center> 0.124 </center></td>
            <td rowspan=2><center> 70.2% </center></td>
        </tr>
        <tr>
            <td><center> 1 </center></td>
            <td><center> 0.81 </center></td>
            <td><center> 0.56 </center></td>
            <td><center> 0.66 </center></td>
        </tr>
    </tbody>
</table>
