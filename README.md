# Face Recognition with Face Classification and Verification

## Face Detection with HAAR Features

<table>
  <tr>
    <td><strong>Original Image</strong></td>
    <td><strong>Detected Face in the Image</strong></td>
    <td><strong>Cropped Face</strong></td>
    <td><strong>Fandmarked Face</strong></td>

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
  <li><strong>Network:</strong>
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

### Validation Dataset

<ul>
  <li><strong>166800 comparison between two images</strong>
  <ul>
      <li>The result of matching two images is in the <em>verification_dev.csv</em></li>
    </ul>
  </li>
</ul>
