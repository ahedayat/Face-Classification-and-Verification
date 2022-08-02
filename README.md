# Face Recognition with Face Classification and Verification

# Face Detection with HAAR Features

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

# Face Classification

<ul>
  <li>Network:
    <ul>
      <li>Gray + ResNet-18</li>
      <li>RGB + ResNet-18</li>
      <li>RGB + ResNet-50</li>
    </ul>
  </li>
  <li>Loss Function:
    <ul>
      <li>ArcFace</li>
    </ul>
  </li>
</ul>

<ul>
  <li>Hyper-Parameters:
  </li>
</ul>

| **Hyper-Parameters** | **Value** |
| :------------------: | :-------: |
|          s           |    64     |
|          m           |    0.5    |
|        Epoch         |    100    |
|  Momentum (for SGD)  |    0.9    |
|     Weight Decay     |   5e-4    |

<ul>
  <li>Learning Rate:
  </li>
</ul>

| **Epoch** | **Value** |
| :-------: | :-------: |
|  [1,20]   |   1e-1    |
|  [21,30]  |   1e-2    |
|  [31,40]  |   1e-3    |
|  [41,60]  |   1e-4    |
|  [61,90]  |   1e-5    |
| [91,100]  |   1e-6    |

# Face Verification
