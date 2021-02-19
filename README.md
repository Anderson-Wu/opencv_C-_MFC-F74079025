# opencv_MFC
## a. About
Learning image processing by using opencv 
## b. UI 
Using MFC to build UI

![UI](./img/UI.png)

## c. function
### Image Processing
#### Load Image
Open a new window to show the image (dog.bmp)
Show the height and width of the image in console mode
![1.1](./img/1.1.png)

#### Color Conversior
Exchange 3 channels of the image BGR to RBG
<table style="width:100%">
    <tr>
        <td><img src="./img/1.2_before.png"></td>
        <td><img src="./img/1.2_after.png"></td>
    </tr>
</table>

#### Image Flipping
Flip the image (dog.bmp) and open a new window to show the result.
<table style="width:100%">
    <tr>
        <td><img src="./img/1.3_before.bmp"></td>
        <td><img src="./img/1.3_after.png"></td>
    </tr>
</table>

#### Image Blemnding
Combine two images (dog.bmp and the its flip image) and Use Trackbar to change the weights and show the result in the new window.
![Blending](./img/1.4_after.png)

### Adaptive threshold
#### Global Threshold
Show the result after applying global threshold.(Threshold value is 80)
![Global](./img/2.1.png)
#### local Threshold
Show the result after applying local threshold.(blockSize = 19, offset = -1)
![local](./img/2.2.png)

### Image transformation
#### Transforms: Rotation, Scaling, Translation
rotate, scale and translate the small squared image with parameters entering in GUI
![Rotation, Scaling, Translation](./img/3.1.png)
#### Perspective transform
Click 4 points showed in console window. (start from top-left corner of the original image, and then click clock-wise)
Warp the original image to the location (20,20), (20,450), (450,450), (450,20). Open second window to show the result.
<table style="width:100%">
    <tr>
        <td><img src="./img/3.2_before.png" style="width:70%;"></td>
        <td><img src="./img/3.2_after.png"></td>
    </tr>
</table>

### Convolution
#### Gaussian
Convert the RGB image to grayscale image and then smooth the grayscale	image by using 3x3 Gaussian smoothing filter and show the result.
<table style="width:100%">
    <tr>
        <td><img src="./img/4.1_before.jpg"></td>
        <td><img src="./img/4.1_after.png"></td>
    </tr>
</table>

#### Sobel x
Sobel edge detection to detect vertical edge 
<table style="width:100%">
    <tr>
        <td><img src="./img/4.2_before.png"></td>
        <td><img src="./img/4.2_after.png"></td>
    </tr>
</table>

#### Sobel y
Sobel edge detection to detect horizon edge 
<table style="width:100%">
    <tr>
        <td><img src="./img/4.3_before.png"></td>
        <td><img src="./img/4.3_after.png"></td>
    </tr>
</table>

#### magnitude
Use  result(sobel x and sobel y) to calculate the magnitude and show.
<table style="width:100%">
    <tr>
        <td><img src="./img/4.4_after.png"></td>
    </tr>
</table>