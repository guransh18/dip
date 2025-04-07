
### **Practical 01 **  
```matlab
% Define the filename
filename = "C:\TANISH PERSONAL\PHOTU\919_1.jpg";

% Get image info and display it
imageinfo = imfinfo(filename);
disp(imageinfo);

% Read the image and display its dimensions
I = imread(filename);
dimensions = size(I);
disp(dimensions);

% Convert to grayscale and display
grayImage = rgb2gray(I);
figure;
imshow(grayImage);
title('Grayscale Image');

% Perform multiple image operations and visualization
figure;
subplot(3, 3, 1); 
imshow(grayImage);
title('Grayscale Image');

subplot(3, 3, 2);
imshow(I);
title('Original Image (Color)');

% Apply edge detection
edges = edge(grayImage, 'canny');
subplot(3, 3, 3);
imshow(edges);
title('Edge Detection');

% Display variable info
whos grayImage; 
whos I;

% Plot multiple overlapping lines
figure;
hold on;
plot([9 10 11 12], 'DisplayName', 'Line 1');
plot([8 9 10 11], 'DisplayName', 'Line 2');
plot([7 8 9 10], 'DisplayName', 'Line 3');
plot([6 7 8 9], 'DisplayName', 'Line 4');
plot([5 6 7 8], 'DisplayName', 'Line 5');
plot([4 5 6 7], 'DisplayName', 'Line 6');
plot([3 4 5 6], 'DisplayName', 'Line 7');
plot([2 3 4 5], 'DisplayName', 'Line 8');
plot([1 2 3 4], 'DisplayName', 'Line 9');
hold off;
legend("Location", "northeastoutside");
title('Line Plots');

% Convert image to binary and display
BW = imbinarize(grayImage);
figure;
imshow(BW);
title('Binary Image');
```

### **Practical 02**

```matlab
clc; clear; close all;

% ========== 1. Slice Viewer ==========
I = imread("C:\Users\admin\Downloads\HD-wallpaper-anime-vagabond-miyamoto-musashi.jpg");
cmap = parula(200);
figure;
sliceViewer(I, "Colormap", cmap);

title('Slice Viewer');

% ========== 2. Imcrop ==========
figure;
imshow(I);
J = imcrop(I);
figure;
imshow(J);
title('Cropped Image');

% ========== 3. Resize and Rotate ==========
croppedImage = imcrop(I);
resizedImage = imresize(croppedImage, [300 400]);
rotatedImage = imrotate(resizedImage, 45);
figure;
imshow(rotatedImage);
title('Resized and Rotated Image');

% ========== 4. Imrescale ==========
I_rescaled = imresize(I, 0.2);
figure;
imshow(I_rescaled);
title('Rescaled Image (20%)');

% ========== 5. 2D View ==========
viewer2D = viewer2d;
imageshow(I, 'Parent', viewer2D);
viewer2D.ScaleBar = "off";
title('2D View');

% ========== 6. 3D View ==========
viewer3D = viewer3d(BackgroundColor="white", GradientColor=[0.5 0.5 0.5], Lighting="on");
load(fullfile(toolboxdir("images"), "imdata", "BrainMRILabeled", "images", "vol_001.mat"));
mriVol = volshow(vol, 'Parent', viewer3D);
viewer3D.CameraPosition = [120 120 200];
viewer3D.CameraTarget = [120 120 -10];
viewer3D.CameraUpVector = [0 1 0];
viewer3D.CameraZoom = 1.5;
title('3D View');

```

---

### **Practical 03**

```matlab
% ========== contrast ==========

I = imread("C:\Users\admin\Downloads\HD-wallpaper-anime-vagabond-miyamoto-musashi.jpg");
I_gray = rgb2gray(I);  
h1 = figure;
imshow(I_gray);
imcontrast;  
 
% ========== sharpen ==========
 
 a=imread("C:\Users\admin\Downloads\HD-wallpaper-anime-vagabond-miyamoto-musashi.jpg");
figure,imshow(a);
title('OG Img');
b = imsharpen(a);
figure, imshow(b)
title('Sharpened Image');

% ========== fuse ==========

A = imread("C:\TANISH PERSONAL\PHOTU\10666 (1).jpg"); % First image
B = imread("C:\TANISH PERSONAL\PHOTU\screen-6.jpg");  % Second image

% Resize images to the same size (if needed)
B = imresize(B, [size(A,1), size(A,2)]); 

% Fuse both images using 'blend' or 'falsecolor'
C = imfuse(A, B, 'blend', 'Scaling', 'joint'); % You can also try 'falsecolor'
imshow(C);

% ========== squeeze ==========

A = rand(1, 5, 1); % A 1×5×1 array
B = squeeze(A); % B becomes a 5×1 array
size(B) % Output: [5,1]

% ========== montage ==========

% Read images into a cell array
img1 = imread("C:\Users\admin\Downloads\sample1.bmp");  % First image
img2 = imread("C:\Users\admin\Downloads\yoga.jpg");  % Second image

% Combine images into a cell array
images = {img1, img2};

% Display images as a montage
montage(images);
title('Image Montage');

% ========== histogram equilization ==========

I = imread("C:\Users\admin\Downloads\ironman.jpg");
% Conversion of RGB to YIQ format
b = rgb2ntsc(I);
% Histogram equalization of Y component alone
b(:, :, 1) = histeq(b(:, :, 1));
% Conversion of YIQ to RGB format
c = ntsc2rgb(b);
% Display original and histogram equalized images
imshow(I), title("Original Image");
figure, imshow(c), title("Histogram Equalized Image");

% ========== histogram  ==========

I = imread("C:\Users\admin\Downloads\ironman.jpg");
imshow(I), figure;
I = im2double(I);

% Convert the image to HSV
hsv = rgb2hsv(I);
h = hsv(:, :, 1);  % Hue channel
s = hsv(:, :, 2);  % Saturation channel
v = hsv(:, :, 3);  % Value channel

% Total number of pixels
pixels = numel(h);

% Find locations of black and white pixels
darks = find(v < 0.2);
lights = find(s < 0.05 & v > 0.85);
h([darks; lights]) = -1;

% Get the number of pixels for each color bin
black = length(darks) / pixels;
white = length(lights) / pixels;
red = length(find((h > 0.9167 | h <= 0.083) & h ~= -1)) / pixels;
yellow = length(find(h > 0.083 & h <= 0.25)) / pixels;
green = length(find(h > 0.25 & h <= 0.4167)) / pixels;
cyan = length(find(h > 0.4167 & h <= 0.5833)) / pixels;
blue = length(find(h > 0.5833 & h <= 0.75)) / pixels;
magenta = length(find(h > 0.75 & h <= 0.9167)) / pixels;

% Plot histogram of color distribution
hold on;
fill([0 0 1 1], [0 red red 0], 'r');
fill([1 1 2 2], [0 yellow yellow 0], 'y');
fill([2 2 3 3], [0 green green 0], 'g');
fill([3 3 4 4], [0 cyan cyan 0], 'c');
fill([4 4 5 5], [0 blue blue 0], 'b');
fill([5 5 6 6], [0 magenta magenta 0], 'm');
fill([6 6 7 7], [0 white white 0], 'w');
fill([7 7 8 8], [0 black black 0], 'k');
axis([0 8 0 1]);

Trick to Remember
HSV Hue is basically a color wheel mapped from 0 to 1 (instead of 0°–360°).
Each primary and secondary color spans roughly 60° (or 1/6 = 0.1667 in the 0-1 range)
Key Positions:
Red: Starts at 0, wraps at 1
Yellow: 1/6 (≈0.1667)
Green: 2/6 (≈0.3333)
Cyan: 3/6 (≈0.5)
Blue: 4/6 (≈0.6667)
Magenta: 5/6 (≈0.8333)
Red again: 1 (or 0)
```

### **Practical 04 **
```matlab
% ========== Basic convolution ==========

x = [12:34];
y = [56;78];
z = conv2(x,y,"same");
disp("basic convolution");
disp(z);

% ========== Circular convolution ==========

x = [1,1,1,1];
y = [1,2,3,4];
X = fft(x);
Y = fft(y);
z = ifft(X.*Y);
disp("Circular")
disp(z);

% ========== Circular correlation between two signals ==========


x = [5 10; 15 20];
y = [3 6; 9 12];

h1 = fliplr(y);
h2 = flipud(h1);

x1 = fft2(x);
x2 = fft2(h2);

y1 = x1 .* x2;
y2 = ifft2(y1);

disp("Circular Correlation");
disp(real(y2));
```

### **Practical 05**
```matlab
% ========== 2D DFT ==========

input_image = [1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 1];
kernel = dftmtx(4);
output = kernel * input_image * kernel';
disp('Input Image:');
disp(input_image);
disp('DFT Kernel:');
disp(kernel);
disp('Output after DFT transformation:');
disp(output);

% ========== Demonstrate rotation property of DFT ==========

%code to generate original image
a=zeros(256);
[m n]=size(a);
for i = 120:145
   for j=120:145
   	a(i,j)=255;
   end
end

%original image rotated by 45 degree
b=imrotate(a,45,'bilinear','crop');

%spectrum of original image
a1=log(1+abs(fftshift(fft2(a))));

%spectrum of rotated image
b1=log(1+abs(fftshift(fft2(b))));

subplot(2,2,1),imshow(a),title('Original image');
subplot(2,2,2),imshow(b),title('Image rotated by 45 degree');
subplot(2,2,3),imshow(mat2gray(a1)),title('Original Image Spectrum');
subplot(2,2,4),imshow(mat2gray(b1)),title('spectrum of rotated image');

% ========== Interchange phase between 2 images  ==========

a = imread('cameraman.tif');
b = imread('C:\Users\admin\Downloads\images.png');

% Resize image b to the size of image a
b = imresize(b, size(a));

% Perform 2D FFT on both images
ffta = fft2(double(a));
fftb = fft2(double(b));

% Calculate magnitude and phase for both images
mag_a = abs(ffta);
ph_a = angle(ffta);
mag_b = abs(fftb);
ph_b = angle(fftb);

% Combine magnitudes and phases for each image
newfft_a = mag_a .* exp(1i * ph_b);
newfft_b = mag_b .* exp(1i * ph_a);

% Perform inverse FFT to reconstruct images
rec_a = ifft2(newfft_a);
rec_b = ifft2(newfft_b);

% Display the images
imshow(a), title('Original Image a');
figure, imshow(b), title('Original Image b');
figure, imshow(uint8(real(rec_a))), title('Image a after phase reversal');
figure, imshow(uint8(real(rec_b))), title('Image b after phase reversal');


```

### **Practical 06**
```matlab
 % ========== Walsh Transform  ==========
 
 n = input('Enter the basis matrix dimension:');
m = n;

for u = 0:n-1
    for v = 0:n-1
        for x = 0:n-1
            for y = 0:n-1
                powervalue = 1;
                sn = log2(n);
                for i = 0:sn-1
                    a = dec2bin(x, sn);
                    b = bin2dec(a(sn-i));
                    c = dec2bin(y, sn);
                    d = bin2dec(c(sn-i));
                    e = dec2bin(u, sn);
                    f = bin2dec(e(i+1));
                    e = dec2bin(v, sn);
                    a = bin2dec(e(i+1));
                    powervalue = powervalue * (-1)^(b*f + d*a);
                end
                basis{u+1, v+1}(x+1, y+1) = powervalue;
            end
        end
    end
end

figure(1)
k = 1;
for i = 1:m
    for j = 1:n
        subplot(m, n, k)
        imshow(basis{i, j}, []) 
        k = k + 1;
    end
end

 % ========== Haar Transform  ==========
 

n = input('Enter the basis matrix dimension: ');
m = n;
for u = 0:n-1
    for v = 0:n-1
        for x = 0:n-1
            for y = 0:n-1
                powervalue = 0;
                sn = log2(n);
                for i = 0:sn-1
                    a = dec2bin(x, sn);
                    b = bin2dec(a(sn-i));
                    c = dec2bin(y, sn);
                    d = bin2dec(c(sn-i));
                    e = dec2bin(u, sn);
                    f = bin2dec(e(i+1));
                    e = dec2bin(v, sn);
                    a = bin2dec(e(sn-i));
                    powervalue = powervalue + (b*f + d*a);
                end
                basis{u+1, v+1}(x+1, y+1) = (-1)^powervalue;
            end
        end
    end
end

mag = basis;
figure(4)
k = 1;

% Code to plot Haar basis
for i = 1:m
    for j = 1:n
        subplot(m, n, k)
        % Ensure mag{i, j} is converted to a matrix
        imshow(double(mag{i, j}), [])
        k = k + 1;
    end
end


 % ========== DCT Transform  ==========
 
 m = input('Enter the basis matrix dimension: ');
n = m;
alpha2 = ones(1, n) * sqrt(2/n);
alpha2(1) = sqrt(1/n);
alpha1 = ones(1, m) * sqrt(2/m);
alpha1(1) = sqrt(1/m);

for u = 0:m-1
    for v = 0:n-1
        for x = 0:m-1
            for y = 0:n-1
                a{u+1, v+1}(x+1, y+1) = alpha1(u+1) * alpha2(v+1) * ...
                    cos((2*x+1)*u*pi/(2*m)) * cos((2*y+1)*v*pi/(2*n));
            end
        end
    end
end

mag = a;
figure(3);
k = 1;

for i = 1:m
    for j = 1:n
        subplot(m, n, k);
        imshow(mag{i, j}, []);
        k = k + 1;
    end
end


 % ========== DFT Transform  ========== 
 
 img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
img = rgb2gray(img);
img_double = double(img);
dft_result = fft2(img_double);
dft_shifted = fftshift(dft_result);
magnitude_spectrum = log(abs(dft_shifted) + 1);
subplot(1, 2, 1), imshow(img, []), title('Original Image');
subplot(1, 2, 2), imshow(magnitude_spectrum, []), title('DFT Magnitude Spectrum');

```

### **Practical 07**
```matlab
% ========== Brightness ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
%img = rgb2gray(img); 
bright_img = img + 50;
%bright_img(bright_img > 255) = 255;
figure;
subplot(1, 2, 1), imshow(img, []), title('Original Image');
subplot(1, 2, 2), imshow(bright_img, []), title('Brightened Image (+50)');

% ========== contrast ==========

a = imread("C:\TANISH PERSONAL\PHOTU\02.jpg");
b=rgb2gray(a);
c=b*.5;
d=b*20;
imshow(a),title('Original Image')
figure,imshow(c),title('Increase in contrast');
figure,imshow(d),title('Decrease in contrast');

% ========== Digital Negative ==========

a = imread("C:\Users\admin\Downloads\luffy.jpg");
k = 255 - a;
subplot(2, 1, 1), imshow(a), title('Original Image')
subplot(2, 1, 2), imshow(k), title('Negative of Original Image')

% ========== threshold ==========

img = imread("C:\TANISH PERSONAL\PHOTU\02.jpg");
img = rgb2gray(img);
T = graythresh(img) * 255;
binary_img = img > T;
imshow(binary_img);

% ========== graylevel slicing ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
img = rgb2gray(img); 
low_threshold = 100;
high_threshold = 200;
sliced_img = img;
sliced_img(img >= low_threshold & img <= high_threshold) = 255;
subplot(1, 2, 1), imshow(img),title('Original Grayscale Image');
subplot(1, 2, 2), imshow(sliced_img),title('Gray Level Sliced Image');

% ========== bit plain slicing ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
img = rgb2gray(img); 
for bit = 1:8
    bit_plane = bitget(img, bit) * 255;
    subplot(2, 4, bit),imshow(bit_plane),title('Bit Plane');
end

% ========== log transform ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
img = rgb2gray(img);
img = double(img);
c = 255 / log(1 + max(img(:))); 
log_transformed = c * log(1 + img);
log_transformed = uint8(log_transformed);
figure;
subplot(1, 2, 1), imshow(uint8(img)), title('Original Grayscale Image');
subplot(1, 2, 2), imshow(log_transformed, []), title('Log Transformed Image');

% ========== power law transform ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
if size(img, 3) == 3 
   img = rgb2gray(img);
end
img = double(img) / 255; 
gamma = 0.5; 
c = 1; 
power_transformed = c * (img .^ gamma);
figure;
subplot(1, 2, 1), imshow(img), title('Original Grayscale Image');
subplot(1, 2, 2), imshow(power_transformed), title(['Power-Law Transform (Gamma = ', num2str(gamma), ')']);

```

### **Practical 08**
```matlab

```

### **Practical 09 **
```matlab

% ========== Dilation ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
img = rgb2gray(img); 
threshold = graythresh(img);
binary_img = imbinarize(img, threshold);
se = strel('disk', 5); 
dilated_img = imdilate(binary_img, se);
figure;
subplot(1, 2, 1), imshow(binary_img), title('Original Binary Image');
subplot(1, 2, 2), imshow(dilated_img), title('Dilated Image');

% ========== Erosion ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
img = rgb2gray(img); 
threshold = graythresh(img);
binary_img = imbinarize(img, threshold);
se = strel('disk', 5); % You can change 'disk' to 'square', 'line', etc.
eroded_img = imerode(binary_img, se);
figure;
subplot(1, 2, 1), imshow(binary_img), title('Original Binary Image');
subplot(1, 2, 2), imshow(eroded_img), title('Eroded Image');


% ========== Closing ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
img = rgb2gray(img); 
threshold = graythresh(img);
binary_img = imbinarize(img, threshold);
se = strel('disk', 5); 
closed_img = imclose(binary_img, se);
figure;
subplot(1, 2, 1), imshow(binary_img), title('Original Binary Image');
subplot(1, 2, 2), imshow(closed_img), title('Closed Image');

% ========== opening ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
img = rgb2gray(img); 
threshold = graythresh(img);
binary_img = imbinarize(img, threshold);
se = strel('disk', 5); 
opened_img = imopen(binary_img, se);
figure;
subplot(1, 2, 1), imshow(binary_img), title('Original Binary Image');
subplot(1, 2, 2), imshow(opened_img), title('Opened Image');

% ========== Chain rule ==========

% Define input binary matrix
X = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0];

% Define structuring elements
B1 = [0 1 0; 1 1 1; 0 1 0];
B2 = [1 1 1; 1 1 1; 1 1 1];

% Chain Rule 1: Erosion
Erx = imerode(X, B1);
Erf = imerode(Erx, B2);
disp('Chain Rule 1 LHS:');
disp(Erf);

DiB = imdilate(B1, B2);
Erx1 = imerode(X, DiB);
disp('Chain Rule 1 RHS:');
disp(Erx1);

% Chain Rule 2: Dilation
Dix1 = imdilate(X, B1);
Dix = imdilate(Dix1, B2);
disp('Chain Rule 2 LHS:');
disp(Dix);

DiB = imdilate(B1, B2);
Dix2 = imdilate(X, DiB);
disp('Chain Rule 2 RHS:');
disp(Dix2);

% ========== idempotency==========

% Define input binary matrix
x = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0];

% Define structuring element
B = [0 1 0; 1 1 1; 0 1 0];

% Idempotency of Erosion
Erx = imerode(x, B);
Erf = imerode(Erx, B);
disp('Idempotency1 LHS:');
disp(Erf);

Erx1 = imerode(x, B);
disp('Idempotency1 RHS:');
disp(Erx1);

% Idempotency of Dilation
Dix1 = imdilate(x, B);
Dix = imdilate(Dix1, B);
disp('Idempotency2 LHS:');
disp(Dix);

Dix2 = imdilate(x, B);
disp('Idempotency2 RHS:');
disp(Dix2);

% ========== Colour image processing ==========

% CMY model
RGB=imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
R=RGB;
G=RGB;
B=RGB;
R(:, :, 2)=0;
R(:, :, 3)=0;
G(:, :, 1)=0;
G(:, :, 3)=0;
B(:, :, 1)=0;
B(:, :, 2)=0;
subplot(2, 2, 1), imshow(RGB), title('original image')
subplot(2, 2, 2), imshow(R), title('Red Component')
subplot(2, 2, 3), imshow(G), title('Green Component')
subplot(2, 2, 4), imshow(B), title('Blue Component')

% ========== Remove RGB ==========

a = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
a1 = a;
b1 = a;
c1 = a;
a1(:, :, 1) = 0;  % Remove Red channel
b1(:, :, 2) = 0;  % Remove Green channel
c1(:, :, 3) = 0;  % Remove Blue channel
subplot(2,2,1), imshow(a), title("Original Image");
subplot(2,2,2), imshow(a1), title("Red Missing!");
subplot(2,2,3), imshow(b1), title("Green Missing!");
subplot(2,2,4), imshow(c1), title("Blue Missing!");


%========== pseudo-colouring operation ==========

clc;
clear all;
% Load the image
input_img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
% Check if the image is RGB or grayscale
if size(input_img, 3) == 3
% Convert RGB to grayscale if the image is color
input_img = rgb2gray(input_img);
end
% Convert image to double for processing
input_img = double(input_img);
% Get the size of the image
[m, n] = size(input_img);
% Initialize the output image
output_img = zeros(m, n, 3);
% Perform pseudo-colouring based on intensity values
for i = 1:m
for j = 1:n
if input_img(i, j) >= 0 && input_img(i, j) < 50
output_img(i, j, 1) = input_img(i, j) + 50;
output_img(i, j, 2) = input_img(i, j) + 100;
output_img(i, j, 3) = input_img(i, j) + 10;
elseif input_img(i, j) >= 50 && input_img(i, j) < 100
output_img(i, j, 1) = input_img(i, j) + 35;
output_img(i, j, 2) = input_img(i, j) + 128;
output_img(i, j, 3) = input_img(i, j) + 10;
elseif input_img(i, j) >= 100 && input_img(i, j) < 150
output_img(i, j, 1) = input_img(i, j) + 152;
output_img(i, j, 2) = input_img(i, j) + 130;
output_img(i, j, 3) = input_img(i, j) + 15;
elseif input_img(i, j) >= 150 && input_img(i, j) < 200
output_img(i, j, 1) = input_img(i, j) + 50;
output_img(i, j, 2) = input_img(i, j) + 140;
output_img(i, j, 3) = input_img(i, j) + 25;
elseif input_img(i, j) >= 200 && input_img(i, j) <= 256
output_img(i, j, 1) = input_img(i, j) + 120;
output_img(i, j, 2) = input_img(i, j) + 160;
output_img(i, j, 3) = input_img(i, j) + 45;
end
end
end
% Ensure the output image values are within the valid range [0, 255]
output_img = uint8(min(max(output_img, 0), 255));
% Display the input and pseudo-colored images
subplot(2, 2, 1), imshow(uint8(input_img)), title('Input Image')
subplot(2, 2, 2), imshow(output_img), title('Pseudo Coloured Image')

%========== Gamma correction ==========

clear all;
clc;
I=imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
gamma=1;
max_intensity =255;%for uint8 image
%Look up table creation
LUT = max_intensity .* ( ([0:max_intensity]./max_intensity).^gamma );
LUT = floor(LUT);
%Mapping of input pixels into lookup table values
J = LUT(double(I)+1);
imshow(I), title('original image');
figure, imshow(uint8(J)), title('Gamma corrected image')
xlabel(sprintf('Gamma value is %g', gamma))

```

### **Practical 10**
```matlab
%========== Segmentary ==========

a = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
%Conversion of RGB to YCbCr
b=rgb2ycbcr(a);
%Threshold is applied only to Cb component
mask=b(:, :, 2)>120;
imshow(a), title('original image')
figure, imshow(mask), title('Segmented image')

%========== Edge Detection ==========

a = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
a = rgb2gray(a);
b = edge(a, 'roberts');
c = edge(a, 'sobel');
d = edge(a, 'prewitt');
e = edge(a, 'log');
f = edge(a, 'canny');
figure;
subplot(2,3,1), imshow(a), title('Original Image');
subplot(2,3,2), imshow(b), title('Roberts');
subplot(2,3,3), imshow(c), title('Sobel');
subplot(2,3,4), imshow(d), title('Prewitt');
subplot(2,3,5), imshow(e), title('LoG');
subplot(2,3,6), imshow(f), title('Canny');

%========== Hough transform ==========

img = imread("C:\TANISH PERSONAL\PHOTU\919_1.jpg");
gray = rgb2gray(img);
edges = edge(gray, "canny");
[H, T, R] = hough(edges, "RhoResolution", 0.5, "Theta", -90:0.5:89);
subplot(2,1,1), imshow(img), title('Original Image');
subplot(2,1,2), imshow(imadjust(rescale(H)), 'XData', T, 'YData', R, 'InitialMagnification', 'fit');
xlabel('\theta'), ylabel('\rho'), title('Hough Transform');
axis on, axis normal, colormap hot;
```

