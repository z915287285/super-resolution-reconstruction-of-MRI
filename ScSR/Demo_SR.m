% =========================================================================
% Simple demo codes for image super-resolution via sparse representation
%
% Reference
%   J. Yang et al. Image super-resolution as sparse representation of raw
%   image patches. CVPR 2008.
%   J. Yang et al. Image super-resolution via sparse representation. IEEE 
%   Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010
%
% Jianchao Yang
% ECE Department, University of Illinois at Urbana-Champaign
% For any questions, send email to jyang29@uiuc.edu
% =========================================================================

clear all; 
clc;
close all;

% read test image
img = load_untouch_nii('/brain.nii.gz');
im_o = img.img;
im_o=im_o(:,:,120);
%figure,imshow(im_o,[]);

maxV = max(im_o(:));
minV = min(im_o(:));
im_h = (im_o-minV)./(maxV-minV);



im_h_mod = im_h(1:258,1:309);
im_l= imresize(im_h_mod, 1/3, 'nearest');
%maxI = max(im_l(:));
%minI = min(im_l(:));
%im_l1 = (im_l-minI)./(maxI-minI);
%im_l= imresize(tmp, 3, 'bicubic');

% set parameters
lambda = 0.2;                   % sparsity regularization
overlap = 5;                    % the more overlap the better (patch size 5x5)
up_scale = 3;                   % scaling factor, depending on the trained dictionary
maxIter =50;                   % if 0, do not use backprojection

% load dictionary
% load('Dictionary/D_512_0.15_5_s3.mat');
load('Dictionary/author/D_512_0.15_5.mat');
% change color space, work on illuminance only
% im_l_ycbcr = rgb2ycbcr(im_l);
% im_l_y = im_l_ycbcr(:, :, 1);
% im_l_cb = im_l_ycbcr(:, :, 2);
% im_l_cr = im_l_ycbcr(:, :, 3);


% image super-resolution based on sparse representation
[im_h_y] = ScSR(im_l, up_scale, Dh, Dl, lambda, overlap);
[im_h_y1] = backprojection(im_h_y, im_l, maxIter);
maxh = max(im_h_y1(:));
minh = min(im_h_y1(:));
im_h_y2 = (im_h_y1-minh)./(maxh-minh)
% upscale the chrominance simply by "bicubic" 
[nrow, ncol] = size(im_h_y);
% im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
% im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');

% im_h_ycbcr = zeros([nrow, ncol, 3]);
% im_h_ycbcr(:, :, 1) = im_h_y;
% im_h_ycbcr(:, :, 2) = im_h_cb;
% im_h_ycbcr(:, :, 3) = im_h_cr;
% im_h = ycbcr2rgb(uint8(im_h_ycbcr));
im_h= im_h_y2;
% bicubic interpolation for reference
im_b = imresize(im_l, 3, 'bicubic');
maxb = max(im_b(:));
minb = min(im_b(:));
im_b1 = (im_b-minb)./(maxb-minb)
%im_b=double( im_b);
% read ground truth image
% im = imread('Data1/Test/7_180_0.305735833183102.png');
% im = double(im);
im=im_h_mod;
% compute PSNR for the illuminance channel
bb_rmse = compute_rmse(im, im_b1);
sp_rmse = compute_rmse(im, im_h);

bb_psnr = 20*log10(255/bb_rmse);
sp_psnr = 20*log10(255/sp_rmse);

fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);


bb_ssim = metrix_ssim(im, im_b1);
sp_ssim = metrix_ssim(im, im_h);

fprintf('SSIM for Bicubic Interpolation: %f dB\n', bb_ssim);
fprintf('SSIM for Sparse Representation Recovery: %f dB\n', sp_ssim);
% show the images
figure, imshow(im_h,[]);
title('Sparse Recovery');
sc=getimage(gcf);
imwrite(sc,'sc_image.png')
figure, imshow(im_b1,[]);
title('Bicubic Interpolation');
bc=getimage(gcf);
imwrite(bc,'bc_image.png')
figure, imshow(im);
title('H');
h=getimage(gcf);
imwrite(h,'h_image.png')