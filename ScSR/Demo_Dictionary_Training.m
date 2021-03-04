% ========================================================================
% Demo codes for dictionary training by joint sparse coding
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

clear all; clc; close all;
addpath(genpath('RegularizedSC'));

TR_IMG_PATH = 'Data/Training';

dict_size   = 1024;          % dictionary size
lambda      = 0.15;         % sparsity regularization
patch_size  = 1;            % image patch size
nSmp        = 10000;       % number of patches to sample
upscale     = 3;            % upscaling factor
%fileinfo = h5info('./data_train_re_10000.h5');
% randomly sample image patches
train_input = h5read('Data\data_train_patch.h5','/input_edsr');
train_label = h5read('Data\data_train_patch.h5','/label_edsr');
%img = load_untouch_nii('/brain.nii.gz');
[Xh, Xl] = rnd_smp_patch(train_label,patch_size, nSmp, upscale);

% prune patches with small variances, threshould chosen based on the
% training data
[Xh, Xl] = patch_pruning(Xh, Xl, 10);

% joint sparse coding 
[Dh, Dl] = train_coupled_dict(Xh, Xl, dict_size, lambda, upscale);
dict_path = ['Dictionary/D_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '_4.12_.mat' ];
save(dict_path, 'Dh', 'Dl');