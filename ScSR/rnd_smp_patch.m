function [Xh, Xl] = rnd_smp_patch(input, patch_size, num_patch, upscale)

%img_dir = dir(fullfile(img_path, type));

Xh = [];
Xl = [];

img_num = length(input);
nper_img = zeros(1, img_num);
input1 = reshape(input,[img_num,99,99])
for i1 = 1:length(input1),
    im1 = input1( i1, :, :);
    im11 = reshape(im1,[99,99])
    nper_img(i1) = prod(size(im11));
end
%ÖµÈ¡Õû
%nper_img = floor(nper_img*num_patch/sum(nper_img));

for ii = 1:img_num,
    patch_num = nper_img(ii);
    im2 = input1(ii, :, :);
    im12 = reshape(im2,[99,99]);
    im122 = mapminmax(im12,0,1);
    [H, L] = sample_patches(im122, patch_size, patch_num, upscale);
    Xh = [Xh, H];
    Xl = [Xl, L];
end

patch_path = ['./Training/rnd_patches_' num2str(patch_size) '_' num2str(num_patch) '_s' num2str(upscale) '.mat'];
save(patch_path, 'Xh', 'Xl');