
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 功能：计算一连串图像峰值信噪比PSNR
% 将RGB转成YCbCr格式进行计算
% 如果直接计算会比转后计算值要小2dB左右（当然是个别测试）
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_path_X =  'C:\Users\Administrator\Desktop\新的方案\cam4\';% 参考图像文件夹路径
file_path_Y = 'C:\Users\Administrator\Desktop\新的方案\duibi\'; % 对比图像文件路径
 fid1=['C:\Users\Administrator\Desktop\新的方案\','4.txt'];
    c=fopen(fid1,'w'); 
img_path_list_X = dir(strcat(file_path_X,'*.jpg'));%获取该文件夹中所有jpg格式的图像  
img_path_list_Y = dir(strcat(file_path_Y,'*.jpg'));%获取该文件夹中所有jpg格式的图像  

img_num = length(img_path_list_X);%获取图像总数量  
if img_num > 0 %有满足条件的图像  
        for j = 1:img_num %逐一读取图像  
            image_name_X = img_path_list_X(j).name;% 图像名 
            image_name_Y = img_path_list_Y(j).name;% 图像名 
            X =  imread(strcat(file_path_X,image_name_X));  
            Y = imread(strcat(file_path_Y,image_name_Y));  
            fprintf('%d %s\n',j,strcat(file_path_X,image_name_X));% 显示正在处理的图像名
            fprintf('%d %s\n',j,strcat(file_path_Y,image_name_Y));% 显示正在处理的图像名  
            %图像处理过程 
            PSNR = metrix_psnr(X, Y);
            SSIM = metrix_ssim(X, Y); 
            fprintf(c,'%f\n\t',PSNR);        %%%q为你要写入的数据，“'%f”为数据格式
            fprintf(c,'%f\n\t\n',SSIM);
            
            

        end  
end 
fclose(c);




 