
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ���ܣ�����һ����ͼ���ֵ�����PSNR
% ��RGBת��YCbCr��ʽ���м���
% ���ֱ�Ӽ�����ת�����ֵҪС2dB���ң���Ȼ�Ǹ�����ԣ�
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_path_X =  'C:\Users\Administrator\Desktop\�µķ���\cam4\';% �ο�ͼ���ļ���·��
file_path_Y = 'C:\Users\Administrator\Desktop\�µķ���\duibi\'; % �Ա�ͼ���ļ�·��
 fid1=['C:\Users\Administrator\Desktop\�µķ���\','4.txt'];
    c=fopen(fid1,'w'); 
img_path_list_X = dir(strcat(file_path_X,'*.jpg'));%��ȡ���ļ���������jpg��ʽ��ͼ��  
img_path_list_Y = dir(strcat(file_path_Y,'*.jpg'));%��ȡ���ļ���������jpg��ʽ��ͼ��  

img_num = length(img_path_list_X);%��ȡͼ��������  
if img_num > 0 %������������ͼ��  
        for j = 1:img_num %��һ��ȡͼ��  
            image_name_X = img_path_list_X(j).name;% ͼ���� 
            image_name_Y = img_path_list_Y(j).name;% ͼ���� 
            X =  imread(strcat(file_path_X,image_name_X));  
            Y = imread(strcat(file_path_Y,image_name_Y));  
            fprintf('%d %s\n',j,strcat(file_path_X,image_name_X));% ��ʾ���ڴ����ͼ����
            fprintf('%d %s\n',j,strcat(file_path_Y,image_name_Y));% ��ʾ���ڴ����ͼ����  
            %ͼ������� 
            PSNR = metrix_psnr(X, Y);
            SSIM = metrix_ssim(X, Y); 
            fprintf(c,'%f\n\t',PSNR);        %%%qΪ��Ҫд������ݣ���'%f��Ϊ���ݸ�ʽ
            fprintf(c,'%f\n\t\n',SSIM);
            
            

        end  
end 
fclose(c);




 