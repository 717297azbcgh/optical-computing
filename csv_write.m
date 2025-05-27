clc;
clear all;

%输入图片循环移位得到3个输入向量，每个输入向量为8bit，切分4个2bit，共12个
% 调制频率为1.25GHz，符号持续时间为4个载波周期，将信号时间切分16份（方便接收）
%% 
data = csvread("output_1480.csv");
signal_1 = zeros(1,320*320);
for i = 1:320
    signal_1((i-1)*320+1:i*320) = data(i,:);
end
% 光向量卷积补充延时
signal_2 = circshift(signal_1,1);
signal_3 = circshift(signal_1,2);

signal_1 = (signal_1 - min(signal_1)) / (max(signal_1) - min(signal_1));
signal_2 = (signal_2 - min(signal_2)) / (max(signal_2) - min(signal_2));
signal_3 = (signal_3 - min(signal_3)) / (max(signal_3) - min(signal_3));

% 浮点有符号数归一化为8bit无符号定点正数
quant_1 = round(signal_1 * 255);
quant_2 = round(signal_2 * 255);
quant_3 = round(signal_3 * 255);

% 切片为4个2bit，从最高位MSB到最低位LSB升序编号
datas1_1 = floor(quant_1 / 64);                
datas1_2 = floor(mod(quant_1, 64) / 16);        
datas1_3 = floor(mod(quant_1, 16) / 4);     
datas1_4 = mod(quant_1, 4);                     

datas2_1 = floor(quant_2 / 64);
datas2_2 = floor(mod(quant_2, 64) / 16);
datas2_3 = floor(mod(quant_2, 16) / 4);
datas2_4 = mod(quant_2, 4);

datas3_1 = floor(quant_3 / 64);
datas3_2 = floor(mod(quant_3, 64) / 16);
datas3_3 = floor(mod(quant_3, 16) / 4);
datas3_4 = mod(quant_3, 4);

datas1_1 = datas1_1 / 3; datas1_2 = datas1_2 / 3;
datas1_3 = datas1_3 / 3; datas1_4 = datas1_4 / 3;

datas2_1 = datas2_1 / 3; datas2_2 = datas2_2 / 3;
datas2_3 = datas2_3 / 3; datas2_4 = datas2_4 / 3;

datas3_1 = datas3_1 / 3; datas3_2 = datas3_2 / 3;
datas3_3 = datas3_3 / 3; datas3_4 = datas3_4 / 3;
%%
% 采样频率与参数
SampleFre = 60e9;
f_zhupin = 1.25e9;
len = 4 * SampleFre / f_zhupin;
t2 = 1/SampleFre:1/SampleFre:len/SampleFre;

for i = 1:3
    for j = 1:4
        eval(['datas_current = datas' num2str(i) '_' num2str(j) ';']);
        part_len = length(datas_current) / 16;
        
        % 分成16个时间片段发送
        for p = 1:16
            tmp = datas_current((p-1)*part_len+1 : p*part_len).';
            k_csv = [zeros(512-128-128,1); ones(128,1); zeros(128,1); tmp; zeros(512,1)];   % 设置同步标记
            pic_csv = zeros(length(k_csv) * len, 1);

            for k = 1:length(k_csv)
                idx = (k-1)*len + (1:len);
                pic_csv(idx) = k_csv(k) * sin(2*pi*f_zhupin*t2);
            end

            filename = ['0416/yiliao_2bit_1.25G_' num2str(i) '_' num2str(j) '_' num2str(p) '.csv'];
            dlmwrite(filename, pic_csv, 'delimiter', ',');
        end
       
    end
end
