clc;
clear all;

%输入10张图片循环移位得到3个输入向量，每个输入向量为8bit，切分4个2bit，输入200张图，共20*12=240份
% 调制频率为1.25GHz，符号持续时间为4个载波周期
%%

data = readmatrix('test_samples.csv');  % 读取 CSV 数据
% 提取图像数据（最后两列是标签，需要去除）
X_test = data(2:end, 1:end-2);
for p=1:20
    signal = X_test(10*(p-1)+1:10*p, :);
    signal_1 = reshape(signal.', 1, []);
    % 光向量卷积补充延时
    signal_2 = circshift(signal_1,1);
    signal_3 = circshift(signal_1,2);

    % 切片为4个2bit，从最高位MSB到最低位LSB升序编号
    datas1_1 = floor(signal_1 / 2^6);
    datas1_2 = floor(mod(signal_1, 2^6) / 2^4);
    datas1_3 = floor(mod(signal_1, 2^4) / 2^2);
    datas1_4 = mod(signal_1, 2^2);

    datas2_1 = floor(signal_2 / 2^6);
    datas2_2 = floor(mod(signal_2, 2^6) / 2^4);
    datas2_3 = floor(mod(signal_2, 2^4) / 2^2);
    datas2_4 = mod(signal_2, 2^2);

    datas3_1 = floor(signal_3 / 2^6);
    datas3_2 = floor(mod(signal_3, 2^6) / 2^4);
    datas3_3 = floor(mod(signal_3, 2^4) / 2^2);
    datas3_4 = mod(signal_3, 2^2);

    datas1_1 = datas1_1 / 3; datas1_2 = datas1_2 / 3;
    datas1_3 = datas1_3 / 3; datas1_4 = datas1_4 / 3;

    datas2_1 = datas2_1 / 3; datas2_2 = datas2_2 / 3;
    datas2_3 = datas2_3 / 3; datas2_4 = datas2_4 / 3;

    datas3_1 = datas3_1 / 3; datas3_2 = datas3_2 / 3;
    datas3_3 = datas3_3 / 3; datas3_4 = datas3_4 / 3;

    % 采样频率与参数
    SampleFre = 60e9;
    f_zhupin = 1.25e9;
    len = 4 * SampleFre / f_zhupin;
    t2 = 1/SampleFre:1/SampleFre:len/SampleFre;

    for i = 1:3
        for j = 1:4
            eval(['datas_current = datas' num2str(i) '_' num2str(j) ';']);

            k_csv = [zeros(512-128-128,1); ones(128,1); zeros(128,1); datas_current.'; zeros(512,1)];
            pic_csv = zeros(length(k_csv) * len, 1);

            for k = 1:length(k_csv)
                idx = (k-1)*len + (1:len);
                pic_csv(idx) = k_csv(k) * sin(2*pi*f_zhupin*t2);
            end

            filename = ['0416/mnist_2bit_1.25G_' num2str(i) '_' num2str(j) '_' num2str(p) '.csv'];
            dlmwrite(filename, pic_csv, 'delimiter', ',');
        end
    end
end
