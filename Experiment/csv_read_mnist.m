clear all
clc
%%
weight = [ 0.38094833  0.36529493  0.24226627;
    0.44256777  0.33907723  0.21108471;
    0.2424691   0.09382253 -0.10800746];
% 归一化
weight2 = weight ./ max(abs(weight), [], 'all');

% 切片为4个2bit，从最高位MSB到最低位LSB升序编号
weight4 = round(abs(weight2) * (2^8-1));
weight_2bit_all = zeros(3, 3, 4);
for k = 1:3
    weight_2bit_all(:,:,5-k) = floor(mod(weight4 ./ (4^(k-1)), 4)) / 3;
end

data = readmatrix('test_samples.csv'); 
X_test = data(2:end, 1:end-2);
signal_1 = reshape(X_test.', 1, []);

datas1 = floor(signal_1 / 2^6);
datas2 = floor(mod(signal_1, 2^6) / 2^4);
datas3 = floor(mod(signal_1, 2^4) / 2^2);
datas4 = mod(signal_1, 2^2);

datas_all = [datas1; datas2; datas3; datas4;];

%%
output = zeros(4,4,26*26,200);
ber = zeros(4,4,200,3);  % 光计算误码个数

tic
for i=1:4
    for j=1:4
        output_4bit = zeros(28,26,200,3);
        for k=1:3
            for p=1:20
                % 生成理论计算结果
                for m=1:3
                    xnew1 = csvread("0416\mnist_2bit_1.25G_"+num2str(m)+"_"+num2str(i)+"_"+num2str(p)+".csv");
                    xnew = xnew1(6:6:end);
                    if m==1
                        data_act = xnew*sign(weight2(k,4-m))*weight_2bit_all(k,4-m,j);
                    else
                        data_act = data_act + xnew*sign(weight2(k,4-m))*weight_2bit_all(k,4-m,j);
                    end
                end

                img_osa = csvread("4.16\part\"+num2str(i)+"-"+num2str(j)+"-"+num2str(p)+"-"+num2str(k)+".csv");
                img_a_ave = img_osa;
                % r = xcorr(img_a_ave,data_act);
                % [cor,i2] = max(abs((r)));
                
                % 时间对齐以得到实际计算结果
                i2=307455;
                img_a_final = img_a_ave(i2-length(img_a_ave)+1:i2-length(img_a_ave)+length(data_act));
                k1 = data_act(512*32+2:4:512*32+2+28*28*10*32-4);
                k2 = img_a_final(512*32+2:4:512*32+2+28*28*10*32-4);

                % 归一化
                pf = zeros(2,1);
                pf(1) = sum((k1 - mean(k1)) .* (k2 - mean(k2))) / sum((k1 - mean(k1)).^2);
                if ~isnan(pf(1))
                    pf(2) = mean(k2) - pf(1) * mean(k1);
                    k2 = (k2-pf(2))/pf(1);
                else
                    k2 = k2-mean(k2);
                end

                % 取平均再量化
                k1_1 = reshape(k1,8,[]);
                k2_1 = reshape(k2,8,[]);

                k2_2 = mean(abs(k2_1([3 4 7 8],:)), 1);
                k2_3 = round(k2_2*9);
                k2_4 = k2_3.*sign(k2_1(7,:));

                k1_2 = mean(abs(k1_1([3 4 7 8],:)), 1);
                k1_3 = round(k1_2*9);
                k1_4 = k1_3.*sign(k1_1(7,:));

                % MSE= mean((k1_2 - k2_2).^2);
                % SNR= 10 * log10(var(k1_2) /MSE);
                ber(i, j, p, k) = sum(k2_4 ~= k1_4);
                if ber(i, j, p, k)~=0
                    disp([i, j, p, k])               
                    error_idx = find(k2_4 ~= k1_4);
                    original_idx = error_idx * 8;
                    figure('Name',num2str(i)+"-"+num2str(j)+"-"+num2str(p)+"-"+num2str(k));
                    plot(abs(k1)*9, '-b', 'LineWidth', 1.5);
                    hold on
                    plot(abs(k2)*9, '-r', 'LineWidth', 1.5);
                    scatter(original_idx, abs(k2(original_idx)*9), 50, 'k', 'filled')
                end
                
                output_1 = permute(reshape(k2_4,28,28,[]),[2 1 3]);
                output_4bit(:,:,(p-1)*10+1:p*10,k) = output_1(:,3:28,:);
            end
        end
        % 行权值向量加和
        tmp = output_4bit(1:26,:,:,1)+output_4bit(2:27,:,:,2)+output_4bit(3:28,:,:,3);
        tmp = squeeze(tmp);

        output(i,j,:,:) = reshape(tmp,26*26,200);
        toc
    end
end


%%
output_final = zeros(26,26,200);
for i=1:4
    for j=1:4
        output_final = output_final + reshape(output(i,j,:,:) * 4 ^ (8 - i - j),26,26,200);
    end
end

BER=zeros(1,200);
output_final_std = zeros(26,26,200);
for i=1:200
    output_final_std(:,:,i) = conv2(reshape(signal_1(784*(i-1)+1:784*i),28,28).',rot90(weight4.*sign(weight2),2),'valid');
    BER(i)=sum(find(output_final(:,:,i)~=output_final_std(:,:,i)));
end
disp(find(BER~=0))

output_quant = (output_final / (2^8-1)^2 * max(abs(weight), [], 'all') - sum(weight,"all") * 0.1307) / 0.3081;

output_std = zeros(26,26,200);
for i=1:200
    output_std(:,:,i) = conv2((reshape(signal_1(784*(i-1)+1:784*i),28,28).' /255 -0.1307) / 0.3081,rot90(weight,2),'valid');
end

signal_power = mean((output_std).^2,"all");
noise_power = mean((output_std - output_quant).^2,'all');
SNR = 10 * log10(signal_power / noise_power);
% dlmwrite('o_output_1480.csv', output_quant, 'delimiter', ',');
