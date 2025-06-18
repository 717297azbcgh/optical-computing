clear all
clc
%%
weight = [-0.3693022	-0.010668459	-0.1258502
    -0.3631103	0.06318058	0.12932211
    -0.07583695	-0.019362083	0.23523524];

% 归一化 
weight2 = weight ./ max(abs(weight), [], 'all');

% 切片为1个2bit，从最高位MSB到最低位LSB升序编号
weight4 = round(abs(weight2) * (2^32-1));
weight_2bit_all = zeros(3, 3, 16);
for k = 1:16
    weight_2bit_all(:,:,17-k) = floor(mod(weight4 ./ (4^(k-1)), 4)) / 3;
end

data = csvread("output_1480.csv");
signal_1 = zeros(1,320*320);
for i = 1:320
    signal_1((i-1)*320+1:i*320) = data(i,:);
end

signal_2 = (signal_1 - min(signal_1)) / (max(signal_1) - min(signal_1));

quant_1 = round(signal_2 * (2^32 - 1));

datas_all = [];
for k = 1:16
    shift = 32 - 2*k;
    datas_k = bitand(bitshift(quant_1, -shift), 3);
    datas_all = [datas_all; datas_k];
end


%%
output = zeros(16,16,318*318);
ber = zeros(16,16,16,3);  % 光计算误码个数
BER = zeros(16,16);  % 输出切片误码个数

tic
for i=1:16
    for j=1:16
        output_4bit = zeros(320,318,3);
        for k=1:3   
            for p=1:16
                % 生成理论计算结果
                for m=1:3
                    xnew1 = csvread("0416\yiliao_2bit_1.25G_"+num2str(m)+"_"+num2str(i)+"_"+num2str(p)+".csv");
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
                i2=257511;
                img_a_final = img_a_ave(i2-length(img_a_ave)+1:i2-length(img_a_ave)+length(data_act));
                k1 = data_act(512*32+2:4:512*32+2+320*320*32/16-4);
                k2 = img_a_final(512*32+2:4:512*32+2+320*320*32/16-4);

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

                k2_2 = mean(abs(k2_1), 1);
                k2_3 = round(k2_2*9);
                k2_4 = k2_3.*sign(k2_1(7,:));

                k1_2 = mean(abs(k1_1), 1);
                k1_3 = round(k1_2*9);
                k1_4 = k1_3.*sign(k1_1(7,:));

                % MSE= mean((k1_2 - k2_2).^2);
                % SNR= 10 * log10(var(k1_2) /MSE);
                ber(i, j, p, k) = sum(k2_4~=k1_4);
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

                % 16个时间片段组合
                output_1 = reshape(k2_4,320,[]).';
                output_4bit((p-1)*20+1:p*20,:,k) = output_4bit((p-1)*20+1:p*20,:,k) + output_1(:,3:320);
            end
        end
        % 行权值向量加和
        tmp = output_4bit(1:318,:,1)+output_4bit(2:319,:,2)+output_4bit(3:320,:,3);

        output(i,j,:) = reshape(tmp,[],1);
        output_std = conv2(reshape(datas_all(i,:),320,320).',rot90(3.*sign(weight2).*weight_2bit_all(:,:,j), 2),"valid");
        
        BER(i,j) = sum(tmp(:)~=output_std(:));
        
        if BER(i,j)~=0
            disp([i, j])
        end
        toc
    end
end


%%
output_final = zeros(318,318);
for i=1:16
    for j=1:16
        output_final = output_final + reshape(output(i,j,:)* 4 ^ (32 - i - j),318,318);
    end
end

output_final_std = conv2(reshape(quant_1,320,320).',rot90(weight4.*sign(weight2),2),'valid');

sum(find(output_final(:)~=output_final_std(:)))

output_quant = (output_final / (2^32-1)^2 * max(abs(weight), [], 'all') * (max(signal_1) - min(signal_1)) + sum(weight,"all") * min(signal_1));

% dlmwrite('o_output_1480.csv', output_quant, 'delimiter', ',');
