weight = [-0.3693022	-0.010668459	-0.1258502
    -0.3631103	0.06318058	0.12932211
    -0.07583695	-0.019362083	0.23523524];

% 归一化
weight2 = weight ./ max(abs(weight), [], 'all');

data = csvread("output_1480.csv");
signal_1 = zeros(1,320*320);
for i = 1:320
    signal_1((i-1)*320+1:i*320) = data(i,:);
end

signal_2 = (signal_1 - min(signal_1)) / (max(signal_1) - min(signal_1));

%%
output_conv = zeros(318,318);
output = zeros(320,318,3);

for k=1:3
    for p=1:16
        % 生成理论计算结果
        for m=1:3
            xnew1 = csvread("0416\conv\yiliao_1.25G_"+num2str(m)+"_"+num2str(p)+".csv");
            xnew = xnew1(6:6:end);
            if m==1
                data_act = xnew*weight2(k,4-m);
            else
                data_act = data_act + xnew*weight2(k,4-m);
            end
        end

        img_osa = csvread("4.16\part\"+num2str(p)+"-"+num2str(k)+".csv");
        img_a_ave = img_osa;
        % r = xcorr(img_a_ave,data_act);
        % [cor,i2] = max(abs((r)));

        % 时间对齐以得到实际计算结果
        i2=257455;
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
        k1_2 = mean(abs(k1_1), 1);

        k2_3 = k2_2.*sign(k2_1(7,:));
        k1_3 = k1_2.*sign(k1_1(7,:));

        MSE= mean((k1_3 - k2_3).^2);
        SNR= 10 * log10(var(k1_3) /MSE);

        % figure('Name',num2str(p)+"-"+num2str(k));
        % plot(abs(k1)*9, '-b', 'LineWidth', 1.5);
        % hold on
        % plot(abs(k2)*9, '-r', 'LineWidth', 1.5);

        % 16个时间片段组合
        output_1 = reshape(k2_3,320,[]).';
        output((p-1)*20+1:p*20,:,k) = output((p-1)*20+1:p*20,:,k) + output_1(:,3:320);
    end
end

% 行权值向量加和
output_conv = output(1:318,:,1)+output(2:319,:,2)+output(3:320,:,3);
output_std = conv2(reshape(signal_2,320,320).',rot90(weight2, 2),"valid");

signal_power = mean((output_std).^2,"all");
noise_power = mean((output_std - output_conv).^2,'all');
SNR = 10 * log10(signal_power / noise_power);

%%

% output_final_std = conv2(reshape(quant_1,320,320).',rot90(weight4.*sign(weight2),2),'valid');

% sum(find(output_final(:)~=output_final_std(:)))

output_conv_1 = (output_conv * max(abs(weight), [], 'all') * (max(signal_1) - min(signal_1)) + sum(weight,"all") * min(signal_1));
output_std = conv2(reshape(signal_1,320,320).',rot90(weight, 2),"valid");
% dlmwrite('o_output_1480.csv', output_conv_1, 'delimiter', ',');
% save("tmp.mat","output_std","output_conv_1","output_quant","-mat")

%%
load('tmp')
output_quant_1 = output_quant;
output_std_1 = output_std;
% output_conv_1 = output_conv_1;

output_quant_1 = output_quant_1./max(max(abs(output_std_1)));
output_conv_1 = output_conv_1./max(max(abs(output_std_1)));
output_std_1 = output_std_1./max(max(abs(output_std_1)));

erros = output_std_1(:)-output_quant_1(:);
erros2 = output_std_1(:)-output_conv_1(:);

% 使用hist函数生成直方图
figure;
[counts_svd, bins_svd] = hist(erros2, 50, 'Normalization', 'pdf');
[counts_gpst, bins_gpst] = hist(erros, bins_svd, 'Normalization', 'pdf');

%计算概率密度曲线
[f1, x1] = ksdensity(erros2,bins_svd);
[f2, x2] = ksdensity(erros,bins_gpst);
f1=f1./sum(f1).*100;
f2=f2./sum(f2).*100;

bar(bins_svd, counts_svd./sum(counts_svd).*100,'LineWidth', 1);
hold on 
bar(bins_gpst, counts_gpst./sum(counts_gpst).*100,'LineWidth', 1);
hold on 
plot(x1, f1, 'r', 'LineWidth', 2);  %svd

hold on 
plot(x2, f2, "Color",[0.9255, 0.7059, 0.1412], 'LineWidth', 2);  %gpst
legend("Normal",'Ours')
set(gca,'fontsize',13);
xlabel("error")
ylabel("probability(%)")

set(gcf,'unit','centimeters','position',[16 10 9 8])


% dlmwrite('o_conv.csv', output_conv_1, 'delimiter', ',');
% dlmwrite('o_quant.csv', output_quant, 'delimiter', ',');

