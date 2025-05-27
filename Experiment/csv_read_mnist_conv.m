weight = [ 0.38094833  0.36529493  0.24226627;
    0.44256777  0.33907723  0.21108471;
    0.2424691   0.09382253 -0.10800746];
% 归一化
weight2 = weight ./ max(abs(weight), [], 'all');

data = readmatrix('test_samples.csv');
X_test = data(2:end, 1:end-2);
signal_1 = reshape(X_test.', 1, []);

%%
output = zeros(28,26,200,3);

for k=1:3
    for p=1:20
        % 生成理论计算结果
        for m=1:3
            xnew1 = csvread("0416\conv\mnist_1.25G_"+num2str(m)+"_"+num2str(p)+".csv");
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

        output_1 = permute(reshape(k2_3,28,28,[]),[2 1 3]);
        output(:,:,(p-1)*10+1:p*10,k) = output_1(:,3:28,:);
    end
end
% 行权值向量加和
tmp = output(1:26,:,:,1)+output(2:27,:,:,2)+output(3:28,:,:,3);
output_conv = squeeze(tmp);

output_final_std = zeros(26,26,200);
for i=1:200
    output_final_std(:,:,i) = conv2(reshape(signal_1(784*(i-1)+1:784*i)/255,28,28).',rot90(weight2,2),'valid');
end

signal_power = mean((output_final_std).^2,"all");
noise_power = mean((output_final_std - output_conv).^2,'all');
SNR = 10 * log10(signal_power / noise_power);

%%
output_conv_1 = (output_conv * max(abs(weight), [], 'all') - sum(weight,"all") * 0.1307) / 0.3081;

% dlmwrite('o_output_1480.csv', output_conv_1, 'delimiter', ',');

output_std = zeros(26,26,200);
for i=1:200
    output_std(:,:,i) = conv2((reshape(signal_1(784*(i-1)+1:784*i),28,28).' /255 -0.1307) / 0.3081,rot90(weight,2),'valid');
end

% dlmwrite('o_output_1480.csv', output_conv, 'delimiter', ',');
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


% dlmwrite('o_conv.csv', reshape(permute(output_conv_1,[2 1 3]),26*26,[]).', 'delimiter', ',');
% dlmwrite('o_quant.csv', reshape(permute(output_quant,[2 1 3]),26*26,[]).', 'delimiter', ',');
