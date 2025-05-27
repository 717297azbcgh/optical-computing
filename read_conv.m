clear
clc

%%
ber=zeros(11,3*16);
A1=zeros(3,16);
A2=zeros(3,16);
A_mean=zeros(3,16);
A_std=zeros(3,16);
k1_flat = zeros(3*16*51200, 1);
k2_flat = zeros(3*16*51200, 1);
idx = 1;
for p=1:3
    for m=1:16
        for i=1:1
            xnew1 = csvread("0416\conv\yiliao_1.25G_"+num2str(p)+"_"+num2str(m)+".csv");
            img_osa = csvread("4.16\conv\"+num2str(p)+"-"+num2str(m)+"-"+num2str(i)+".csv");
            xnew = xnew1(6:6:end);
            % 找点
            img_a_ave = (img_osa- mean(img_osa))./std(img_osa).*std(xnew);
            r = xcorr(img_a_ave,xnew);
            [cor,i2((p-1)*16+m)] = max(abs((r)));
            % i2((p-1)*16+m)=257454;

            img_a_final = img_osa(i2((p-1)*16+m)-length(img_osa)+1:i2((p-1)*16+m)-length(img_osa)+length(xnew));
            % a1 = img_osa(i2-length(img_osa)+256*32+2:4:i2-length(img_osa)+256*32+128*32-2);
            % A1((p-1)*128+(q-1)*16+m) = median(a1(1:2:end));
            % A2((p-1)*128+(q-1)*16+m) = median(a1(2:2:end));
            % img_final = (img_a_final-(A1((p-1)*128+(q-1)*16+m)+A2((p-1)*128+(q-1)*16+m))/2)/(A1((p-1)*128+(q-1)*16+m)-A2((p-1)*128+(q-1)*16+m))*2;

            % 有效信号
            k1 = xnew(512*32+2:4:512*32+2+320*320*32/16-4);
            k2 = img_a_final(512*32+2:4:512*32+2+320*320*32/16-4);
            % 取点
            k2_1 = reshape(k2,8,[]);
            k2_2 = reshape(k2_1([7,8], :),[],1);

            k1_1 = reshape(k1,8,[]);
            k1_2 = reshape(k1_1([7,8], :),[],1);

            % 得到采样信号的幅值范围
            if sum(round(k1_2*3)>0)~=0
                [~,max_idx]=max(round(k1_2*3));
                [~,min_idx]=min(round(k1_2*3));
                max_idx = find(round(k1_2*3)==round(k1_2(max_idx)*3));
                min_idx = find(round(k1_2*3)==round(k1_2(min_idx)*3));

                A1(p,m) = mean(k2_2(max_idx)*sign(k2_2(max_idx(1))));
                A2(p,m) = mean(k2_2(min_idx)*sign(k2_2(max_idx(1))));
            else
                A1(p,m) = max(k2_2);
                A2(p,m) = min(k2_2);
            end
            A_mean(p,m)=mean(k2_2);
            A_std(p,m)=std(k2_2);


            % Fs = 10e9;                           % 采样率 10 GHz
            % N = length(img_a_final);                   % 信号长度
            % f = (-N/2:N/2-1)*(Fs/N);            % 构造频率轴
            %
            % % 计算频谱并归一化
            % K2_f = fftshift(fft(img_a_final));
            % K2_mag = 20*log10(abs(K2_f));   % 转换为 dB 幅度
            % K1_f = fftshift(fft(xnew));
            % K1_mag = 20*log10(abs(K1_f));   % 转换为 dB 幅度
            %
            % % 画图
            % figure;
            % plot(f/1e9, K1_mag, 'b', 'LineWidth', 1.5);
            % xlabel('频率 (GHz)');
            % ylabel('幅度 (dB)');
            % title('k1 信号的频谱');
            % grid on;
            % xlim([0 Fs/2/1e9]);
            % figure;
            % plot(f/1e9, K2_mag, 'b', 'LineWidth', 1.5);
            % xlabel('频率 (GHz)');
            % ylabel('幅度 (dB)');
            % title('k2 信号的频谱');
            % grid on;
            % xlim([0 Fs/2/1e9]);
            % figure;
            % plot(f/1e9, K2_mag-K1_mag, 'b', 'LineWidth', 1.5);
            % xlabel('频率 (GHz)');
            % ylabel('幅度 (dB)');
            % title('CSI 频率');
            % grid on;
            % xlim([0 Fs/2/1e9]);

            % Fs = 10e9;                     % 采样率 10 GHz
            % N = length(img_a_final);              % 信号长度
            % f = (-N/2:N/2-1)*(Fs/N);       % 构造频率轴，单位 Hz
            %
            % % % 傅里叶变换并中心化
            % K2_f = fftshift(fft(img_a_final));
            %
            % f_abs = abs(f);
            % atten_dB = zeros(size(f));
            % atten_dB(f_abs > 10e9) = -80;
            % idx = (f_abs <= 10e9);
            % atten_dB(idx) = 1.9008e-09 * f_abs(idx);
            %
            % atten_linear = 10.^(atten_dB / 20).';
            % % 逆变换回时域
            % % img_a_final = ifft(ifftshift(K2_f .*atten_linear), 'symmetric');
            % k1 = xnew(512*32+2:4:512*32+2+320*320*32/16-4);
            % k2 = img_a_final(512*32+2:4:512*32+2+320*320*32/16-4);


            % 归一化
            % if round(A1(p,q,m)/0.03)~=0
            % k2_1 = reshape(k2,8,[]);
            % k2_2 = reshape(k2_1([7,8], :),[],1);
            %
            % k1_1 = reshape(k1,8,[]);
            % k1_2 = reshape(k1_1([7,8], :),[],1);

            pf = zeros(2,1);
            pf(1) = sum((k1 - mean(k1)) .* (k2 - mean(k2))) / sum((k1 - mean(k1)).^2);
            if ~isnan(pf(1))
                pf(2) = mean(k2) - pf(1) * mean(k1);
                % pf = polyfit(k1_2, k2_2, 1);
                k2 = (k2-pf(2))/pf(1);
            else
                k2 = k2-mean(k2);
            end
            %
            % if sum(round(k1_2*3)>0)~=0
            %     k2 = (k2-mean(k2_2))/std(k2_2)*std(k1_2);
            % else
            %     k2 = k2-mean(k2_2);
            % end

            % k2 = (k2-mean(k2_2))/std(k2_2)*std(k1_2); % '1'值接近
            % k2 = (k2-(A1(p,q,m)+A2(p,q,m))/2)/(A1(p,q,m)-A2(p,q,m))*2*round(A1(p,q,m)/0.03)/3; % '1'值接近
            % 数据衰减较大，特殊处理
            % if p==1&&q==4&&m==8
            % k2 = k2 / 2 * 3;
            % end
            % else
            % k2 = (k2-mean(k2))/0.09;
            % k2 = (k2-(A1(p,q,m)+A2(p,q,m))/2)/0.09;
            % end
            % k2 = 0.0192*k2.^3  +0.9872*k2;

            % 8个点，不同的取点方式；9是8点平均，10是3、4、7、8取平均，11是7、8取平均
            for k=1:11
                k2_1 = reshape(abs(k2),8,[]);
                if k<9
                    k2_1 = k2_1(k, :);
                elseif k==9
                    k2_1 = mean(k2_1, 1);
                elseif k==10
                    k2_1 = mean(k2_1([3,4,7,8],:), 1);
                else
                    k2_1 = mean(k2_1([7,8],:), 1);
                end
                % k2_1 = max(k2_1, [], 1);
                % k2_2 = round(k2_1*9)/9;

                k1_1 = reshape(abs(k1),8,[]);
                if k<9
                    k1_1 = k1_1(k, :);
                elseif k==9
                    k1_1 = mean(k1_1, 1);
                elseif k==10
                    k1_1 = mean(k1_1([3,4,7,8],:), 1);
                else
                    k1_1 = mean(k1_1([7,8],:), 1);
                end
                % k1_1 = max(k1_1, [], 1);
                % k1_2 = round(k1_1*3)/3;

                MSE(k,(p-1)*16+m)= mean((k1_1 - k2_1).^2);
                SNR(k,(p-1)*16+m)= 10 * log10(var(k1_1) /MSE(k,(p-1)*16+m));
                ber(k,(p-1)*16+m) = sum(k2_1~=k1_1);
            end

            % if std(k1_2)~=0
            %     coefficients=0.92:0.001:1.05;
            %     for k=1:length(coefficients)
            %         mse(k)= mean((k1_1 - k2_1/coefficients(k)).^2);
            %     end
            %     [~,k]=min(mse);
            %     A_std(p,q,m) = A_std(p,q,m) * coefficients(k);
            %     if coefficients(k)~=1
            %         disp(coefficients(k))
            %     end
            % end
            % 11取点的误码处画图
            % if ber(11,(p-1)*128+(q-1)*16+m)~=0
            % disp([p,q,m])
            % disp(ber(:,(p-1)*128+(q-1)*16+m))
            % error_idx = find(k2_2 ~= k1_2);
            % original_idx = error_idx*8;
            % fig = figure;
            % % fig.WindowState = 'maximized';
            % plot(abs(k1)*27, '-b', 'LineWidth', 1.5);
            % hold on
            % plot(abs(k2)*27, '-r', 'LineWidth', 1.5);
            % scatter(original_idx, abs(k1(original_idx))*27, 50, 'k', 'filled')
            % % end

            % 记录归一化结果
            k1_flat(idx:idx+51200-1) = k1(:);
            k2_flat(idx:idx+51200-1) = k2(:);
            idx = idx + 51200;


        end
    end
end
%%
% % load("A.mat")
figure;plot(A1(:))
figure;plot(A2(:))
% % save('A.mat',"A1","A2")


%% 查看分布
tmp=reshape(k2_flat,8,[]);
tmp1=reshape(tmp(7:8,:),[],1);
tmp2=mean(abs(tmp([7,8],:)),1);
% tmp1=0.0192*tmp1.^3 +0.9872*tmp1;
figure;histogram(tmp1*9);


aaa1=tmp1(round(tmp1*3)==0);
aaa2=tmp1(round(tmp1*3)==1);
aaa3=tmp1(round(tmp1*3)==2);
aaa4=tmp1(round(tmp1*3)==3);
aaa5=tmp1(round(tmp1*3)==-1);
aaa6=tmp1(round(tmp1*3)==-2);
aaa7=tmp1(round(tmp1*3)==-3);

x=[0,1/3,2/3,3/3,-1/3,-2/3,-3/3];
aaa_mean(1)=mean(aaa1);
aaa_mean(2)=mean(aaa2);
aaa_mean(3)=mean(aaa3);
aaa_mean(4)=mean(aaa4);
aaa_mean(5)=mean(aaa5);
aaa_mean(6)=mean(aaa6);
aaa_mean(7)=mean(aaa7);
aaa_std(1)=std(aaa1);
aaa_std(2)=std(aaa2);
aaa_std(3)=std(aaa3);
aaa_std(4)=std(aaa4);
aaa_std(5)=std(aaa5);
aaa_std(7)=std(aaa6);
aaa_std(4)=std(aaa7);

aaa_mean*9
3.5120*sin(0.2877*x)*9
round(tmp1*9)/9;





% %%
% figure
% for p=1:3
%     for q=1:8
% subplot(3,8,(p-1)*8+q)
% boxplot(squeeze(a1(p,q,:)));
% % ylim([0 0.5])
% A_mean(p,q) = mean(squeeze(a1(p,q,:)));
%     end
% end
% %%
% figure
% for p=1:3
%     for q=1:8
% subplot(3,8,(p-1)*8+q)
% boxplot(squeeze(a2(p,q,:)));
% % ylim([0 0.5])
% A_mean(p,q) = mean(squeeze(a2(p,q,:)));
%     end
% end
