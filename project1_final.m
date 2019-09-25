%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
org_image = imread('pollen.png'); % open original image
%org_image = rgb2gray(org_image);    % grayscale
[width, height] = size(org_image);  % dimensions
total_pixels = width*height;        % total pixels

L_in = numel(unique(org_image));     % input gray levels
L_out = 256;                        % output gray levels

tau = L_in/L_out;

lo_gray = min(org_image(:));         % lowest gray level input
hi_gray = max(org_image(:));          % highest gray level input

clip_limit = 4; % a multiple of the average

max_distortion = 4; % gray level distortion

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Histogram Equalization
hist = zeros(hi_gray-lo_gray,2);    
for i = lo_gray:hi_gray
    hist(i,1) = i;
    hist(i,2) = sum(org_image(:) == i);
end
hist(:,2) = hist(:,2)./total_pixels;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CDF
CDF = zeros(hi_gray-lo_gray,2);  
for i = lo_gray:hi_gray
    CDF(i,1) =  hist(i,1);
    CDF(i,2) = sum(hist((1:i),2));
end
CDF_scaled = ((CDF-min(CDF))./(max(CDF)-min(CDF)))*255;
CDF_scaled_double = CDF_scaled(:,2);
CDF_scaled = uint8(round(CDF_scaled));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Contrast-limited Adaptive Histogram Equalization
clip_values = hist(:,2)-(mean(hist(:,2)).*clip_limit);
clip_values(clip_values < 0) = 0;
CLAHE_offset = sum(clip_values)/L_in;
CLAHE = (hist(:,2) - clip_values);
CLAHE = CLAHE + CLAHE_offset;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CDF for CLAHE
CDF_CLAHE = cumsum(CLAHE);
CDF_CLAHE_scaled = ((CDF_CLAHE-min(CDF_CLAHE))./(max(CDF_CLAHE)-min(CDF_CLAHE)))*255;
CDF_CLAHE_scaled = uint8(round(CDF_CLAHE_scaled));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transformations
HE = CDF_scaled(org_image,2);
HE = reshape(HE,[width,height]);

CLAHE_im = CDF_scaled(org_image,2);
CLAHE_im = reshape(CLAHE_im,[width,height]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OCTM ALGORITHM
d = 10; % set distortion limit 1/d
u = 10;  % set upper limit

A = ones(1,size(hist(:,2),1));
b = L_out - 1; % 255

gray_level = size(hist,1);

lower_bound = repmat(1/d, gray_level);  % create lower limit for all iterations 
upper_bound = repmat(u, gray_level);    % create upper limit for all iterations

% Apply linear programming function
[sj,fval,exitflag,output] = linprog(-hist(:,2)',A,b,[],[],lower_bound,upper_bound);
disp(output);
% A.*sj < b
% lowerbound(i) < sj(i) < upperbound(i)

% get cumulative sum
sj_cum = cumsum(sj);
sj_cum = round(sj_cum + 0.5);
sj_cum = ((sj_cum-min(sj_cum))./(max(sj_cum)-min(sj_cum)))*255;
sj_cum = uint8(round(sj_cum));
OCTM = sj_cum(org_image);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Figures
% Histogram
figure
subplot(3,2,1);
plot(hist(:,2));            % Plot Histogram
title('Original Histogram');

subplot(3,2,2);
plot(CDF_scaled(:,2));      % CDF of Histogram
title('HE CDF');

% CLAHE
subplot(3,2,3);
plot(CLAHE);
title('CLAHE Histogram');

subplot(3,2,4);
plot(CDF_CLAHE_scaled);
title('CLAHE CDF');

% OCTM
subplot(3,2,5);
plot(sj);
title('Optimized S values');
subplot(3,2,6);
plot(sj_cum);
title('OCTM CDF');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Show images
figure
subplot(2,2,1);
imshow(org_image);
title('Original Image')
subplot(2,2,2);
imshow(HE);
title('Histogram Equalization')
subplot(2,2,3);
imshow(CLAHE_im);
title('CLAHE')
subplot(2,2,4);
imshow(OCTM);
title('OCTM')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;