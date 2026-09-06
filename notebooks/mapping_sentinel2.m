clc;clear all;
% Area 
addpath(genpath('D:\HNS'))
addpath(genpath('C:\Program Files\MATLAB\R2022b'))
addpath(genpath('C:\Users\SEOUL\Documents\MATLAB\Sentinel'))

%fl = ls('*.nc');

%for k = 1  % 1:size(fl,1)

    % file_name
    %fn = fl(k,:);
    
    fn = 'Busan1_S2A_MSIL2A_20210529T020651_N0500_R103_T52SDD_20230227T170302_resampled.nc';
 

    % load variables (lon,lat) --> (lat,lon)
    lon_r = ncread(fn,'lon'); lon_r = lon_r'; % longitude (deg)
    lat_r = ncread(fn,'lat'); lat_r = lat_r'; % latitude (deg) 
    b1_r = ncread(fn,'B1'); b1_r = b1_r'; % Band 1 Reflectance (0-1)
    b2_r = ncread(fn,'B2'); b2_r = b2_r'; % Band 2 Reflectance (0-1)
    b3_r = ncread(fn,'B3'); b3_r = b3_r'; % Band 3 Reflectance (0-1)
    b4_r = ncread(fn,'B4'); b4_r = b4_r'; % Band 4 Reflectance (0-1)
    b5_r = ncread(fn,'B5'); b5_r = b5_r'; % Band 5 Reflectance (0-1)
    b6_r = ncread(fn,'B6'); b6_r = b6_r'; % Band 6 Reflectance (0-1)
    b7_r = ncread(fn,'B7'); b7_r = b7_r'; % Band 7 Reflectance (0-1)
    b8_r = ncread(fn,'B8'); b8_r = b8_r'; % Band 8 Reflectance (0-1)
    b8A_r = ncread(fn,'B8A'); b8A_r = b8A_r';
    b9_r = ncread(fn,'B9'); b9_r = b9_r'; % Band 9 Reflectance (0-1)
    b11_r = ncread(fn,'B11'); b11_r = b11_r';
    b12_r = ncread(fn,'B12'); b12_r = b12_r';
    %cloud_prob_r = ncread(fn,'quality_cloud_confidence'); cloud_prob_r = cloud_prob_r'; % 0~100
    %classifi_r = ncread(fn,'quality_scene_classification'); classifi_r = classifi_r';
    
    %자동으로 위경도 추출
    lon_min = min(min(lon_r));
    lon_max = max(max(lon_r));
    lat_min = min(min(lat_r));
    lat_max = max(max(lat_r));  
    lat_mid = (lat_min+lat_max)/2;


    interv = 10/(2*pi*6400*1000*cos(lat_mid*pi/180)/360);
    lon_str = lon_min - 5*interv;
    lon_end = lon_max + 5*interv;
    lat_str = lat_min - 5*interv;
    lat_end = lat_max + 5*interv;

    % data cropping
    [loc_lat,loc_lon] = find(lon_r >= lon_str & lon_r <= lon_end & lat_r >= lat_str & lat_r <= lat_end); % find the location 
    loc_lon_1 = min(loc_lon(:)); loc_lon_2 = max(loc_lon(:));
    loc_lat_1 = min(loc_lat(:)); loc_lat_2 = max(loc_lat(:));

    lon_crop = lon_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    lat_crop = lat_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2); 

    [xx,yy] = meshgrid(lon_min:interv:lon_max,lat_min:interv:lat_max);

    % find nearest location 
    idx_loc = knnsearch([lon_crop(:) lat_crop(:)], [xx(:) yy(:)]); 

    % band data mapping
    B1 = b1_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B1 = reshape(B1(idx_loc), size(xx)); 

    B2 = b2_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B2 = reshape(B2(idx_loc), size(xx)); 

    B3 = b3_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B3 = reshape(B3(idx_loc), size(xx)); 

    B4 = b4_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B4 = reshape(B4(idx_loc), size(xx)); 

    B5 = b5_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B5 = reshape(B5(idx_loc), size(xx)); 

    B6 = b6_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B6 = reshape(B6(idx_loc), size(xx)); 

    B7 = b7_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B7 = reshape(B7(idx_loc), size(xx)); 

    B8 = b8_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B8 = reshape(B8(idx_loc), size(xx)); 

    B8A = b8A_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B8A = reshape(B8A(idx_loc), size(xx));

    B9 = b9_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B9 = reshape(B9(idx_loc), size(xx)); 

    B11 = b11_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B11 = reshape(B11(idx_loc), size(xx)); 

    B12 = b12_r(loc_lat_1:loc_lat_2, loc_lon_1:loc_lon_2);
    B12 = reshape(B12(idx_loc), size(xx)); 
    

    % change of variable name
    lon_ref = xx;
    lat_ref = yy;

    fn_save = fn(1:end-3);
    save(fn_save,'-v7.3','xx', 'yy','B*') 
%end
