% This function initializes a gray-level image with Shepard's 2D interpolation.
% Author: Suhas Sreehari, Purdue University

function x = shepard_initialize(image, measurement_mask, window, p)

%% Initializations

if nargin == 1
    if strcmp(class(image), 'uint8') == 1
        bit_depth = 8;
    elseif strcmp(class(image), 'uint16') == 1
        bit_depth = 16;
    elseif strcmp(class(image), 'uint32') == 1
        bit_depth = 32;
    end
    max_value = (2^bit_depth)-1;
    measurement_mask = uint16(image*max_value)/(max_value-1);
    window = 5;
    p = 2;
elseif nargin == 2
    window = 5;
    p = 2;
elseif nargin == 3
    p = 2;
end
    
wing = floor(window/2); % Length of each "wing" of the window.

[h, w] = size(image);

x = image; % ML initialization

%% Interpolation


for i = 1:h
    i_lower_limit = -min(wing, i-1);
    i_upper_limit = min(wing, h-i);
    for j = 1:w
       if measurement_mask(i, j) == 0 % checking if there's a need to interpolate
           j_lower_limit = -min(wing, j-1);
           j_upper_limit = min(wing, w-j);
           count = 0; % keeps track of how many measured pixels are withing the neighborhood
           sum_IPD = 0;
           interpolated_value = 0;
           
           for neighborhood_i = i+i_lower_limit:i+i_upper_limit
               for neighborhood_j = j+j_lower_limit:j+j_upper_limit
                  if measurement_mask(neighborhood_i, neighborhood_j) == 1
                      count = count + 1;
                      % IPD: "inverse pth-power distance".
                      IPD(count) = double(1/((neighborhood_i - i)^p + (neighborhood_j - j)^p));
                      sum_IPD = double(sum_IPD + IPD(count));
                      pixel(count) = image(neighborhood_i, neighborhood_j);
                  end
               end
           end
           
           for c = 1:count
               weight = IPD(c)/sum_IPD;
               interpolated_value = double(interpolated_value + weight*pixel(c));
           end
           x(i, j) = interpolated_value;
       end
    end
end

%% Plotting the images
% 
% figure(1)
% 
% subplot(1, 2, 1)
% imagesc(image);axis image;colormap(gray);
% title('Original image');
% 
% subplot(1, 2, 2)
% imagesc(x);axis image;colormap(gray);
% title('Shepard-interpolated image');

end




