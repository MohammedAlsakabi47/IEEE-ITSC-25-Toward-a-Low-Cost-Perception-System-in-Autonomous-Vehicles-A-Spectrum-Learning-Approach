close all; clear all; clc
%% Input Parameters
% Define folder paths for camera image semantic segmentations and radar depth maps
FolderPath_camera = 'Example Frames/camera scemantic segmentations/';
FolderPath_radar = 'Example Frames/radar depth maps/';

% Algorithm parameters
M_camera = 200;
M_radar = 20;
theta = -70:1:70;
phi = -70:1:70;

%% Algorithm
nFiles = 10; % number of files
for n = 1:nFiles
    load(strcat(FolderPath_radar, num2str(n), '.mat')) % load radar depth map
    load(strcat(FolderPath_camera, num2str(n), '.mat')) % load camera image

    % Estimate spectrums of radar and camera
    P_radar = estimate(depth_map, M_radar, theta, phi);
    P_camera = estimate(double(Semantic_Segmentations), M_camera, theta, phi);

    % take log for compression and noramlize
    % radar
    P_radar = imresize(P_radar, [256 256], 'nearest');
    P_radar = log10(P_radar);
    P_radar = 100*(P_radar + abs(min(min(P_radar))));
    P_radar = P_radar/max(max(P_radar));
    
    % camera
    P_camera = imresize(P_camera, [256 256], 'nearest');
    P_camera = real(log10(P_camera));
    P_camera = P_camera + abs(min(min(P_camera)));
    P_camera = P_camera/max(max(P_camera));
    
    % save ground truth and training data
    imwrite(P_camera.*P_radar, strcat('Example Frames/camera spetrum/',num2str(n),'.jpg'));
    imwrite(P_radar, strcat('Example Frames/radar depth map spectrum/',num2str(n),'.jpg'));

end

%% Example Output
figure
imagesc(P_radar)
figure
imagesc(P_camera)

%% Supporting Functions
function P = estimate(image, M, theta, phi)
    % Preliminary computations
    m = (1:M)';
    theta = theta*(pi/180); % convert to radian
    phi = phi*(pi/180); % convert to radian
    image = imresize(image, [length(phi) length(theta)]); %resize image for compatible positional encoding
    
    % Compute basis and covariance matrix
    basis_phi = exp(-j*pi*m*sin(phi));
    basis_theta = exp(-j*pi*m*sin(theta));
    C_phi = basis_phi'*basis_phi; % covariance matrix of phi
    C_theta = basis_theta'*basis_theta; % covariance matrix of theta

    % Estimate 2D spectrum
    P = zeros(length(phi),length(theta)); % initialize P
    for ii = 1:size(P,2) % columns
        for jj = 1:size(P,1) % rows
            elevation = transpose(C_theta(ii, :));
            azimuth = C_phi(jj, :);
            P(jj,ii) = sum(abs(image.*(transpose(elevation*azimuth))), 'all');
        end
    end
end


