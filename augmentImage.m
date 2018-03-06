  function [im_aug,rot_angle,scale_ratio, tran_out] = augmentImage(im_in,varargin)
%%  Rotate, scale, and translate and crop to 128 sizes...
%    By default, everything is random (you can just use the default setting)
%      Possible parameters are:
%        rot_angle - rotation angle
%        scale     - the dimension of the output image (Eg. 256 for a scale
%                    ratio of 1
%        translation - the 2 element vector with the tranlation of the
%               height and width in the original image
%        min_scale: smallest size in the original image for the output image (128 = .5 of input of 256)
%      
%        A value of -1 means use a random values (default)
%   This function ensures the entire image is the wallpaper pattern (thus
%       max_scale is implicitly calculated depending on rotation)
%  
%   The call of this function for all random parameters is is 
%           [im_aug,rot_angle,scale_ratio, translation] = augmentImage(im_in); 
%              im_aug is the augmented output image
%              rot_angle is the rotation angle used here
%              scale ratio is the scale from the original image
%              translation is the translation in the height and width of the original image

% Parse optional parameters 
p = inputParser;
addParameter(p,'rot_angle',-1);
addParameter(p,'scale',-1);
addParameter(p,'translation',[-1,-1]);
addParameter(p,'min_scale',128);
parse(p,varargin{:});
rot_angle = p.Results.rot_angle;
scale = p.Results.scale;
tran = p.Results.translation;
min_scale = p.Results.min_scale;



[h, w, ~] = size(im_in);

% Rotate, Scale and Translate in the same affine rotation.
%     So translate and crop between buffer of dim and the random scaling, rotate, and random scaling (all at once!)

if rot_angle < 0
    rot_angle = randi(360);
end

[mH, mW] = calculateLargestProportionalRect(rot_angle, h,w);
dim = floor(min(mH,mW)); % find maximum size square for scaling
if scale < 0
    scale = rand() * (dim-min_scale)+min_scale; % get random scale
elseif scale == 0
    scale = 181; %  Max allowed if allowing for any rotation
end

scale_ratio = 128.0/scale;

% Rotate and Scale image
im_aug = imrotate(im_in,rot_angle,'bilinear','loose');
im_aug = imresize(im_aug,scale_ratio,'bilinear','Antialiasing',false);

% Find max translation here and scale it down and make random
rangeH = mH-scale;
rangeW = mW-scale;
if all(tran < 0)
    hshift = scale_ratio*(rand()*rangeH - rangeH/2);
    wshift = scale_ratio*(rand()*rangeW - rangeW/2);
else
    hshift = scale_ratio*tran(1);
    wshift = scale_ratio*tran(2);
end
% Save translation for out
tran_out = [hshift, wshift];
% Now crop and translate.  
[augH, augW,~] = size(im_aug);
rangeH = round(augH/2-63+hshift);
rangeH = rangeH:rangeH+127;
rangeW = round(augW/2-63+wshift);
rangeW = rangeW:rangeW+127;

im_aug = im_aug(rangeH, rangeW);

% fprintf('Dim: %d\tScale: %.02f\tRot: %03d\thshift: %.01f\twshift: %.01f\n',dim,scale_ratio,rot_angle,hshift,wshift)



function [h,w] = calculateLargestProportionalRect(angle, origHeight,origWidth)
angle = deg2rad(angle);
if (origWidth <= origHeight)
    w0 = origWidth;
    h0 = origHeight;
    
else
    w0 = origHeight;
    h0 = origWidth;
end

% Angle normalization in range [-PI..PI)
ang = angle - floor((angle + pi) / (2*pi)) * 2*pi;
ang = abs(ang);
if (ang > pi / 2)
    ang = pi - ang;
end
c = w0 / (h0 * sin(ang) + w0 * cos(ang));
if (origWidth <= origHeight)
    w = w0 * c;
    h = h0 * c;
else
    w = h0 * c;
    h = w0 * c;
end



% Python Implementation

%     #  Rotate, scale, and translate and crop to 128 sizes...
%     #   Rotate is angle of rotation degrees
%     #   Scale is the scale of the image in pixels between [w*cos(45 degrees),w]
%     #   Translation is the X and Y translation of the image in pixels of the original image
%
%
%     h, w = im.shape
%     min_size = 128; # smallest size in the original image for the output image
%     # Rotate, Scale and Translate in the same affine rotation.
%     #     So translate and crop between buffer of dim and the random scaling, rotate, and random scaling (all at once!)
%
%     if rot_angle < 0:
%         rot_angle = randrange(360)
%
%
%     #print(rot_angle)
%     mH, mW = calculateLargestProportionalRect(rot_angle, h,w)
%     dim = np.floor(min(mH,mW))
%     if scale < 0:
%         scale = randrange(min_size,dim)
%     elif scale == 0:
%         scale = 181 # Max allowed if allowing for any rotation
%
%     # Get rotation and scale matrix
%     scale_ratio = 128.0/scale
%     M = cv2.getRotationMatrix2D((h/2,w/2),rot_angle,scale_ratio)#randrange(360),1)
%
%      # Find max difference and scale it down and make random
%
%     if tran < 0:
%         xshift = scale_ratio*(randint(-(dim-scale),dim-scale)/2.0)
%         yshift = scale_ratio*(randint(-(dim-scale),dim-scale)/2.0)
%     else:
%         xshift = scale_ratio*tran
%         yshift = scale_ratio*tran
%     #(random.randrange(1,3)*2-3)#
%     #xshift = (128.0/rand_scale)*((dim-rand_scale)/2.0)
%     #yshift = (128.0/rand_scale)*((dim-rand_scale)/2.0)
%     M[0,2] = M[0,2] - (h/2 - 64) - xshift # shift center and add random translation in x
%     M[1,2] = M[1,2] - (w/2 - 64) - yshift # shift center and add random translation in y
%
%     #print('Dim: {}\tScale: {}\tRot: {:03d}\txshift: {}\tyshift: {}'.format(dim,scale,rot_angle,xshift,yshift))
%
%     # Apply Transformation in only valid area and use inter_area since downsampling (CV suggestion)
%     im = cv2.warpAffine(im,M,(128,128),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


% def calculateLargestProportionalRect(angle, origHeight,origWidth):
%     angle = np.deg2rad(angle)
%     if (origWidth <= origHeight):
%         w0 = origWidth
%         h0 = origHeight
%
%     else:
%         w0 = origHeight
%         h0 = origWidth
%
%     # Angle normalization in range [-PI..PI)
%     ang = angle - np.floor((angle + np.pi) / (2*np.pi)) * 2*np.pi;
%     ang = np.abs(ang);
%     if (ang > np.pi / 2):
%         ang = np.pi - ang;
%     c = w0 / (h0 * np.sin(ang) + w0 * np.cos(ang));
%     if (origWidth <= origHeight) :
%         w = w0 * c
%         h = h0 * c
%
%     else:
%         w = h0 * c
%         h = w0 * c
%
%     return h,w
