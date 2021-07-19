function imshow3(img,range,shape)
% imshow3(img, [ range, [shape )
%
% function to display series of images as a montage.
%
% img - a 3D array representing a series of images
% range - window level (similarly to imshow
% shape - a 2x1 vector representing the shape of the motage
%
% Example:
% 		im = repmat(phantom(128),[1,1,6]);
%		figure;
%		imshow3(im,[],[2,3]);
%
% (c) Michael Lustig 2012






img = img(:,:,:);
[sx,sy,nc] = size(img);

% for i = 1:nc
%     img(:,:,i) = img(:,:,i) / max(vec(abs(img(:,:,i))));
% end

if nargin < 2
    range = [min(img(:)), max(img(:))];
end

if isempty(range)==1
    range = [min(img(:)), max(img(:))];
end

if nargin < 3
    
    shape = mosaic_dims( nc );
    if (nc ~= prod(shape))
        nc = prod(shape);
        img(end,end,nc) = 0;
    end
    
end
img = reshape(img,sx,sy*nc);
img = permute(img,[2,3,1]);
img = reshape(img,sy*shape(2),shape(1),sx);
img = permute(img,[3,2,1]);
img = reshape(img,sx*shape(1),sy*shape(2));

% Resize

iSize = size(img);
sSize = get(0,'Screensize');
sSize = floor([sSize(4),sSize(3)]) * 0.67;
minScale = min( sSize ./ iSize );

rSize = [iSize(1)*minScale,iSize(2)*minScale];

img = imresize(img,rSize,'Method','nearest');


imshow(img,range);
end

% From Martin Uecker
function shape = mosaic_dims( nc )
shape(1) = floor(sqrt(nc));
shape(2) = floor(nc / shape(1));

while (nc > shape(1) * shape(2))
    shape(2) = shape(2) + 1;
end

if ((shape(1) - 1) * (shape(2) + 1) == nc)
    
    shape(1) = shape(1) - 1;
    shape(2) = shape(2) + 1;
end
end
