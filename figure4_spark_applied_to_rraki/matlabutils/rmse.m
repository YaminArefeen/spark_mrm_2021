function out = rmse(groundtruth,comparison)
out = norm(groundtruth(:) - comparison(:))/norm(groundtruth(:));
end
