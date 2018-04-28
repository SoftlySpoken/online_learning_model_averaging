function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');  % column vector

assert(size(labels,1) == numLabels, 'Mismatch in label count');

% Convert label 0 to 10 to be compatible with Octave's 1-indexing
for iter = 1 : numLabels,
	if labels(iter) == 0,
		labels(iter) = 10;
	end;
end;

fclose(fp);

end
