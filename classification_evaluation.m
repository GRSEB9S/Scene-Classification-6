function classification_evaluation()  
%% initialization
    clear;
    caffe.reset_all();
    net_model =  '.\model\googlenet_places365.prototxt';
    net_weights = '.\model\googlenet_places365.caffemodel';  
    im_path = 'E:\dataset\Places2\val_large\';
    im_name_path='.\data\val_mosque.txt'
    evaluation_results='.\data\pred.txt';
    phase = 'test'; % run with phase test (so that dropout isn't applied)
    use_gpu=true;
    gpu_id=0;
    % Set caffe mode  
    if  use_gpu
        caffe.set_mode_gpu();  
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end
    if ~exist(net_weights, 'file')
        error('model is not exist');
    end
    % Initialize a network  
    net = caffe.Net(net_model, net_weights, phase);
%% load file name and label
    % Files = dir(im_path);
    % im_names = cell(1,length(Files)-2);
    % for j = 3:length(Files)
    %     im_names{j-2} = Files(j).name;
    % end
    [filelist, labels] = textread(im_name_path, '%s %d');
    fid=fopen(evaluation_results,'wt+');
%% evaluation
    for j = 1:length(filelist)
        im = imread([im_path, filelist{j}]);
        input_data = {prepare_image(im)};
        tic;
        scores = net.forward(input_data);
        toc;
        scores = scores{1};
        scores = mean(scores, 2);  % take average scores over 10 crops
        
        %sort
        [sort_scores sort_index]=sort(scores);
        sort_scores=flipud(sort_scores);
        sort_index=flipud(sort_index)-1;
        
        fprintf(fid,'%s %d %d %d %d %d\n',filelist{j},sort_index(1),sort_index(2),sort_index(3),sort_index(4),sort_index(5)); 
        % call caffe.reset_all() to reset caffe
        if j == length(filelist)
            caffe.reset_all();
        end
    end
end
% ------------------------------------------------------------------------  
function crops_data = prepare_image(im)  
    d = load('D:\CNN\caffe-master\matlab\+caffe\imagenet\ilsvrc_2012_mean.mat');
    mean_data = d.mean_data;
    IMAGE_DIM = 256;
    CROPPED_DIM = 224;
  
    % Convert an image returned by Matlab's imread to im_data in caffe's data
    % format: W x H x C with BGR channels
    im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
    im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)
  
    % oversample (4 corners, center, and their x-axis flips)
    crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
    indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
    n = 1;
    for i = indices
        for j = indices
            crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
            crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
            n = n + 1;
        end
    end
    center = floor(indices(2) / 2) + 1;
    crops_data(:,:,:,5) = ...
    im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
    crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);
end