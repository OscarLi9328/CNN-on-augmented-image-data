clc
close all
clear

load('aug.mat');

%%%%%% plot the statistics of the augmented images %%%%%%

for i = 1:5
    figure(i);
    subplot(1, 3, 1); 
    histogram(rot_angle(:,i));
    title('angle of rotation');
    xlabel('angle');
    ylabel('# of images');
    subplot(1, 3, 2)
    histogram(scale(:,i));
    title('ratio of scaling');
    xlabel('ratio');
    ylabel('# of images');
    subplot(1, 3, 3)
    histogram([translation{:,i}]);
    title('pixels of translation');
    xlabel('pixels');
    ylabel('# of images');
    axes( 'Position', [0, 0.96, 1, 0] );
    set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
    text( 0.5, 0, ['augmentation ' num2str(i)], 'FontSize', 14', 'FontWeight', 'Bold', ...
      'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
    saveas(gcf, ['image' num2str(i)], 'png');
end

%%%%%%% calculate the mean and the std of the images %%%%%%%%

mean_rot_angle = zeros(5, 1);
mean_scale = zeros(5, 1);
mean_trans = zeros(5, 1);
std_rot_angle = zeros(5, 1);
std_scale = zeros(5, 1);
std_trans = zeros(5, 1);


for i = 1 : 5
    mean_rot_angle(i) = mean(rot_angle(:,i));
    std_rot_angle(i) = std(rot_angle(:,i));
    mean_scale(i) = mean(scale(:,i));
    std_scale(i) = std(scale(:,i));
    mean_trans(i) = mean([translation{:,i}]);
    std_trans(i) = std([translation{:,i}]);   
end
save('stat.mat', 'mean_rot_angle', 'mean_scale', 'mean_trans', 'std_rot_angle', 'std_scale', 'std_trans');
