function PlotTwoClasses(data, cls, figName)
figure;
hold on;
A = data(:, cls==0);
B = data(:, cls==1);
plot(A(1,:), A(2,:), 'b.');
plot(B(1,:), B(2,:), 'r.');
grid on;
xlabel('x-lavel'); ylabel('y-value');
title(figName);
legend('Class 1', 'Class 2', 'Location', 'best');
end
