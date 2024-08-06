clear; close all;

f1 = @(x) (x + 5).^2 - cos(5 * (x + 5));
f2 = @(x) (x - 5).^2 - cos(5 * (x + 5));

num_points = 1e3;
xi = linspace(-2, 2, num_points)';
y1 = f1(xi);
y2 = f2(xi);



% figure; hold on;
% plot (y1, y2);

% Identify nondominated points
sorted_matrix = sortrows([y1, y2], 'ascend');

nondominated_points = sorted_matrix(1, :);
last_pareto_point = sorted_matrix(1, :);

for point_index = 2:(num_points - 1)

    current_point = sorted_matrix(point_index, :);

    if last_pareto_point(2) > current_point(2)
        nondominated_points = [nondominated_points; current_point];
        last_pareto_point = current_point;
    end
end


    % # Traverse the sorted array to figure out Pareto-optimal points
    % for i in range(points.shape[0]):
    % 
    %     # Add this point to the Pareto front if it isn't dominated by the last Pareto-optimal point
    %     # --------
    %     # TODO: You code here.
    %     if pareto_y > points[i, 1]:       # <--
    %         pareto_indices.append(i)
    % 
    %         # Update the last Pareto-optimal point using this point
    %         # --------
    %         # TODO: Your code here.
    %         pareto_x = points[i, 0]
    %         pareto_y = points[i, 1]
    % 
    % # Return the Pareto front
    % pareto_front = points[pareto_indices]

figure;
hold on;
plot (xi, y1, 'LineWidth', 1.5);
plot (xi, y2, 'LineWidth', 1.5);
xlabel ('x');
legend ('f1', 'f2', 'Location', 'East');

figure;
hold on;
plot (y1, y2, 'LineWidth', 1.5, 'Color', 0.5*ones(1,3));;
% plot (nondominated_points(:, 1), nondominated_points(:, 2), '.', 'MarkerSize', 20)
xlabel ('f1');
ylabel ('f2');

