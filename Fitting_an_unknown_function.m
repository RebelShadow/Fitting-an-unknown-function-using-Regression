%% definitions
% Load the data
load("proj_fit_10.mat");

% Define the degree of the polynomial (random at first)
m = 16;

% Calculate the total number of terms in the polynomial
nrterms = (m + 1) * (m + 2) / 2;

% Extract data from the 'id' dataset
x1_id = id.X{1, 1};
x2_id = id.X{2, 1};
y_id = id.Y;
id_dim = id.dims;

% Extract data from the 'val' dataset
x1_val = val.X{1, 1};
x2_val = val.X{2, 1};
y_val = val.Y;
val_dim = val.dims;

%% Identification

% Plot the original function in the 'id' dataset
figure;
subplot(2, 1, 1);
surf(x1_id, x2_id, y_id);
title("Identification Data","FontSize",24);
xlabel('X1',"FontSize",20);
ylabel('X2',"FontSize",20);
zlabel('Y',"FontSize",20);


% Initialize the PHI matrix with ones for identification
PHI = ones(id_dim(1) * id_dim(2), nrterms);

% Loop through data points to construct the polynomial basis functions for identification
for i = 1:id_dim(1) % construct 41 of j iteration to do the 41x41 rows
    for j = 1:id_dim(2) % construct first 41 rows in one repetitive structure
        k = 2; % because the first element is 1, due to initialization, k is the index that tells to go to the right
        for q = 1:m % Loops to construct powers from 1 to m
            for p = 0:q
                PHI((i - 1) * id_dim + j, k) = x1_id(i)^(q - p) * x2_id(j)^p;
                k = k + 1;
                % i go down 41 rows
                % j go down 1 row
                % k goes to right 1 column
                % q-p power of x1
                % p power of x2
            end
        end
    end
end

% Calculate the coefficients theta for identification dividing PHI matrix
% by left division by Y reshaped into a column vector
theta = PHI \ y_id(:);

% Calculate the aproximated values for identification in a column vector
ghat_id = PHI * theta;

% reshape ghat into a matrix, because due to the properties of matrix
% multiplication ghat results in a column vector of dim(1)*dim(2) X 1
ghat_id = reshape(ghat_id, id_dim(1), id_dim(2));

% Plot the approximated function for identification
subplot(2, 1, 2);
surf(x1_id, x2_id, ghat_id);
title("Aproximated Function for Identification Data","FontSize",24);
xlabel('X1',"FontSize",20);
ylabel('X2',"FontSize",20);
zlabel('Y Aproximated',"FontSize",20);

%% Validation

% Plot the original function in the 'val' dataset
figure;
subplot(2, 1, 1);
surf(x1_val, x2_val, y_val);
title("Validation Data","FontSize",24);
xlabel('X1',"FontSize",20);
ylabel('X2',"FontSize",20);
zlabel('Y',"FontSize",20);

% Initialize the PHI matrix with ones
PHI_val = ones(val_dim(1) * val_dim(2), nrterms);

% Loop through data points to construct the polynomial basis functions for validation
for i = 1:val_dim(1) % construct 71 of j iteration to do the 71x71 rows
    for j = 1:val_dim(2) % construct first 71 rows in one repetitive structure
        k = 2; % because the first element is 1 k is the index that tells to go to the right
        for q = 1:m % Loops to construct powers from 1 to m
            for p = 0:q
                PHI_val((i - 1) * val_dim(1) + j, k) = x1_val(i)^(q - p) * x2_val(j)^p;
                k = k + 1;
                % i go down 71 rows
                % j go down 1 row
                % k goes to right 1 column
                % q-p power of x1
                % p power of x2
            end
        end
    end
end

% Calculate the aproximated values for validation in a column vector
% We use theta, identified in the identification dataset
ghat_val = PHI_val * theta;

% Reshape ghat into a matrix, because due to the properties of matrix
% multiplication ghat results in a column vector of dim(1)*dim(2) X 1
ghat_val = reshape(ghat_val, val_dim(1), val_dim(2));

% Calculate MSE
true_mse = mean((y_val(:) - ghat_val(:)).^2);

% Plot the approximated function for validation
subplot(2, 1, 2);
surf(x1_val, x2_val, ghat_val);
title("Aproximated Function for Validation Data","FontSize",24);
xlabel('X1',"FontSize",20);
ylabel('X2',"FontSize",20);
zlabel('Y Aproximated',"FontSize",20);

%% MSE used to find the optimal value for the polynomial degree

% Define the maximum rank to calculate the MSE to
m = 30;

% Initialize an array to store MSE values for different polynomial ranks
% for identification and validation
mse_id = zeros(1, m);
mse_val = zeros(1, m);

for l = 1:m %% compute all aproximations for rank 1 to n

    % Calculate the total number of terms in the polynomial
    nrterms = (l + 1) * (l + 2) / 2;

    % Initialize the PHI matrix with ones for identification
    PHI = ones(id_dim(1) * id_dim(2), nrterms);

    % Loop through data points to construct the polynomial basis functions for identification
    for i = 1:id_dim(1) % construct 41 of j iteration to do the 41x41 rows
        for j = 1:id_dim(2) % construct first 41 rows in one repetitive structure
            k = 2; % because the first element is 1 k is the index that tells to go to the right
            for q = 1:l % Loops to construct powers from 1 to l
                for p = 0:q
                    PHI((i - 1) * id_dim + j, k) = x1_id(i)^(q - p) * x2_id(j)^p;
                    k = k + 1;
                    % i go down 41 rows
                    % j go down 1 row
                    % k goes to right 1 column
                    % q-p power of x1
                    % p power of x2
                end
            end
        end
    end

    % Calculate the coefficients theta for identification dividing PHI matrix
    % by left division by Y reshaped into a column vector
    theta = PHI \ y_id(:);

    % Calculate the aproximated values for identification in a column vector
    ghat_id = PHI * theta;

    % reshape ghat into a matrix, because due to the properties of matrix
    % multiplication ghat results in a column vector of dim(1)*dim(2) X 1
    ghat_id = reshape(ghat_id, id_dim(1), id_dim(2));

    %validation

    % Initialize the PHI matrix with ones
    PHI_val = ones(val_dim(1) * val_dim(2), nrterms);

    % Loop through data points to construct the polynomial basis functions for validation
    for i = 1:val_dim(1) % construct 71 of j iteration to do the 71x71 rows
        for j = 1:val_dim(2) % construct first 71 rows in one repetitive structure
            k = 2; % because the first element is 1 k is the index that tells to go to the right
            for q = 1:l % Loops to construct powers from 1 to l
                for p = 0:q
                    PHI_val((i - 1) * val_dim(1) + j, k) = x1_val(i)^(q - p) * x2_val(j)^p;
                    k = k + 1;
                    % i go down 71 rows
                    % j go down 1 row
                    % k goes to right 1 column
                    % q-p power of x1
                    % p power of x2
                end
            end
        end
    end

    % Calculate the aproximated values for validation in a column vector
    % We use theta, identified in the identification dataset
    ghat_val = PHI_val * theta;

    % Reshape ghat into a matrix, because due to the properties of matrix
    % multiplication ghat results in a column vector of dim(1)*dim(2) X 1
    ghat_val = reshape(ghat_val, val_dim(1), val_dim(2));

    % Calculate the MSE and store it into a vector
    mse_id(l) = mean((y_id(:) - ghat_id(:)).^2);
    mse_val(l) = mean((y_val(:) - ghat_val(:)).^2);
end

%% plot MSE
figure;
plot(mse_id);
hold;
plot(mse_val);
[min_mse_val, i_mse_val]=min(mse_val);

plot(i_mse_val, min_mse_val,'r*');
set(gca,'xtick',linspace(1,30,30))
hold;

title("MSE values depending on the polynomial degree");
legend('MSE for Identification data','MSE for Validation data',['The optimal degree of the polynomial = '  num2str(i_mse_val)]);grid;
ylabel('MSE value');xlabel('Polinomial degree');