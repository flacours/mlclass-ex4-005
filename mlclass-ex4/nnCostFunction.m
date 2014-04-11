function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%fprintf("num_labels       = %d\n", num_labels);
%fprintf("input_layer_size = %d\n", input_layer_size);
%fprintf("hidden_layer_size= %d\n", hidden_layer_size);
%disp(size(Theta1));
%disp(Theta1);

%disp(size(Theta2));
%disp(Theta2);

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
J = 0;

% add 1
a1 = [ones(m, 1) X];
z2 = Theta1 * a1';
a2 = sigmoid(z2)';
m2 = size(a2,1);
a2 = [ones(m2, 1) a2];
a3 = sigmoid(Theta2 * a2')';
for(i = 1 : m)
   % need to recode the output
   yrecoded = zeros(num_labels,1);
   yrecoded(y(i)) = 1;
   for(k = 1 : num_labels);
     a = ( -yrecoded(k) * log(a3(i,k)) ) - ( (1-yrecoded(k)) * log (1-a3(i,k)) );
     J += a;
   end;
end
J /= m;

% Regularized cost function
%disp('theta1');

t1reg = 0;
%disp('size t1');
%disp(hidden_layer_size);
%disp(input_layer_size);
for(j = 1 : hidden_layer_size)
   for(k = 1 : input_layer_size);
      t1reg += Theta1(j,k+1)^2;
   end;
end;

%disp('theta2');

t2reg = 0;
for(j = 1 : num_labels)
   for(k = 1 : hidden_layer_size);
      t2reg += Theta2(j,k+1)^2;
   end;
end;

J += ( lambda / (2 * m)) * (t1reg + t2reg);


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%size(a1)
%size(a2)
%size(a3)
%fprintf("m=%d\n", m);
emptycol = zeros(num_labels,1);
%disp('size empty col');
%size(emptycol)
for(i = 1 : m) %
% need to recode the output
   yrecoded = zeros(num_labels,1);
   yrecoded(y(i)) = 1;

% 1 set input values and compute activation

    a_1 = a1(i, :);
    a_2 = a2(i, :);
    a_3 = a3(i, :);

% 2 compute delta_3
    delta_3 = a_3' - yrecoded;

% 3 compute delta_2
    sg = sigmoidGradient(a_2)';
%    disp('size sg');
%    disp(size(sg));
    delta_2 = (Theta2' * delta_3) .* sg;


% 4 accumulate the gradient
    delta_2 = delta_2(2:end);
    a_1 = a_1(:, 2:end);
    a_2 = a_2(:, 2:end);
%    disp('size delta_2');
%    size(delta_2)
%    disp('a_1');
%    size(a_1)
%    disp('size delta_3');
%    size(  delta_3)
%    disp('a_2');
%    size(a_2)
    d2 = delta_2 * a_1;
%    disp('size d2');
%    size(d2)
    d3 = delta_3 * a_2;
%    disp('size d3');
    Theta1_grad += [emptycol d2];
    Theta2_grad += [emptycol d3];
end;
% 5 obtain the unregularized gradient
    Theta1_grad /= m;
    Theta2_grad /= m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
