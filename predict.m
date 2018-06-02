function p = predict(theta1,theta2,theta3, X)
%PREDICT Predict whether the label is 0 or 1 or 2 using learned logistic 
%regression parameters theta


m = size(X, 1); % Number of training examples
H1 = zeros(1,1);
H2 = zeros(1,1);
H3 = zeros(1,1);
p = 10*ones(1,1);

H1 = sigmoid(X * theta1);
H2 = sigmoid(X * theta2);
H3 = sigmoid(X * theta3);



for i = 1:m
    
    if (H1(i)>=H2(i))
        if(H1(i)>= H3(i))
            p = [p;0];
        end
    end
    if (H2(i)>=H1(i))
        if(H2(i)>=H3(i))
            p = [p;1];
        end

    end
    if (H3(i)>=H2(i))
        if(H3(i)>=H1(i))
            p = [p;2];
        end
    end
end

p = p(2:m+1);


end
