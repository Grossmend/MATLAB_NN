
n = 60000;
b = target;

a = zeros(n,10);
for i = 1:n
    for j = 1:10
        if b(i) == j
            a(i,j) = 1;
        else
            a(i,j) = 0;
        end
    end
end

clearvars i j n