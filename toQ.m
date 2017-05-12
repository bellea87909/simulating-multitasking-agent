function f = toQ(x)
temp = 0;
if rem(x,3) == 0
    temp = 3;
else
    temp = rem(x,3);
end
f = temp;