// repeated subexpressions in a polynomial and a dead temporary
x = 0; acc = 0; n = 1500;
while (x < n) {
  p = (x + 3) * (x + 3) + (x + 3) * 2 + 1;
  q = (x + 3) * (x + 3) - 5;
  unused = p * q * 17;    // dead
  acc = acc + p - q;
  x = x + 1;
}
print acc;
