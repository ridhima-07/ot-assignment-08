// fully constant prefix + iterative Fibonacci mod a constant
m = 1000 * 1000 + 7; scale = 4 * 4; a = 0; b = 1; i = 0; n = 3000 + 0 * m;
while (i < n) {
  t = (a + b) % m;
  a = b; b = t;
  i = i + 1 * 1;
}
print a * scale / 16;
