// loop-invariant expressions and constants recomputed every iteration
n = 2000; a = 7; b = 3; i = 0; s = 0;
while (i < n) {
  k = a * b + 4;          // invariant
  s = s + i * 8 + k;      // i*8 -> shift
  i = i + 1;
}
print s;
