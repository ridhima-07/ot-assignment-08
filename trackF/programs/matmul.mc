// 3 nested loops over an implicit 20x20 matrix; index arithmetic i*N+j recomputed
N = 20; i = 0; tr = 0;
while (i < N) {
  j = 0;
  while (j < N) {
    k = 0; c = 0;
    while (k < N) {
      aik = (i * N + k) % 7 + 1;
      bkj = (k * N + j) % 5 + 1;
      c = c + aik * bkj;
      k = k + 1;
    }
    if (i == j) { tr = tr + c; }
    j = j + 1;
  }
  i = i + 1;
}
print tr;
