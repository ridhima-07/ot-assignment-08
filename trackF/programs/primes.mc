// trial-division prime count; debug flag is constant false
debug = 0; limit = 400; count = 0; n = 2;
while (n < limit) {
  d = 2; isp = 1;
  while (d * d <= n) {
    if (n % d == 0) { isp = 0; }
    d = d + 1;
  }
  if (debug) { print n * 1000 + d; }
  count = count + isp;
  n = n + 1;
}
print count;
