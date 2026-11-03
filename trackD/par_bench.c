// Track D: real parallel micro-benchmarks with pthreads.
//   ./par_bench sched <threads> <strategy>   strategy: block | cyclic | central | steal
//   ./par_bench locality <n>                 ijk vs ikj vs blocked matrix multiply
//   ./par_bench sync <threads> <mode>        mode: mutex | atomic | local
// Build: gcc -O2 -pthread par_bench.c -o par_bench -lm
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// ------------------------------------------------------------ scheduling
#define NTASKS 2000
static long work[NTASKS];
static volatile double sink[64 * 8];

static double do_task(long iters) {          // CPU-bound, no memory traffic
  double x = 1.0;
  for (long i = 0; i < iters; i++) x = x * 1.0000001 + 1e-9;
  return x;
}

static void make_work(void) {               // heavy-tailed (lognormal) task sizes, deterministic
  uint64_t s = 42;
  for (int i = 0; i < NTASKS; i++) {
    double u1, u2;
    s = s * 6364136223846793005ULL + 1; u1 = ((s >> 11) + 1.0) / 9007199254740993.0;
    s = s * 6364136223846793005ULL + 1; u2 = ((s >> 11) + 1.0) / 9007199254740993.0;
    double z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    work[i] = (long)(20000 * exp(1.2 * z));
  }
  // put the heaviest tasks at the end so that a static block split is unbalanced (realistic: sorted input)
  for (int i = 0; i < NTASKS; i++) for (int j = i + 1; j < NTASKS; j++) if (work[j] < work[i]) { long t = work[i]; work[i] = work[j]; work[j] = t; }
}

typedef struct { int lo, hi; pthread_mutex_t m; int buf[NTASKS]; } deque_t;   // tasks buf[lo..hi)
static deque_t dq[64];
static int P; static char strategy[16];
static atomic_int next_task;
static double busy[64];
static long steals[64];

static int pop_bottom(int me) {
  int t = -1; pthread_mutex_lock(&dq[me].m);
  if (dq[me].hi > dq[me].lo) t = dq[me].buf[--dq[me].hi];
  pthread_mutex_unlock(&dq[me].m); return t;
}
static int steal_top(int victim) {
  int t = -1; pthread_mutex_lock(&dq[victim].m);
  if (dq[victim].hi > dq[victim].lo) t = dq[victim].buf[dq[victim].lo++];
  pthread_mutex_unlock(&dq[victim].m); return t;
}

static void *worker(void *arg) {
  int me = (int)(intptr_t)arg; double t0 = now(), acc = 0; unsigned seed = me * 7919 + 1;
  if (!strcmp(strategy, "block")) {
    int per = (NTASKS + P - 1) / P;
    for (int i = me * per; i < NTASKS && i < (me + 1) * per; i++) acc += do_task(work[i]);
  } else if (!strcmp(strategy, "cyclic")) {
    for (int i = me; i < NTASKS; i += P) acc += do_task(work[i]);
  } else if (!strcmp(strategy, "central")) {        // dynamic self-scheduling from one shared counter
    for (int i; (i = atomic_fetch_add(&next_task, 1)) < NTASKS;) acc += do_task(work[i]);
  } else {                                          // work stealing: start from block split, steal when empty
    int remaining_victims;
    for (;;) {
      int t = pop_bottom(me);
      if (t >= 0) { acc += do_task(work[t]); continue; }
      remaining_victims = 0;
      for (int k = 0; k < P && t < 0; k++) {
        int v = (me + 1 + (rand_r(&seed) % P)) % P; if (v == me) continue;
        t = steal_top(v);
        if (dq[v].hi > dq[v].lo) remaining_victims = 1;
      }
      if (t >= 0) { steals[me]++; acc += do_task(work[t]); continue; }
      int any = 0; for (int v = 0; v < P; v++) if (dq[v].hi > dq[v].lo) any = 1;
      if (!any && !remaining_victims) break;
    }
  }
  sink[me * 8] = acc; busy[me] = now() - t0; return NULL;
}

static int run_sched(int threads, const char *strat) {
  P = threads; strncpy(strategy, strat, 15); make_work(); atomic_store(&next_task, 0);
  int per = (NTASKS + P - 1) / P;
  for (int p = 0; p < P; p++) {
    pthread_mutex_init(&dq[p].m, NULL); dq[p].lo = dq[p].hi = 0; steals[p] = 0;
    for (int i = p * per; i < NTASKS && i < (p + 1) * per; i++) dq[p].buf[dq[p].hi++] = i;
  }
  pthread_t th[64]; double t0 = now();
  for (int p = 0; p < P; p++) pthread_create(&th[p], NULL, worker, (void *)(intptr_t)p);
  for (int p = 0; p < P; p++) pthread_join(th[p], NULL);
  double T = now() - t0, mx = 0, sum = 0; long st = 0;
  for (int p = 0; p < P; p++) { if (busy[p] > mx) mx = busy[p]; sum += busy[p]; st += steals[p]; }
  printf("{\"threads\":%d,\"strategy\":\"%s\",\"secs\":%.4f,\"imbalance\":%.3f,\"steals\":%ld}\n", P, strat, T, mx / (sum / P), st);
  return 0;
}

// ------------------------------------------------------------ locality
static int run_locality(int n) {
  double *A = malloc(sizeof(double) * n * n), *B = malloc(sizeof(double) * n * n), *C = calloc(n * n, sizeof(double));
  for (int i = 0; i < n * n; i++) { A[i] = (i % 7) * 0.5; B[i] = (i % 5) * 0.25; }
  double t, ref;
  t = now();
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { double s = 0; for (int k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j]; C[i * n + j] = s; }
  double t_ijk = now() - t; ref = C[(n / 2) * n + n / 3];
  memset(C, 0, sizeof(double) * n * n); t = now();
  for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) { double a = A[i * n + k]; for (int j = 0; j < n; j++) C[i * n + j] += a * B[k * n + j]; }
  double t_ikj = now() - t; double c2 = C[(n / 2) * n + n / 3];
  memset(C, 0, sizeof(double) * n * n); int bs = 64; t = now();
  for (int ii = 0; ii < n; ii += bs) for (int kk = 0; kk < n; kk += bs) for (int jj = 0; jj < n; jj += bs)
    for (int i = ii; i < ii + bs && i < n; i++) for (int k = kk; k < kk + bs && k < n; k++) {
      double a = A[i * n + k]; for (int j = jj; j < jj + bs && j < n; j++) C[i * n + j] += a * B[k * n + j]; }
  double t_blk = now() - t; double c3 = C[(n / 2) * n + n / 3];
  printf("{\"n\":%d,\"ijk_secs\":%.4f,\"ikj_secs\":%.4f,\"blocked64_secs\":%.4f,\"results_agree\":%s}\n", n, t_ijk, t_ikj, t_blk,
         (fabs(ref - c2) < 1e-6 && fabs(ref - c3) < 1e-6) ? "true" : "false");
  return 0;
}

// ------------------------------------------------------------ synchronisation
#define INCS 4000000L
static long counter; static atomic_long acounter; static pthread_mutex_t cm = PTHREAD_MUTEX_INITIALIZER;
static char smode[16]; static int ST;
static void *sync_worker(void *arg) {
  (void)arg; long per = INCS / ST;
  if (!strcmp(smode, "mutex")) for (long i = 0; i < per; i++) { pthread_mutex_lock(&cm); counter++; pthread_mutex_unlock(&cm); }
  else if (!strcmp(smode, "atomic")) for (long i = 0; i < per; i++) atomic_fetch_add(&acounter, 1);
  else { long local = 0; for (long i = 0; i < per; i++) { local++; __asm__ volatile("" : "+r"(local)); }  // stop the compiler folding the loop
         pthread_mutex_lock(&cm); counter += local; pthread_mutex_unlock(&cm); }
  return NULL;
}
static int run_sync(int threads, const char *mode) {
  ST = threads; strncpy(smode, mode, 15); counter = 0; atomic_store(&acounter, 0);
  pthread_t th[64]; double t0 = now();
  for (int p = 0; p < ST; p++) pthread_create(&th[p], NULL, sync_worker, NULL);
  for (int p = 0; p < ST; p++) pthread_join(th[p], NULL);
  double T = now() - t0; long total = !strcmp(mode, "atomic") ? atomic_load(&acounter) : counter;
  printf("{\"threads\":%d,\"mode\":\"%s\",\"secs\":%.4f,\"correct\":%s}\n", ST, mode, T, total == (INCS / ST) * ST ? "true" : "false");
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 4 && !strcmp(argv[1], "sched")) return run_sched(atoi(argv[2]), argv[3]);
  if (argc >= 3 && !strcmp(argv[1], "locality")) return run_locality(atoi(argv[2]));
  if (argc >= 4 && !strcmp(argv[1], "sync")) return run_sync(atoi(argv[2]), argv[3]);
  fprintf(stderr, "usage: see header\n"); return 1;
}
