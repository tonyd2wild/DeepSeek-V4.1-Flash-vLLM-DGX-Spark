// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tech2wild
//
// dsv41_rowio: exact DeepSeek-V4.1 Engram row reads from safetensors shards
// (the full official shards, or our node-local sparse copies) for the SGLang
// lane. Original code.
//
// rowio_lookup() has the CUhostFn signature. engram.py enqueues it with
// cuLaunchHostFunc, so it runs in stream order, eagerly or as a host node of a
// captured CUDA graph, and it never calls CUDA. For every id it writes the
// 256-byte fp8 row and the 8-byte e8m0 scale row when this rank owns the id,
// and zeros otherwise (the TP all-reduce sums the ranks). A persistent thread
// pool issues the preads in parallel. A short read aborts the process: a
// forward must never continue with a missing row.
//
// Design note: running the row reads as a CUDA host function inside the graph
// follows the idea of 0xSero's MIT row_store.cpp; this file is an independent
// implementation (thread pool, sparse-copy support, no private row cache: the
// kernel page cache is reclaimable, which matters on unified-memory GB10).

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct Table {
  int fd_w = -1;
  int fd_s = -1;
  uint64_t w_base = 0;  // file offset of global row 0 of layers.N.engram.embed.weight
  uint64_t s_base = 0;  // file offset of global row 0 of layers.N.engram.embed.scale
  uint64_t lo = 0;      // first owned global row
  uint64_t hi = 0;      // one past the last owned global row
  uint64_t dim = 0;     // weight row bytes (fp8 e4m3, one byte per element)
  uint64_t sb = 0;      // scale row bytes (e8m0, one byte per 32-element block)
  bool direct = false;  // O_DIRECT reads through an aligned bounce buffer
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> owned_rows{0};
};

// Must match _RowioJob (ctypes.Structure) in overlay/engram.py.
struct Job {
  Table *table;
  const int64_t *ids;
  uint8_t *w_out;  // [n, dim]
  uint8_t *s_out;  // [n, sb]
  uint64_t n;
};

[[noreturn]] void die(const char *what, uint64_t off) {
  std::fprintf(stderr, "[dsv41_rowio] fatal: %s at file offset %llu (errno %d: %s)\n", what,
               static_cast<unsigned long long>(off), errno, std::strerror(errno));
  std::fflush(stderr);
  std::abort();
}

uint8_t *bounce_buffer() {
  thread_local uint8_t *buf = nullptr;
  if (buf == nullptr) {
    void *p = nullptr;
    if (posix_memalign(&p, 4096, 16384) != 0) die("posix_memalign", 0);
    buf = static_cast<uint8_t *>(p);
  }
  return buf;
}

void read_exact(int fd, uint8_t *dst, size_t len, uint64_t off, bool direct) {
  if (!direct) {
    size_t got = 0;
    while (got < len) {
      const ssize_t r = pread(fd, dst + got, len - got, static_cast<off_t>(off + got));
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) die("short read", off + got);
      got += static_cast<size_t>(r);
    }
    return;
  }
  // A 256-byte row can straddle a 4 KiB boundary: read the aligned span.
  const uint64_t start = off & ~uint64_t(4095);
  const uint64_t end = (off + len + 4095) & ~uint64_t(4095);
  const size_t span = static_cast<size_t>(end - start);
  if (span > 16384) die("row wider than the O_DIRECT bounce buffer", off);
  uint8_t *buf = bounce_buffer();
  ssize_t r;
  do {
    r = pread(fd, buf, span, static_cast<off_t>(start));
  } while (r < 0 && errno == EINTR);
  if (r < 0 || static_cast<uint64_t>(r) < (off - start) + len) die("short O_DIRECT read", off);
  std::memcpy(dst, buf + (off - start), len);
}

void do_rows(const Job &job, uint64_t begin, uint64_t end) {
  Table &t = *job.table;
  uint64_t owned = 0;
  for (uint64_t i = begin; i < end; ++i) {
    const int64_t id = job.ids[i];
    uint8_t *w = job.w_out + i * t.dim;
    uint8_t *s = job.s_out + i * t.sb;
    if (id < 0 || static_cast<uint64_t>(id) < t.lo || static_cast<uint64_t>(id) >= t.hi) {
      std::memset(w, 0, t.dim);  // fp8 zero times any scale is 0.0
      std::memset(s, 0, t.sb);
      continue;
    }
    const uint64_t row = static_cast<uint64_t>(id);
    read_exact(t.fd_w, w, t.dim, t.w_base + row * t.dim, t.direct);
    read_exact(t.fd_s, s, t.sb, t.s_base + row * t.sb, t.direct);
    ++owned;
  }
  t.owned_rows.fetch_add(owned, std::memory_order_relaxed);
}

struct Run {
  const Job *job = nullptr;
  uint64_t chunk = 0;
  uint64_t chunks = 0;
  std::atomic<uint64_t> next{0};
};

void drain(Run &run) {
  for (;;) {
    const uint64_t c = run.next.fetch_add(1);
    if (c >= run.chunks) return;
    const uint64_t b = c * run.chunk;
    do_rows(*run.job, b, std::min(run.job->n, b + run.chunk));
  }
}

// Workers plus the calling thread split one lookup into chunks. Lookups are
// serialized; run() returns only after every worker that joined has left.
class Pool {
 public:
  explicit Pool(int threads) {
    for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { loop(); });
  }

  int size() const { return static_cast<int>(workers_.size()); }

  void run(const Job &job) {
    const uint64_t lanes = static_cast<uint64_t>(workers_.size()) + 1;
    const uint64_t chunk = std::clamp<uint64_t>((job.n + lanes - 1) / lanes, 4, 256);
    if (job.n <= chunk || workers_.empty()) {
      do_rows(job, 0, job.n);
      return;
    }
    std::lock_guard<std::mutex> serial(serial_);
    Run r;
    r.job = &job;
    r.chunk = chunk;
    r.chunks = (job.n + chunk - 1) / chunk;
    {
      std::lock_guard<std::mutex> g(m_);
      current_ = &r;
      ++generation_;
    }
    cv_.notify_all();
    drain(r);
    std::unique_lock<std::mutex> lk(m_);
    current_ = nullptr;  // late wakers skip this run
    idle_.wait(lk, [this] { return active_ == 0; });
  }

 private:
  void loop() {
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lk(m_);
    for (;;) {
      cv_.wait(lk, [&] { return generation_ != seen; });
      seen = generation_;
      Run *r = current_;
      if (r == nullptr) continue;
      ++active_;
      lk.unlock();
      drain(*r);
      lk.lock();
      if (--active_ == 0) idle_.notify_all();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex serial_;
  std::mutex m_;
  std::condition_variable cv_;
  std::condition_variable idle_;
  Run *current_ = nullptr;
  uint64_t generation_ = 0;
  int active_ = 0;
};

Pool *pool() {
  // Never destroyed: the workers live as long as the process.
  static Pool *p = [] {
    int n = 32;
    if (const char *e = std::getenv("DSV41_ENGRAM_IO_THREADS")) n = std::atoi(e);
    return new Pool(std::max(0, std::min(n, 256)));
  }();
  return p;
}

int open_ro(const char *path, bool direct) {
  const int fd = open(path, O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
  // Random 256-byte reads: no readahead, so the page cache holds only rows we used.
  if (fd >= 0 && !direct) posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
  return fd;
}

}  // namespace

extern "C" {

void *rowio_open(const char *path_w, uint64_t w_base, const char *path_s, uint64_t s_base,
                 uint64_t lo, uint64_t hi, uint64_t dim, uint64_t sb, int direct) {
  auto *t = new Table;
  t->direct = direct != 0;
  t->fd_w = open_ro(path_w, t->direct);
  t->fd_s = open_ro(path_s, t->direct);
  struct stat sw {};
  struct stat ss {};
  const bool bad = t->fd_w < 0 || t->fd_s < 0 || fstat(t->fd_w, &sw) != 0 ||
                   fstat(t->fd_s, &ss) != 0 || lo > hi || dim == 0 || sb == 0 ||
                   w_base + hi * dim > static_cast<uint64_t>(sw.st_size) ||
                   s_base + hi * sb > static_cast<uint64_t>(ss.st_size);
  if (bad) {
    std::fprintf(stderr, "[dsv41_rowio] cannot serve rows [%llu, %llu) from %s / %s (errno %d)\n",
                 static_cast<unsigned long long>(lo), static_cast<unsigned long long>(hi), path_w,
                 path_s, errno);
    if (t->fd_w >= 0) close(t->fd_w);
    if (t->fd_s >= 0) close(t->fd_s);
    delete t;
    return nullptr;
  }
  t->w_base = w_base;
  t->s_base = s_base;
  t->lo = lo;
  t->hi = hi;
  t->dim = dim;
  t->sb = sb;
  pool();  // start the workers now, not inside the first host function
  return t;
}

void rowio_lookup(void *job) {
  const Job &j = *static_cast<const Job *>(job);
  j.table->calls.fetch_add(1, std::memory_order_relaxed);
  pool()->run(j);
}

void rowio_stats(void *table, uint64_t *out) {
  const auto *t = static_cast<const Table *>(table);
  out[0] = t->calls.load();
  out[1] = t->owned_rows.load();
}

int rowio_threads() { return pool()->size(); }

}  // extern "C"
