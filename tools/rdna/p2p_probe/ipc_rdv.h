// ipc_rdv.h - file-based rendezvous for cooperating processes (header only).
//
// Dependency-free on purpose: C++17 + POSIX, no HIP, no torch. Used by
// ipc_probe.cpp (T1 cross-arch IPC probe) and meant to be reused by the hx
// protocol loopback test to exchange hipIpcMemHandle_t values between a hub
// process and its expert-server processes.
//
// Model
//   All processes share one directory (--dir). Every message is a small text
//   file written with an atomic publish (write a hidden temp file in the same
//   directory, then rename(2)), so a reader sees either no file or the whole
//   file. Readers poll with an adaptive sleep (50 us .. 1 ms) and always have
//   a timeout. The directory must NOT be under /tmp (box rule): open() refuses
//   /tmp, /var/tmp and /private/tmp (also after realpath).
//
//   rdv::Rdv rv;
//   rv.open("/home/u/w4a8_runs/x/rdv", "hub");        // mkdir -p, pid file
//   rv.watch("srv0");                                 // liveness of a peer
//   rv.publish("hub.buf", rdv::kv_text({{"handle", rdv::hex_encode(&h, 64)}}),
//              /*ephemeral=*/true);
//   std::string txt;
//   if (!rv.wait_read("srv0.buf", 30000, &txt)) fail(rv.error());
//   rdv::Kv kv = rdv::kv_parse(txt);                  // "k=v" lines
//   rv.barrier("start", {"hub", "srv0"}, 30000);      // all publish, all wait
//   rv.cleanup();                                     // unlink ephemerals
//
// Liveness
//   open() publishes the ephemeral file "pid.<me>". After watch(id), every
//   blocking wait also checks that the peer's pid (read from "pid.<id>" and
//   remembered) is alive (kill(0) plus /proc state, so a zombie child counts
//   as dead) and fails with "peer <id> died" instead of waiting for the
//   timeout. The file is checked before liveness, so a peer that published
//   its last message and exited is still readable.
//
// Cleanup
//   Files published with ephemeral=true (handles, pid) are unlinked by
//   cleanup(), by an atexit hook, and by the SIGINT/SIGTERM/SIGHUP handler
//   (which unlinks them with async-signal-safe calls and re-raises the signal
//   with the default action, so a process blocked in a GPU call still dies).
//   Non-ephemeral files (results, barrier markers) are kept as a record.
//   Bound: 256 ephemeral files per process.
//
// Names are plain file names (no '/'), at most 200 bytes.
#pragma once

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace rdv {

inline int64_t now_us() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}
inline int64_t now_ms() { return now_us() / 1000; }
inline void sleep_us(long us) {
  timespec ts;
  ts.tv_sec = us / 1000000;
  ts.tv_nsec = (us % 1000000) * 1000;
  nanosleep(&ts, nullptr);
}

// ---- hex (IPC handles are opaque 64-byte blobs) ---------------------------
inline std::string hex_encode(const void* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s(2 * n, '0');
  const unsigned char* b = (const unsigned char*)p;
  for (size_t i = 0; i < n; i++) {
    s[2 * i] = d[b[i] >> 4];
    s[2 * i + 1] = d[b[i] & 15];
  }
  return s;
}
inline bool hex_decode(const std::string& s, void* out, size_t n) {
  if (s.size() != 2 * n) return false;
  unsigned char* b = (unsigned char*)out;
  for (size_t i = 0; i < n; i++) {
    int v[2];
    for (int k = 0; k < 2; k++) {
      char ch = s[2 * i + k];
      if (ch >= '0' && ch <= '9')
        v[k] = ch - '0';
      else if (ch >= 'a' && ch <= 'f')
        v[k] = ch - 'a' + 10;
      else if (ch >= 'A' && ch <= 'F')
        v[k] = ch - 'A' + 10;
      else
        return false;
    }
    b[i] = (unsigned char)(v[0] << 4 | v[1]);
  }
  return true;
}

// ---- key=value text -------------------------------------------------------
using Kv = std::map<std::string, std::string>;
inline Kv kv_parse(const std::string& text) {
  Kv kv;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t nl = text.find('\n', pos);
    if (nl == std::string::npos) nl = text.size();
    std::string line = text.substr(pos, nl - pos);
    pos = nl + 1;
    size_t eq = line.find('=');
    if (eq == std::string::npos || eq == 0) continue;
    kv[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return kv;
}
inline std::string kv_get(const Kv& kv, const std::string& k,
                          const std::string& def = "") {
  auto it = kv.find(k);
  return it == kv.end() ? def : it->second;
}
inline long long kv_int(const Kv& kv, const std::string& k, long long def = 0) {
  auto it = kv.find(k);
  if (it == kv.end() || it->second.empty()) return def;
  return strtoll(it->second.c_str(), nullptr, 0);
}
inline std::string kv_text(
    std::initializer_list<std::pair<std::string, std::string>> items) {
  std::string s;
  for (auto& p : items) s += p.first + "=" + p.second + "\n";
  return s;
}

// ---- liveness -------------------------------------------------------------
// True if `pid` exists and is not a zombie. On Linux the state letter in
// /proc/<pid>/stat is checked (a dead child that nobody wait()ed for is 'Z').
inline bool pid_alive(long pid) {
  if (pid <= 0) return false;
  if (kill((pid_t)pid, 0) != 0 && errno != EPERM) return false;
  if (access("/proc/self/stat", F_OK) != 0) return true;  // no procfs
  char path[64];
  snprintf(path, sizeof path, "/proc/%ld/stat", pid);
  FILE* f = fopen(path, "r");
  if (!f) return false;
  char buf[512];
  size_t n = fread(buf, 1, sizeof buf - 1, f);
  fclose(f);
  buf[n] = 0;
  const char* rp = strrchr(buf, ')');
  if (!rp || rp[1] != ' ') return true;
  return rp[2] != 'Z' && rp[2] != 'X';
}

// ---- cleanup registry (async-signal-safe) ---------------------------------
namespace detail {
constexpr int kMaxPaths = 256;
constexpr int kPathLen = 1024;
struct Registry {
  char paths[kMaxPaths][kPathLen];
  char pending[kPathLen];  // temp file of a publish in flight
  volatile sig_atomic_t n;
  volatile sig_atomic_t stop;
  volatile sig_atomic_t installed;
};
inline Registry& reg() {
  static Registry r;  // zero-initialised
  return r;
}
inline void unlink_all() {
  Registry& r = reg();
  if (r.pending[0]) unlink(r.pending);
  for (int i = 0; i < r.n; i++) unlink(r.paths[i]);
  r.n = 0;
  r.pending[0] = 0;
}
inline void on_signal(int sig) {
  reg().stop = 1;
  unlink_all();
  signal(sig, SIG_DFL);
  raise(sig);
}
inline void atexit_cleanup() { unlink_all(); }
inline void install_handlers() {
  Registry& r = reg();
  if (r.installed) return;
  r.installed = 1;
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGHUP, &sa, nullptr);
  atexit(atexit_cleanup);
}
inline bool track(const std::string& path) {
  Registry& r = reg();
  if (r.n >= kMaxPaths || path.size() >= (size_t)kPathLen) return false;
  memcpy(r.paths[r.n], path.c_str(), path.size() + 1);
  r.n = r.n + 1;
  return true;
}
inline bool under_tmp(const std::string& p) {
  static const char* bad[] = {"/tmp", "/var/tmp", "/private/tmp",
                              "/private/var/tmp"};
  for (const char* b : bad) {
    size_t n = strlen(b);
    if (p.compare(0, n, b) == 0 && (p.size() == n || p[n] == '/')) return true;
  }
  return false;
}
}  // namespace detail

// Unlink every ephemeral file published by this process.
inline void cleanup_all() { detail::unlink_all(); }
// True after SIGINT/SIGTERM/SIGHUP was received (only observable if the
// default action was replaced afterwards).
inline bool stop_requested() { return detail::reg().stop != 0; }

class Rdv {
 public:
  // Create (mkdir -p) and adopt the directory; publish pid.<me>. Returns false
  // and sets error() on failure.
  bool open(const std::string& dir, const std::string& me) {
    if (dir.empty()) return fail("empty rendezvous directory");
    std::string abs = dir;
    if (abs[0] != '/') {
      char cwd[PATH_MAX];
      if (!getcwd(cwd, sizeof cwd)) return fail("getcwd failed");
      abs = std::string(cwd) + "/" + dir;
    }
    if (detail::under_tmp(abs))
      return fail("refusing rendezvous directory under /tmp: " + abs);
    if (!mkdir_p(abs)) return fail("mkdir -p failed for " + abs);
    char real[PATH_MAX];
    if (!realpath(abs.c_str(), real)) return fail("realpath failed " + abs);
    dir_ = real;
    if (detail::under_tmp(dir_))
      return fail("refusing rendezvous directory under /tmp: " + dir_);
    if (!valid_name(me)) return fail("bad participant id " + me);
    me_ = me;
    detail::install_handlers();
    char pidtxt[32];
    snprintf(pidtxt, sizeof pidtxt, "%ld\n", (long)getpid());
    return publish("pid." + me_, pidtxt, true);
  }

  const std::string& dir() const { return dir_; }
  const std::string& me() const { return me_; }
  const std::string& error() const { return err_; }
  std::string path(const std::string& name) const { return dir_ + "/" + name; }

  // Atomically create <dir>/<name> with `text`. Ephemeral files are removed by
  // cleanup(), the exit hook and the signal handler.
  bool publish(const std::string& name, const std::string& text,
               bool ephemeral = false) {
    if (!valid_name(name)) return fail("bad file name " + name);
    std::string fin = path(name);
    std::string tmp = dir_ + "/." + name + ".tmp." + std::to_string(getpid());
    if (tmp.size() >= (size_t)detail::kPathLen ||
        (ephemeral && !detail::track(fin)))
      return fail("rendezvous path too long or registry full: " + name);
    memcpy(detail::reg().pending, tmp.c_str(), tmp.size() + 1);
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return fail("open " + tmp + ": " + strerror(errno));
    size_t off = 0;
    while (off < text.size()) {
      ssize_t w = ::write(fd, text.data() + off, text.size() - off);
      if (w < 0 && errno == EINTR) continue;
      if (w <= 0) {
        int e = errno;
        ::close(fd);
        unlink(tmp.c_str());
        detail::reg().pending[0] = 0;
        return fail("write " + tmp + ": " + strerror(e));
      }
      off += (size_t)w;
    }
    ::close(fd);
    if (rename(tmp.c_str(), fin.c_str()) != 0) {
      int e = errno;
      unlink(tmp.c_str());
      detail::reg().pending[0] = 0;
      return fail("rename to " + fin + ": " + strerror(e));
    }
    detail::reg().pending[0] = 0;
    return true;
  }

  // Non-blocking: false if the file does not exist yet.
  bool try_read(const std::string& name, std::string* out) const {
    if (!valid_name(name)) return false;
    int fd = ::open(path(name).c_str(), O_RDONLY);
    if (fd < 0) return false;
    std::string s;
    char buf[4096];
    for (;;) {
      ssize_t n = ::read(fd, buf, sizeof buf);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) break;
      s.append(buf, (size_t)n);
    }
    ::close(fd);
    *out = s;
    return true;
  }
  bool exists(const std::string& name) const {
    return valid_name(name) && access(path(name).c_str(), F_OK) == 0;
  }

  // Remember a peer whose pid file is "pid.<id>"; waits fail if it dies.
  void watch(const std::string& id) {
    if (!peers_.count(id)) peers_[id] = 0;
  }

  // Block until <name> exists (or timeout / peer death / stop request).
  // `out` may be null.
  bool wait_read(const std::string& name, int64_t timeout_ms,
                 std::string* out = nullptr) {
    std::string txt;
    int64_t t0 = now_ms();
    long delay = 50;
    for (;;) {
      if (try_read(name, &txt)) {
        if (out) *out = txt;
        return true;
      }
      if (stop_requested()) return fail("interrupted waiting for " + name);
      std::string dead = dead_peer();
      if (!dead.empty())
        return fail("peer " + dead + " died while waiting for " + name);
      if (now_ms() - t0 >= timeout_ms)
        return fail("timeout after " + std::to_string(timeout_ms) +
                    " ms waiting for " + name);
      sleep_us(delay);
      if (delay < 1000) delay *= 2;
    }
  }

  // Publish bar.<name>.<me> and wait for bar.<name>.<p> of every other party.
  // `parties` includes this process (ignored if absent). Names must be unique
  // per use.
  bool barrier(const std::string& name, const std::vector<std::string>& parties,
               int64_t timeout_ms) {
    if (!publish("bar." + name + "." + me_, "1\n", false)) return false;
    for (const std::string& p : parties) {
      if (p == me_) continue;
      watch(p);
      if (!wait_read("bar." + name + "." + p, timeout_ms, nullptr))
        return false;
    }
    return true;
  }

  // Id of a watched peer whose pid is known and no longer alive, else "".
  std::string dead_peer() {
    for (auto& kv : peers_) {
      if (kv.second == 0) {
        std::string t;
        if (try_read("pid." + kv.first, &t)) kv.second = atol(t.c_str());
      }
      if (kv.second != 0 && !pid_alive(kv.second)) return kv.first;
    }
    return "";
  }

  void cleanup() { cleanup_all(); }

 private:
  static bool valid_name(const std::string& n) {
    return !n.empty() && n.size() <= 200 && n.find('/') == std::string::npos &&
           n != "." && n != "..";
  }
  bool fail(const std::string& m) {
    err_ = m;
    return false;
  }
  static bool mkdir_p(const std::string& p) {
    std::string cur;
    size_t i = 0;
    while (i <= p.size()) {
      size_t j = p.find('/', i);
      if (j == std::string::npos) j = p.size();
      cur = p.substr(0, j);
      if (!cur.empty() && mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
        return false;
      i = j + 1;
    }
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
  }

  std::string dir_, me_, err_;
  std::map<std::string, long> peers_;
};

}  // namespace rdv
