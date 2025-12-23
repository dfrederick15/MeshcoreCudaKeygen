// src/gui_main.cpp
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>

#include <string>
#include <atomic>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "gpu_search_ed25519_cuda.h"
#include "cpu_ref.h" // hex_of(), make_keypair_from_seed()

#pragma comment(lib, "Comctl32.lib")

// -------------------- Control IDs --------------------
enum : int {
  IDC_PREFIX    = 101,
  IDC_BATCH     = 102,
  IDC_START     = 103,
  IDC_STOP      = 104,
  IDC_BENCH     = 105,

  IDC_POOL_MB   = 106,
  IDC_POOL_INFO = 107,

  IDC_GPU_NAME  = 110,
  IDC_GPU_LOAD  = 111,
  IDC_TRIES     = 112,
  IDC_SPEED     = 113,
  IDC_RUNTIME   = 114,
  IDC_ESTIMATE  = 115,

  IDC_RESULT_PUB  = 120,
  IDC_RESULT_PRIV = 121,
};

static const UINT WM_APP_RATE      = WM_APP + 1;
static const UINT WM_APP_FOUND     = WM_APP + 2;
static const UINT WM_APP_ERROR     = WM_APP + 3;
static const UINT WM_APP_BENCHDONE = WM_APP + 4;

static const UINT_PTR TIMER_UI = 1;

// -------------------- Globals --------------------
static HWND g_hwndMain   = nullptr;
static HWND g_hPrefix    = nullptr;
static HWND g_hBatch     = nullptr;
static HWND g_hPoolMB    = nullptr;
static HWND g_hPoolInfo  = nullptr;

static HWND g_hStart     = nullptr;
static HWND g_hStop      = nullptr;
static HWND g_hBench     = nullptr;

static HWND g_hGpuName   = nullptr;
static HWND g_hGpuLoad   = nullptr;
static HWND g_hTries     = nullptr;
static HWND g_hSpeed     = nullptr;
static HWND g_hRuntime   = nullptr;
static HWND g_hEstimate  = nullptr;

static HWND g_hPub       = nullptr;
static HWND g_hPriv      = nullptr;

static HANDLE g_worker = nullptr;
static HANDLE g_benchThread = nullptr;

static std::atomic<bool> g_running{false};
static std::atomic<bool> g_stop{false};

static std::atomic<unsigned long long> g_totalTries{0};     // exact integer
static std::atomic<unsigned long long> g_lastRateKps{0};    // integer keys/sec
static std::atomic<unsigned long long> g_benchRateKps{0};   // integer keys/sec
static std::atomic<bool> g_benchRunning{false};

// Runtime timer
static std::atomic<bool> g_timerArmed{false};
static std::chrono::steady_clock::time_point g_runStart{};
static std::chrono::steady_clock::time_point g_runEnd{};

// Seed base counter (deterministic per regenerate)
static std::atomic<unsigned long long> g_seedBaseCounter{0x12345678ABCDEF00ULL};

// -------------------- UTF helpers --------------------
static std::wstring utf8_to_wstring(const std::string& s) {
  if (s.empty()) return L"";
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
  return w;
}

static std::string wstring_to_utf8(const std::wstring& w) {
  if (w.empty()) return "";
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
  return s;
}

static std::wstring get_text_w(HWND h) {
  int len = GetWindowTextLengthW(h);
  std::wstring w(len, L'\0');
  GetWindowTextW(h, w.data(), len + 1);
  return w;
}

// -------------------- Pretty formatting --------------------
static std::wstring fmt_commas_u64(unsigned long long v) {
  std::wstring s = std::to_wstring(v);
  int insertPos = (int)s.size() - 3;
  while (insertPos > 0) {
    s.insert((size_t)insertPos, 1, L',');
    insertPos -= 3;
  }
  return s;
}

static std::wstring fmt_percent(double p01) {
  if (!std::isfinite(p01)) return L"N/A";
  if (p01 < 0.0) p01 = 0.0;
  if (p01 > 1.0) p01 = 1.0;
  std::wstringstream ss;
  ss << std::fixed << std::setprecision(1) << (p01 * 100.0) << L"%";
  return ss.str();
}

static std::wstring fmt_seconds(double sec) {
  if (!std::isfinite(sec) || sec < 0) return L"N/A";
  if (sec < 1.0) { std::wstringstream ss; ss << std::fixed << std::setprecision(2) << sec << L"s"; return ss.str(); }
  if (sec < 60.0) { std::wstringstream ss; ss << std::fixed << std::setprecision(1) << sec << L"s"; return ss.str(); }
  if (sec < 3600.0) {
    int m = (int)(sec / 60.0);
    int s = (int)std::fmod(sec, 60.0);
    std::wstringstream ss; ss << m << L"m " << s << L"s"; return ss.str();
  }
  if (sec < 86400.0) {
    int h = (int)(sec / 3600.0);
    int m = (int)std::fmod(sec, 3600.0) / 60;
    std::wstringstream ss; ss << h << L"h " << m << L"m"; return ss.str();
  }
  int d = (int)(sec / 86400.0);
  int h = (int)std::fmod(sec, 86400.0) / 3600;
  std::wstringstream ss; ss << d << L"d " << h << L"h"; return ss.str();
}

static std::wstring fmt_elapsed_hms(double sec) {
  if (!std::isfinite(sec) || sec < 0) sec = 0;
  unsigned long long s = (unsigned long long)(sec);
  unsigned long long h = s / 3600ULL;
  unsigned long long m = (s % 3600ULL) / 60ULL;
  unsigned long long r = s % 60ULL;
  std::wstringstream ss;
  ss << std::setfill(L'0') << std::setw(2) << h << L":" << std::setw(2) << m << L":" << std::setw(2) << r;
  return ss.str();
}

// -------------------- NVML (optional, no nvml types) --------------------
static HMODULE g_nvml = nullptr;
static bool g_nvmlReady = false;
static void* g_nvmlDev = nullptr;

static int (*p_nvmlInit_v2)() = nullptr;
static int (*p_nvmlShutdown)() = nullptr;
static int (*p_nvmlDeviceGetHandleByIndex_v2)(unsigned int, void**) = nullptr;
static int (*p_nvmlDeviceGetUtilizationRates)(void*, void*) = nullptr;

#pragma pack(push, 1)
struct NvmlUtilRaw { uint32_t gpu; uint32_t memory; };
#pragma pack(pop)

static void nvml_try_init() {
  if (g_nvmlReady) return;
  g_nvml = LoadLibraryW(L"nvml.dll");
  if (!g_nvml) return;

  p_nvmlInit_v2 = (int (*)())GetProcAddress(g_nvml, "nvmlInit_v2");
  p_nvmlShutdown = (int (*)())GetProcAddress(g_nvml, "nvmlShutdown");
  p_nvmlDeviceGetHandleByIndex_v2 = (int (*)(unsigned int, void**))GetProcAddress(g_nvml, "nvmlDeviceGetHandleByIndex_v2");
  p_nvmlDeviceGetUtilizationRates = (int (*)(void*, void*))GetProcAddress(g_nvml, "nvmlDeviceGetUtilizationRates");

  if (!p_nvmlInit_v2 || !p_nvmlShutdown || !p_nvmlDeviceGetHandleByIndex_v2 || !p_nvmlDeviceGetUtilizationRates) {
    FreeLibrary(g_nvml); g_nvml = nullptr; return;
  }
  if (p_nvmlInit_v2() != 0) { FreeLibrary(g_nvml); g_nvml = nullptr; return; }
  if (p_nvmlDeviceGetHandleByIndex_v2(0, &g_nvmlDev) != 0) {
    p_nvmlShutdown(); FreeLibrary(g_nvml); g_nvml = nullptr; return;
  }
  g_nvmlReady = true;
}

static bool nvml_get_gpu_load(unsigned int& outGpuPercent) {
  if (!g_nvmlReady) return false;
  NvmlUtilRaw u{};
  if (p_nvmlDeviceGetUtilizationRates(g_nvmlDev, &u) != 0) return false;
  outGpuPercent = (unsigned int)u.gpu;
  return true;
}

static void nvml_shutdown() {
  if (g_nvmlReady) { p_nvmlShutdown(); g_nvmlReady = false; }
  if (g_nvml) { FreeLibrary(g_nvml); g_nvml = nullptr; }
  g_nvmlDev = nullptr;
}

// -------------------- Math for estimates --------------------
static int prefix_nibbles_from_text(const std::string& sRaw) {
  int n = 0;
  for (char c : sRaw) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
    if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) { n++; continue; }
    break;
  }
  return n;
}

static double mean_tries_nibbles(int n) { if (n <= 0) return 1.0; return std::pow(16.0, (double)n); }
static double p95_tries_nibbles(int n)  { if (n <= 0) return 1.0; return mean_tries_nibbles(n) * std::log(20.0); }

static double prob_found_by_tries(double tries, double mean) {
  if (!(mean > 0.0) || tries < 0.0) return 0.0;
  double x = tries / mean;
  if (x > 700.0) return 1.0;
  return 1.0 - std::exp(-x);
}

static double prob_find_within_seconds(double kps, double dtSec, double mean) {
  if (!(mean > 0.0) || kps <= 0.0 || dtSec <= 0.0) return 0.0;
  double x = (kps * dtSec) / mean;
  if (x > 700.0) return 1.0;
  return 1.0 - std::exp(-x);
}

// -------------------- UI update helpers --------------------
static void set_text(HWND h, const std::wstring& w) { SetWindowTextW(h, w.c_str()); }

static double get_elapsed_seconds() {
  if (!g_timerArmed.load(std::memory_order_relaxed)) return 0.0;
  auto end = g_running.load(std::memory_order_relaxed) ? std::chrono::steady_clock::now() : g_runEnd;
  return std::chrono::duration<double>(end - g_runStart).count();
}

static void update_ui_all() {
  const unsigned long long tries = g_totalTries.load(std::memory_order_relaxed);
  const unsigned long long liveKps = g_lastRateKps.load(std::memory_order_relaxed);
  const unsigned long long benchKps = g_benchRateKps.load(std::memory_order_relaxed);

  set_text(g_hTries, L"Keys tried: " + fmt_commas_u64(tries));
  set_text(g_hSpeed, L"Speed: " + fmt_commas_u64(liveKps) + L" keys/sec");

  const double elapsed = get_elapsed_seconds();
  set_text(g_hRuntime, L"Runtime: " + fmt_elapsed_hms(elapsed));

  // seed pool info
  size_t poolSeeds = gpu_seedpool_seed_count();
  if (poolSeeds > 0) {
    set_text(g_hPoolInfo, L"Pool: " + fmt_commas_u64((unsigned long long)poolSeeds) + L" seeds");
  } else {
    set_text(g_hPoolInfo, L"Pool: -");
  }

  const std::string prefix = wstring_to_utf8(get_text_w(g_hPrefix));
  const int nNibbles = prefix_nibbles_from_text(prefix);

  const unsigned long long kpsUseU64 =
    (g_running.load(std::memory_order_relaxed) && liveKps > 0) ? liveKps : benchKps;

  if (nNibbles <= 0 || kpsUseU64 == 0) {
    std::wstring benchNote = (benchKps > 0) ? (L" (bench: " + fmt_commas_u64(benchKps) + L" k/s)") : L"";
    set_text(g_hEstimate, L"Estimates: N/A" + benchNote);
    return;
  }

  const double kpsUse = (double)kpsUseU64;
  const double mean = mean_tries_nibbles(nNibbles);
  const double p95  = p95_tries_nibbles(nNibbles);

  const double eta_avg_from_now = mean / kpsUse;
  const double eta_p95_from_now = p95  / kpsUse;

  const double remaining_to_avg = (mean > (double)tries) ? (mean - (double)tries) / kpsUse : 0.0;
  const double remaining_to_p95 = (p95  > (double)tries) ? (p95  - (double)tries) / kpsUse : 0.0;

  const double p_found_now = prob_found_by_tries((double)tries, mean);
  const double p_next_60s  = prob_find_within_seconds(kpsUse, 60.0, mean);
  const double p_next_10m  = prob_find_within_seconds(kpsUse, 600.0, mean);

  std::wstringstream line1;
  line1 << L"ETA(avg): " << fmt_seconds(eta_avg_from_now)
        << L" | ETA(95%): " << fmt_seconds(eta_p95_from_now)
        << (g_running.load(std::memory_order_relaxed) ? L" (live)" : L" (bench)");

  std::wstringstream line2;
  line2 << L"Countdown(avg): " << fmt_seconds(remaining_to_avg)
        << L" | Countdown(95%): " << fmt_seconds(remaining_to_p95)
        << L" | P(found by now): " << fmt_percent(p_found_now)
        << L" | P(next 60s): " << fmt_percent(p_next_60s)
        << L" | P(next 10m): " << fmt_percent(p_next_10m);

  set_text(g_hEstimate, line1.str() + L"\r\n" + line2.str());
}

// -------------------- Worker params --------------------
struct WorkerParams {
  std::string prefix;
  size_t batch = 500000;
  size_t poolBytes = 0;
};

// -------------------- Worker thread (seed pool + search) --------------------
static DWORD WINAPI WorkerProc(LPVOID lp) {
  WorkerParams* p = (WorkerParams*)lp;

  g_totalTries.store(0, std::memory_order_relaxed);
  g_lastRateKps.store(0, std::memory_order_relaxed);

  cudaStream_t stream = 0;

  // Allocate pool
  if (!gpu_seedpool_init_bytes(p->poolBytes)) {
    std::wstring* w = new std::wstring(L"Failed to allocate seed pool in VRAM. Reduce Seed pool (MB).");
    PostMessageW(g_hwndMain, WM_APP_ERROR, 0, (LPARAM)w);
    delete p;
    g_running.store(false, std::memory_order_relaxed);
    return 0;
  }

  // Fill pool once
  const size_t poolSeeds = gpu_seedpool_seed_count();
  uint64_t base = (uint64_t)g_seedBaseCounter.fetch_add((unsigned long long)poolSeeds, std::memory_order_relaxed);
  if (!gpu_seedpool_generate(base, stream)) {
    std::wstring* w = new std::wstring(L"Failed to generate seed pool.");
    PostMessageW(g_hwndMain, WM_APP_ERROR, 0, (LPARAM)w);
    gpu_seedpool_free();
    delete p;
    g_running.store(false, std::memory_order_relaxed);
    return 0;
  }
  cudaDeviceSynchronize();

  size_t cursor = 0;

  auto t0 = std::chrono::steady_clock::now();
  auto lastRate = t0;

  while (!g_stop.load(std::memory_order_relaxed)) {
    GpuHit hit = gpu_find_pubkey_prefix_ed25519_cuda_seedpool(p->prefix, p->batch, &cursor, stream);

    const unsigned long long total =
      g_totalTries.fetch_add((unsigned long long)p->batch, std::memory_order_relaxed)
      + (unsigned long long)p->batch;

    // regenerate when we wrap
    if (cursor == 0) {
      uint64_t base2 = (uint64_t)g_seedBaseCounter.fetch_add((unsigned long long)poolSeeds, std::memory_order_relaxed);
      gpu_seedpool_generate(base2, stream);
    }

    auto now = std::chrono::steady_clock::now();
    if (now - lastRate >= std::chrono::milliseconds(250)) {
      lastRate = now;
      const double secs = std::chrono::duration<double>(now - t0).count();
      const double rate = (secs > 0.0) ? (double)total / secs : 0.0;
      const unsigned long long kps = (unsigned long long)(rate + 0.5);
      g_lastRateKps.store(kps, std::memory_order_relaxed);
      PostMessageW(g_hwndMain, WM_APP_RATE, 0, 0);
    }

    if (hit.found) {
      // Sanity check on CPU
      {
        std::array<unsigned char,32> seedArr{};
        for (int i = 0; i < 32; i++) seedArr[i] = hit.seed[i];
        auto cpuKp = make_keypair_from_seed(seedArr);
        if (memcmp(cpuKp.pub.data(), hit.pub, 32) != 0) {
          std::wstring* w = new std::wstring(L"ERROR: GPU pub != CPU pub for winning seed.");
          PostMessageW(g_hwndMain, WM_APP_ERROR, 0, (LPARAM)w);
          break;
        }
      }

      const std::string pubHex  = hex_of(hit.pub, 32);
      const std::string privHex = hex_of(hit.priv64, 64);

      std::wstring* payload = new std::wstring(utf8_to_wstring(pubHex + "\n" + privHex));
      PostMessageW(g_hwndMain, WM_APP_FOUND, 0, (LPARAM)payload);
      break;
    }
  }

  gpu_seedpool_free();

  delete p;
  g_running.store(false, std::memory_order_relaxed);
  g_stop.store(false, std::memory_order_relaxed);

  PostMessageW(g_hwndMain, WM_APP_RATE, 0, 0);
  return 0;
}

// -------------------- Benchmark thread --------------------
struct BenchParams { size_t batch = 500000; };

static DWORD WINAPI BenchProc(LPVOID lp) {
  BenchParams* p = (BenchParams*)lp;

  const std::string hardPrefix = "FFFFFFFFFFFFFFFF";
  const int iters = 6;

  unsigned long long totalKeys = 0;
  auto t0 = std::chrono::steady_clock::now();

  try {
    for (int i = 0; i < iters; i++) {
      if (!g_benchRunning.load(std::memory_order_relaxed)) break;
      (void)gpu_find_pubkey_prefix_ed25519_cuda(hardPrefix, p->batch);
      totalKeys += (unsigned long long)p->batch;
    }
  } catch (...) {
    totalKeys = 0;
  }

  auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  const unsigned long long kps = (secs > 0.0) ? (unsigned long long)((double)totalKeys / secs + 0.5) : 0;

  g_benchRateKps.store(kps, std::memory_order_relaxed);
  g_benchRunning.store(false, std::memory_order_relaxed);

  PostMessageW(g_hwndMain, WM_APP_BENCHDONE, 0, 0);

  delete p;
  return 0;
}

// -------------------- Window Proc --------------------
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_CREATE: {
      g_hwndMain = hwnd;

      nvml_try_init();
      SetTimer(hwnd, TIMER_UI, 250, nullptr);

      // GPU model
      std::wstring gpuName = L"GPU: (unknown)";
      int dev = 0;
      if (cudaGetDevice(&dev) == cudaSuccess) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
          gpuName = L"GPU: " + utf8_to_wstring(prop.name);
        }
      }

      int x = 12, y = 12;

      CreateWindowW(L"STATIC", L"Prefix (hex):", WS_CHILD | WS_VISIBLE, x, y, 90, 20, hwnd, nullptr, nullptr, nullptr);
      g_hPrefix = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"146880",
        WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
        x + 100, y - 2, 200, 24, hwnd, (HMENU)IDC_PREFIX, nullptr, nullptr);

      CreateWindowW(L"STATIC", L"Batch:", WS_CHILD | WS_VISIBLE, x + 310, y, 45, 20, hwnd, nullptr, nullptr, nullptr);
      g_hBatch = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"500000",
        WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
        x + 360, y - 2, 110, 24, hwnd, (HMENU)IDC_BATCH, nullptr, nullptr);

      CreateWindowW(L"STATIC", L"Seed pool (MB):", WS_CHILD | WS_VISIBLE, x + 480, y, 95, 20, hwnd, nullptr, nullptr, nullptr);
      g_hPoolMB = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"1024",
        WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
        x + 580, y - 2, 70, 24, hwnd, (HMENU)IDC_POOL_MB, nullptr, nullptr);

      g_hPoolInfo = CreateWindowW(L"STATIC", L"Pool: -", WS_CHILD | WS_VISIBLE,
        x + 660, y, 160, 20, hwnd, (HMENU)IDC_POOL_INFO, nullptr, nullptr);

      g_hStart = CreateWindowW(L"BUTTON", L"Start", WS_CHILD | WS_VISIBLE,
        x, y + 30, 70, 26, hwnd, (HMENU)IDC_START, nullptr, nullptr);

      g_hStop = CreateWindowW(L"BUTTON", L"Stop", WS_CHILD | WS_VISIBLE | WS_DISABLED,
        x + 78, y + 30, 70, 26, hwnd, (HMENU)IDC_STOP, nullptr, nullptr);

      g_hBench = CreateWindowW(L"BUTTON", L"Benchmark", WS_CHILD | WS_VISIBLE,
        x + 156, y + 30, 95, 26, hwnd, (HMENU)IDC_BENCH, nullptr, nullptr);

      y += 70;
      g_hGpuName = CreateWindowW(L"STATIC", gpuName.c_str(), WS_CHILD | WS_VISIBLE,
        x, y, 820, 20, hwnd, (HMENU)IDC_GPU_NAME, nullptr, nullptr);

      y += 22;
      g_hGpuLoad = CreateWindowW(L"STATIC", L"GPU Load: N/A", WS_CHILD | WS_VISIBLE,
        x, y, 160, 20, hwnd, (HMENU)IDC_GPU_LOAD, nullptr, nullptr);

      g_hTries = CreateWindowW(L"STATIC", L"Keys tried: 0", WS_CHILD | WS_VISIBLE,
        x + 170, y, 260, 20, hwnd, (HMENU)IDC_TRIES, nullptr, nullptr);

      g_hSpeed = CreateWindowW(L"STATIC", L"Speed: 0 keys/sec", WS_CHILD | WS_VISIBLE,
        x + 450, y, 270, 20, hwnd, (HMENU)IDC_SPEED, nullptr, nullptr);

      y += 22;
      g_hRuntime = CreateWindowW(L"STATIC", L"Runtime: 00:00:00", WS_CHILD | WS_VISIBLE,
        x, y, 220, 20, hwnd, (HMENU)IDC_RUNTIME, nullptr, nullptr);

      y += 26;
      g_hEstimate = CreateWindowW(L"STATIC", L"Estimates: N/A", WS_CHILD | WS_VISIBLE,
        x, y, 860, 44, hwnd, (HMENU)IDC_ESTIMATE, nullptr, nullptr);

      y += 54;
      CreateWindowW(L"STATIC", L"Result (when found):", WS_CHILD | WS_VISIBLE, x, y, 200, 20, hwnd, nullptr, nullptr, nullptr);

      y += 22;
      CreateWindowW(L"STATIC", L"Public Key:", WS_CHILD | WS_VISIBLE, x, y, 80, 20, hwnd, nullptr, nullptr, nullptr);
      g_hPub = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
        WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | ES_READONLY,
        x + 90, y - 2, 770, 24, hwnd, (HMENU)IDC_RESULT_PUB, nullptr, nullptr);

      y += 34;
      CreateWindowW(L"STATIC", L"Private Key:", WS_CHILD | WS_VISIBLE, x, y, 80, 20, hwnd, nullptr, nullptr, nullptr);
      g_hPriv = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
        WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | ES_READONLY,
        x + 90, y - 2, 770, 24, hwnd, (HMENU)IDC_RESULT_PRIV, nullptr, nullptr);

      HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
      auto setfont = [&](HWND h){ SendMessageW(h, WM_SETFONT, (WPARAM)hFont, TRUE); };

      setfont(g_hPrefix); setfont(g_hBatch); setfont(g_hPoolMB); setfont(g_hPoolInfo);
      setfont(g_hStart); setfont(g_hStop); setfont(g_hBench);
      setfont(g_hGpuName); setfont(g_hGpuLoad); setfont(g_hTries); setfont(g_hSpeed);
      setfont(g_hRuntime); setfont(g_hEstimate);
      setfont(g_hPub); setfont(g_hPriv);

      g_timerArmed.store(false, std::memory_order_relaxed);
      update_ui_all();
      return 0;
    }

    case WM_TIMER: {
      if (wParam == TIMER_UI) {
        unsigned int load = 0;
        if (!g_nvmlReady) nvml_try_init();
        if (nvml_get_gpu_load(load)) set_text(g_hGpuLoad, L"GPU Load: " + std::to_wstring(load) + L"%");
        else set_text(g_hGpuLoad, L"GPU Load: N/A");
        update_ui_all();
      }
      return 0;
    }

    case WM_COMMAND: {
      const int id = LOWORD(wParam);

      if (id == IDC_START) {
        if (g_running.load(std::memory_order_relaxed)) return 0;

        const std::string prefix = wstring_to_utf8(get_text_w(g_hPrefix));

        long long batchLL = _wtoll(get_text_w(g_hBatch).c_str());
        if (batchLL <= 0) batchLL = 500000;
        size_t batch = (size_t)batchLL;

        long long poolMBLL = _wtoll(get_text_w(g_hPoolMB).c_str());
        if (poolMBLL <= 0) poolMBLL = 1024;
        size_t poolBytesWanted = (size_t)poolMBLL * 1024ULL * 1024ULL;

        // Clamp to free VRAM with a margin
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
          size_t margin = 256ULL * 1024ULL * 1024ULL; // 256MB margin
          if (freeB > margin && poolBytesWanted > (freeB - margin)) poolBytesWanted = freeB - margin;
        }

        set_text(g_hPub, L"");
        set_text(g_hPriv, L"");

        // arm runtime timer
        g_runStart = std::chrono::steady_clock::now();
        g_runEnd = g_runStart;
        g_timerArmed.store(true, std::memory_order_relaxed);

        g_stop.store(false, std::memory_order_relaxed);
        g_running.store(true, std::memory_order_relaxed);

        EnableWindow(g_hStart, FALSE);
        EnableWindow(g_hStop, TRUE);
        EnableWindow(g_hBench, FALSE);

        WorkerParams* p = new WorkerParams();
        p->prefix = prefix;
        p->batch = batch;
        p->poolBytes = poolBytesWanted;

        g_worker = CreateThread(nullptr, 0, WorkerProc, p, 0, nullptr);
        return 0;
      }

      if (id == IDC_STOP) {
        if (!g_running.load(std::memory_order_relaxed)) return 0;
        g_stop.store(true, std::memory_order_relaxed);
        EnableWindow(g_hStop, FALSE);

        g_runEnd = std::chrono::steady_clock::now();
        g_running.store(false, std::memory_order_relaxed);
        return 0;
      }

      if (id == IDC_BENCH) {
        if (g_running.load(std::memory_order_relaxed)) return 0;
        if (g_benchRunning.load(std::memory_order_relaxed)) return 0;

        long long batchLL = _wtoll(get_text_w(g_hBatch).c_str());
        if (batchLL <= 0) batchLL = 500000;

        g_benchRunning.store(true, std::memory_order_relaxed);
        EnableWindow(g_hBench, FALSE);
        set_text(g_hEstimate, L"Benchmarking...");

        BenchParams* bp = new BenchParams();
        bp->batch = (size_t)batchLL;
        g_benchThread = CreateThread(nullptr, 0, BenchProc, bp, 0, nullptr);
        return 0;
      }

      return 0;
    }

    case WM_APP_RATE: {
      update_ui_all();
      if (!g_running.load(std::memory_order_relaxed)) {
        EnableWindow(g_hStart, TRUE);
        EnableWindow(g_hStop, FALSE);
        EnableWindow(g_hBench, !g_benchRunning.load(std::memory_order_relaxed));
      }
      return 0;
    }

    case WM_APP_BENCHDONE: {
      EnableWindow(g_hBench, TRUE);
      update_ui_all();
      return 0;
    }

    case WM_APP_FOUND: {
      g_runEnd = std::chrono::steady_clock::now();
      g_running.store(false, std::memory_order_relaxed);

      std::wstring* payload = (std::wstring*)lParam;
      if (payload) {
        size_t pos = payload->find(L'\n');
        std::wstring pub = (pos == std::wstring::npos) ? *payload : payload->substr(0, pos);
        std::wstring priv = (pos == std::wstring::npos) ? L"" : payload->substr(pos + 1);
        set_text(g_hPub, pub);
        set_text(g_hPriv, priv);
        delete payload;
      }

      g_stop.store(true, std::memory_order_relaxed);

      EnableWindow(g_hStart, TRUE);
      EnableWindow(g_hStop, FALSE);
      EnableWindow(g_hBench, TRUE);

      update_ui_all();
      return 0;
    }

    case WM_APP_ERROR: {
      g_runEnd = std::chrono::steady_clock::now();
      g_running.store(false, std::memory_order_relaxed);

      std::wstring* w = (std::wstring*)lParam;
      if (w) { MessageBoxW(hwnd, w->c_str(), L"Error", MB_ICONERROR | MB_OK); delete w; }

      g_stop.store(true, std::memory_order_relaxed);

      EnableWindow(g_hStart, TRUE);
      EnableWindow(g_hStop, FALSE);
      EnableWindow(g_hBench, TRUE);

      update_ui_all();
      return 0;
    }

    case WM_DESTROY: {
      KillTimer(hwnd, TIMER_UI);

      if (g_running.load(std::memory_order_relaxed)) {
        g_stop.store(true, std::memory_order_relaxed);
        if (g_worker) { WaitForSingleObject(g_worker, 3000); CloseHandle(g_worker); g_worker = nullptr; }
      }

      if (g_benchRunning.load(std::memory_order_relaxed)) {
        g_benchRunning.store(false, std::memory_order_relaxed);
        if (g_benchThread) { WaitForSingleObject(g_benchThread, 3000); CloseHandle(g_benchThread); g_benchThread = nullptr; }
      }

      nvml_shutdown();
      PostQuitMessage(0);
      return 0;
    }
  }

  return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// -------------------- GUI main --------------------
static int GuiMain(HINSTANCE hInst, int nCmdShow) {
  INITCOMMONCONTROLSEX icc{ sizeof(icc), ICC_STANDARD_CLASSES };
  InitCommonControlsEx(&icc);

  const wchar_t* CLASS_NAME = L"MeshcoreCudaKeygenGuiWnd";

  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = WndProc;
  wc.hInstance = hInst;
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wc.lpszClassName = CLASS_NAME;

  if (!RegisterClassExW(&wc)) return 0;

  HWND hwnd = CreateWindowExW(
    0, CLASS_NAME, L"Meshcore CUDA Keygen",
    WS_OVERLAPPEDWINDOW,
    CW_USEDEFAULT, CW_USEDEFAULT, 920, 430,
    nullptr, nullptr, hInst, nullptr
  );

  if (!hwnd) return 0;

  ShowWindow(hwnd, nCmdShow);
  UpdateWindow(hwnd);

  MSG m;
  while (GetMessageW(&m, nullptr, 0, 0)) {
    TranslateMessage(&m);
    DispatchMessageW(&m);
  }
  return (int)m.wParam;
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int nCmdShow) {
  return GuiMain(hInst, nCmdShow);
}
