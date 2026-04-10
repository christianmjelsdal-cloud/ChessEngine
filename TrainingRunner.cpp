// TrainingRunner.cpp  --  Standalone Win32 NNUE Training GUI
// No browser, no server, no network required.
// Build: MSVC C++17, link gdiplus.lib comctl32.lib
//
// POST-BUILD EVENT (set in project properties -> Build Events -> Post-Build Event):
// xcopy /Y /E /I "$(SolutionDir)assets" "$(OutDir)assets\"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#pragma comment(linker, "/ENTRY:mainCRTStartup")
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "comctl32.lib")

#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <tlhelp32.h>
#include <objidl.h>
#pragma warning(push, 0)   // suppress code analysis messages in external SDK headers
#include <gdiplus.h>
#pragma warning(pop)
using namespace Gdiplus;
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <functional>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <memory>

namespace fs = std::filesystem;

// ── Layout constants ──────────────────────────────────────────────
static const int PANEL_W  = 295;
static const int TITLE_H  = 36;
static const int LOG_H    = 170;
static const int PROG_H   = 8;
static const int BANNER_H = 24;

// ── Control IDs ───────────────────────────────────────────────────
enum {
    ID_BTN_START        = 1001,
    ID_BTN_STOP         = 1002,
    ID_BTN_PAUSE        = 1003,
    ID_COMBO_PRESET     = 1004,
    ID_EDIT_GENS        = 1010,
    ID_EDIT_GAMES       = 1011,
    ID_EDIT_EPOCHS      = 1012,
    ID_EDIT_BATCHSZ     = 1013,
    ID_EDIT_LR          = 1014,
    ID_EDIT_WD          = 1015,
    ID_EDIT_DROPOUT     = 1016,
    ID_EDIT_LSMOOTH     = 1017,
    ID_EDIT_GRADACCUM   = 1018,
    ID_EDIT_WARMUP      = 1019,
    ID_EDIT_DRAWWT      = 1020,
    ID_EDIT_MATEBOOST   = 1021,
    ID_EDIT_SPLRATIO    = 1022,
    ID_EDIT_WORKERS     = 1023,
    ID_EDIT_DEPTH       = 1024,
    ID_EDIT_STARTGEN    = 1025,
    ID_CHK_ELO          = 1030,
    ID_CHK_OVERFIT      = 1031,
    ID_LOG_BOX          = 1040,
    ID_STATUS_TXT       = 1041,
    ID_PROGRESS         = 1042,
    ID_BTN_SAVE_PRESET  = 1005,
    ID_BTN_DEL_PRESET   = 1006,
    ID_CHK_GRAPH_LOSS   = 1050,
    ID_CHK_GRAPH_ACC    = 1051,
    ID_CHK_GRAPH_LR     = 1052,
    ID_COMBO_VARIANT    = 1060,
    ID_TIMER            = 2001,
};

// ── Dark theme colors ─────────────────────────────────────────────
static const COLORREF C_BG     = RGB(16, 16, 24);
static const COLORREF C_PANEL  = RGB(22, 22, 32);
static const COLORREF C_TEXT   = RGB(215, 215, 228);
static const COLORREF C_DIM    = RGB(100, 100, 120);
static const COLORREF C_ACCENT = RGB(65, 125, 245);

// ── Data structures ───────────────────────────────────────────────
// Variant enum for chess variant selection
enum class ChessVariant { Standard = 0, DuckChess = 1 };

struct Config {
    ChessVariant variant = ChessVariant::Standard;
    int    generations   = 10;
    int    gamesPerGen   = 5000;
    int    epochsPerGen  = 10;
    int    batchSize     = 2048;
    double lr            = 0.001;
    double weightDecay   = 1e-5;
    double dropout       = 0.1;
    double labelSmooth   = 0.05;
    int    gradAccum     = 4;
    int    warmupSteps   = 500;
    double drawWeight    = 0.25;
    double mateBoost     = 3.0;
    double splRatio      = 0.4;
    int    workers       = 12;
    int    depth         = 5;
    int    startGen      = 0;
    int    maxPositions  = 300000;
    int    earlyStop     = 10;
    bool   eloValidate   = true;
    bool   overfitDetect = true;
    std::string exeName  = "ChessEngine.exe";
    std::string pyScript = "train_nnue.py";
    std::string dataDir  = "assets";
    std::string modelDir = "assets";

    // Variant-aware helpers
    std::string weightsBaseName() const {
        return (variant == ChessVariant::DuckChess) ? "duck_nnue_weights" : "nnue_weights";
    }
    std::string trainingDataName() const {
        return (variant == ChessVariant::DuckChess) ? "duck_training_data.bin" : "training_data.bin";
    }
    std::string selfplayPrefix() const {
        return (variant == ChessVariant::DuckChess) ? "duck_selfplay_gen" : "selfplay_gen";
    }
    std::string variantFlag() const {
        return (variant == ChessVariant::DuckChess) ? " --duck-chess" : "";
    }
};

// -- Preset definition --
struct Preset {
    std::string name;
    bool isBuiltin = false;
    int    generations   = 10;
    int    gamesPerGen   = 5000;
    int    epochsPerGen  = 10;
    int    batchSize     = 2048;
    double lr            = 0.001;
    double weightDecay   = 1e-5;
    double dropout       = 0.1;
    double labelSmooth   = 0.05;
    int    gradAccum     = 4;
    int    warmupSteps   = 500;
    double drawWeight    = 0.25;
    double mateBoost     = 3.0;
    double splRatio      = 0.4;
    int    workers       = 12;
    int    depth         = 5;
    int    startGen      = 0;
    bool   eloValidate   = false;
    bool   overfitDetect = true;
};


struct TrainPoint {
    int    gen   = 0;
    int    step  = 0;
    double train = 0.0;
    double val   = 0.0;
    bool   hasVal= false;
    double accuracy = 0.0;
    bool   hasAcc = false;
    double lr = 0.0;
    bool   hasLR = false;
};

// Compute a "signature" for a log line by replacing digit runs with '#'.
// Two lines with the same signature are considered running updates of each other.
static std::string logSignature(const std::string& s) {
    std::string sig;
    sig.reserve(s.size());
    bool inNum = false;
    for (char c : s) {
        if (std::isdigit((unsigned char)c)) {
            if (!inNum) { sig += '#'; inNum = true; }
        } else {
            sig += c;
            inNum = false;
        }
    }
    return sig;
}

struct AppState {
    std::mutex              mtx;
    std::vector<TrainPoint> pts;
    std::deque<std::string> log;
    std::string             lastLogSig;       // signature of last log line
    bool                    logLastReplaced = false; // true when last line was overwritten
    std::string             status   = "Ready";
    std::string             phase;
    int    curGen    = 0, totalGens  = 0;
    int    curEpoch  = 0, totalEpochs= 0;
    double lastTrain = 0.0, lastVal  = 0.0;
    int    lastElo   = 0;
    std::chrono::steady_clock::time_point pipelineStart;
    std::chrono::steady_clock::time_point phaseStart;
    int completedGens = 0;
    int batchEtaSec  = 0;   // countdown: batch ETA from Python ("ETA: Xs")
    int epochEtaSec  = 0;   // countdown: gen training ETA from Python ("Total ETA: Xs")
    int nextEpochSec = 0;   // countdown: next epoch ETA from Python ("Next: ~Xs")
    std::chrono::steady_clock::time_point batchEtaStamp;
    std::chrono::steady_clock::time_point epochEtaStamp;
    std::chrono::steady_clock::time_point nextEpochStamp;
    bool   running   = false;
    std::atomic<bool> stopFlag{false};

    static const size_t MAX_LOG = 800;

    void pushLog(const std::string& s) {
        std::lock_guard<std::mutex> lk(mtx);
        std::string sig = logSignature(s);
        // If the new line is a running update of the last line (same structure,
        // only numbers differ), replace it instead of appending.
        if (!log.empty() && sig == lastLogSig) {
            log.back() = s;
            logLastReplaced = true;
        } else {
            log.push_back(s);
            if (log.size() > MAX_LOG) log.pop_front();
        }
        lastLogSig = sig;
    }
    void setStatus(const std::string& s) { std::lock_guard<std::mutex> lk(mtx); status = s; }
    void setPhase (const std::string& s) { std::lock_guard<std::mutex> lk(mtx); phase  = s; }
    void pushPt(TrainPoint p) {
        std::lock_guard<std::mutex> lk(mtx);
        pts.push_back(p);
        lastTrain = p.train;
        if (p.hasVal) lastVal = p.val;
    }
};

// ── Globals ───────────────────────────────────────────────────────
static AppState   g_st;
static Config     g_cfg;
static HINSTANCE  g_hInst    = nullptr;
static HWND       g_hWnd     = nullptr;
static HWND       g_hGraph   = nullptr;
static HWND       g_hLog     = nullptr;
static HWND       g_hStatus  = nullptr;
static HWND       g_hProg    = nullptr;
static HWND       g_hBanner  = nullptr;
static HWND       g_hStart   = nullptr;
static HWND       g_hStop    = nullptr;
static HWND       g_hPause   = nullptr;
static HWND       g_hPreset  = nullptr;
static HWND       g_hCfgPane = nullptr;
static HFONT      g_fUI      = nullptr;
static HFONT      g_fMono    = nullptr;
static HBRUSH     g_brPanel  = nullptr;
static HBRUSH     g_brBg     = nullptr;
static ULONG_PTR  g_gdip     = 0;
static std::thread g_worker;
static std::map<int,HWND> g_edits;
static HWND g_hChkElo = nullptr, g_hChkOvfit = nullptr;

// -- Preset system globals --
static std::vector<Preset> g_allPresets;
static int g_currentPresetIdx = 1;
static HWND g_hBtnSave = nullptr, g_hBtnDel = nullptr;

// -- Variant selector --
static HWND g_hVariant = nullptr;

// -- Graph toggle globals --
static bool g_showLoss = true;
static bool g_showAcc  = false;
static bool g_showLR   = false;
static HWND g_hChkGLoss = nullptr, g_hChkGAcc = nullptr, g_hChkGLR = nullptr;

// -- Graph hover tracking --
static int   g_graphHoverIdx = -1;
static POINT g_graphMousePt  = {-1, -1};

// -- Log color classification --
enum LogColor {
    LC_GENERAL  = 0,
    LC_SELFPLAY = 1,
    LC_TRAINING = 2,
    LC_SUCCESS  = 3,
    LC_ERROR    = 4,
    LC_PROGRESS = 5,
    LC_META     = 6,
};

static COLORREF LogColorTable[] = {
    RGB(255,255,255),
    RGB(0,139,139),
    RGB(255,139,0),
    RGB(0,255,0),
    RGB(255,0,0),
    RGB(255,215,0),
    RGB(135,206,235),
};



// Active process tracking for stop/pause
static std::atomic<DWORD> g_activePid{0};
static std::atomic<bool>  g_pauseFlag{false};
static PROCESS_INFORMATION g_activePi{};
static std::mutex          g_activePiMtx;

// ── Utilities ─────────────────────────────────────────────────────
static std::wstring W(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring r(n - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &r[0], n);
    return r;
}
static std::string N(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string r(n - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &r[0], n, nullptr, nullptr);
    return r;
}
static std::string getEdit(HWND h) {
    int len = GetWindowTextLengthW(h) + 1;
    std::wstring w(len, 0); GetWindowTextW(h, &w[0], len);
    w.resize(wcslen(w.c_str())); return N(w);
}
static void setEdit(HWND h, const std::string& s) { SetWindowTextW(h, W(s).c_str()); }
static double pDbl(const std::string& s, double d=0){ try{return std::stod(s);}catch(...){return d;} }
static int    pInt (const std::string& s, int    d=0){ try{return std::stoi(s);}catch(...){return d;} }

static std::string exeDir() {
    wchar_t b[MAX_PATH]; GetModuleFileNameW(nullptr, b, MAX_PATH);
    std::wstring p(b); auto i = p.find_last_of(L"\\/");
    return N(i != std::wstring::npos ? p.substr(0,i) : L".");
}
static std::string dbl2s(double v, int p=6) {
    std::ostringstream ss; ss << std::setprecision(p) << v; return ss.str();
}

// -- Preset management --
static std::string presetFilePath() {
    return exeDir() + "\\custom_presets.cfg";
}

static void InitBuiltinPresets() {
    g_allPresets.clear();
    {
        Preset p;
        p.name = "Quick Test"; p.isBuiltin = true;
        p.generations = 3; p.gamesPerGen = 500; p.epochsPerGen = 3;
        p.batchSize = 1024; p.lr = 0.001; p.weightDecay = 1e-5;
        p.dropout = 0.0; p.labelSmooth = 0.0; p.gradAccum = 2;
        p.warmupSteps = 100; p.drawWeight = 0.25; p.mateBoost = 3.0;
        p.splRatio = 0.4; p.workers = 12; p.depth = 4; p.startGen = 0;
        p.eloValidate = false; p.overfitDetect = false;
        g_allPresets.push_back(p);
    }
    {
        Preset p;
        p.name = "Standard"; p.isBuiltin = true;
        p.generations = 10; p.gamesPerGen = 5000; p.epochsPerGen = 10;
        p.batchSize = 2048; p.lr = 0.001; p.weightDecay = 1e-5;
        p.dropout = 0.1; p.labelSmooth = 0.05; p.gradAccum = 4;
        p.warmupSteps = 500; p.drawWeight = 0.25; p.mateBoost = 3.0;
        p.splRatio = 0.4; p.workers = 12; p.depth = 5; p.startGen = 0;
        p.eloValidate = false; p.overfitDetect = true;
        g_allPresets.push_back(p);
    }
    {
        Preset p;
        p.name = "High Quality"; p.isBuiltin = true;
        p.generations = 30; p.gamesPerGen = 20000; p.epochsPerGen = 20;
        p.batchSize = 4096; p.lr = 0.0005; p.weightDecay = 1e-5;
        p.dropout = 0.15; p.labelSmooth = 0.05; p.gradAccum = 8;
        p.warmupSteps = 1000; p.drawWeight = 0.25; p.mateBoost = 3.0;
        p.splRatio = 0.35; p.workers = 12; p.depth = 6; p.startGen = 0;
        p.eloValidate = false; p.overfitDetect = true;
        g_allPresets.push_back(p);
    }
}

static void LoadCustomPresets() {
    std::string path = presetFilePath();
    std::ifstream f(path);
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Preset p; p.isBuiltin = false;
        auto next = [&]() -> std::string {
            std::string t;
            if (std::getline(ss, t, '|')) return t;
            return "";
        };
        p.name = next(); if (p.name.empty()) continue;
        p.generations  = pInt(next(), 10);
        p.gamesPerGen  = pInt(next(), 5000);
        p.epochsPerGen = pInt(next(), 10);
        p.batchSize    = pInt(next(), 2048);
        p.workers      = pInt(next(), 12);
        p.depth        = pInt(next(), 5);
        p.gradAccum    = pInt(next(), 4);
        p.warmupSteps  = pInt(next(), 500);
        p.lr           = pDbl(next(), 0.001);
        p.weightDecay  = pDbl(next(), 1e-5);
        p.dropout      = pDbl(next(), 0.1);
        p.labelSmooth  = pDbl(next(), 0.05);
        p.drawWeight   = pDbl(next(), 0.25);
        p.mateBoost    = pDbl(next(), 3.0);
        p.splRatio     = pDbl(next(), 0.4);
        p.startGen     = pInt(next(), 0);
        p.eloValidate  = pInt(next(), 0) != 0;
        p.overfitDetect= pInt(next(), 1) != 0;
        g_allPresets.push_back(p);
    }
}

static void SaveCustomPresets() {
    std::string path = presetFilePath();
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "# Custom presets\n";
    for (auto& p : g_allPresets) {
        if (p.isBuiltin) continue;
        f << p.name << "|"
          << p.generations << "|" << p.gamesPerGen << "|" << p.epochsPerGen << "|"
          << p.batchSize << "|" << p.workers << "|" << p.depth << "|"
          << p.gradAccum << "|" << p.warmupSteps << "|"
          << dbl2s(p.lr,8) << "|" << dbl2s(p.weightDecay,8) << "|"
          << dbl2s(p.dropout,4) << "|" << dbl2s(p.labelSmooth,4) << "|"
          << dbl2s(p.drawWeight,4) << "|" << dbl2s(p.mateBoost,4) << "|"
          << dbl2s(p.splRatio,4) << "|" << p.startGen << "|"
          << (p.eloValidate?1:0) << "|" << (p.overfitDetect?1:0) << "\n";
    }
}

static void PopulatePresetCombo() {
    if (!g_hPreset) return;
    SendMessageW(g_hPreset, CB_RESETCONTENT, 0, 0);
    for (auto& p : g_allPresets)
        SendMessageW(g_hPreset, CB_ADDSTRING, 0, (LPARAM)W(p.name).c_str());
    if (g_currentPresetIdx >= 0 && g_currentPresetIdx < (int)g_allPresets.size())
        SendMessageW(g_hPreset, CB_SETCURSEL, g_currentPresetIdx, 0);
}

// -- Log line classification --
static LogColor ClassifyLogLine(const std::string& line) {
    if (line.find("[ERR]") != std::string::npos ||
        line.find("[ERROR]") != std::string::npos ||
        line.find("[WARN]") != std::string::npos ||
        line.find("failed") != std::string::npos ||
        line.find("Failed") != std::string::npos ||
        line.find("Error") != std::string::npos)
        return LC_ERROR;

    if (line.find("complete") != std::string::npos ||
        line.find("Complete") != std::string::npos ||
        line.find("done") != std::string::npos ||
        line.find("Done") != std::string::npos ||
        line.find("success") != std::string::npos ||
        line.find("Success") != std::string::npos ||
        line.find("saved") != std::string::npos ||
        line.find("=== Pipeline") != std::string::npos)
        return LC_SUCCESS;

    if (line.find("[SelfPlay]") != std::string::npos ||
        line.find("self-play") != std::string::npos ||
        line.find("Self-play") != std::string::npos ||
        line.find("--generate") != std::string::npos)
        return LC_SELFPLAY;

    if (line.find("Train Loss") != std::string::npos ||
        line.find("Train:") != std::string::npos ||
        line.find("Val Loss") != std::string::npos ||
        line.find("Val:") != std::string::npos ||
        line.find("Epoch") != std::string::npos ||
        line.find("train_nnue") != std::string::npos ||
        line.find("LR:") != std::string::npos)
        return LC_TRAINING;

    if (line.find("---") != std::string::npos ||
        line.find("===") != std::string::npos ||
        line.find("Generation") != std::string::npos ||
        line.find("Gen ") != std::string::npos ||
        line.find("ELO") != std::string::npos ||
        line.find("Elo") != std::string::npos)
        return LC_PROGRESS;

    if (line.find("[CMD]") != std::string::npos ||
        line.find("ETA") != std::string::npos ||
        line.find("elapsed") != std::string::npos ||
        line.find("Time") != std::string::npos)
        return LC_META;

    return LC_GENERAL;
}


// ── Process thread helpers ────────────────────────────────────────
static void SuspendProcessThreads(DWORD pid) {
    if (pid == 0) return;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te{}; te.dwSize = sizeof(te);
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid) {
                HANDLE ht = OpenThread(THREAD_SUSPEND_RESUME, FALSE, te.th32ThreadID);
                if (ht) { SuspendThread(ht); CloseHandle(ht); }
            }
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
}

static void ResumeProcessThreads(DWORD pid) {
    if (pid == 0) return;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te{}; te.dwSize = sizeof(te);
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid) {
                HANDLE ht = OpenThread(THREAD_SUSPEND_RESUME, FALSE, te.th32ThreadID);
                if (ht) { ResumeThread(ht); CloseHandle(ht); }
            }
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
}

static void SuspendOrTerminateActive() {
    DWORD pid = g_activePid.load();
    if (pid == 0) return;
    // Send CTRL_BREAK_EVENT to process group
    GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid);
    // Wait up to 3000ms
    HANDLE hProc = OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, pid);
    if (hProc) {
        if (WaitForSingleObject(hProc, 3000) != WAIT_OBJECT_0) {
            TerminateProcess(hProc, 1);
        }
        CloseHandle(hProc);
    }
}

// ── Subprocess ────────────────────────────────────────────────────
static bool RunProc(const std::wstring& cmd, const std::string& dir,
                    std::function<void(const std::string&)> cb,
                    std::atomic<bool>& stop)
{
    SECURITY_ATTRIBUTES sa{}; sa.nLength = sizeof(sa); sa.bInheritHandle = TRUE;
    HANDLE hR, hW;
    if (!CreatePipe(&hR, &hW, &sa, 0)) return false;
    SetHandleInformation(hR, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{}; si.cb = sizeof(si);
    si.hStdOutput = hW; si.hStdError = hW;
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    PROCESS_INFORMATION pi{};
    std::wstring c = cmd;
    std::wstring wd = W(dir);
    // Log the exact command so failures are diagnosable
    { std::string narrow; narrow.reserve(c.size());
      for (wchar_t wc : c) narrow += static_cast<char>(wc);
      cb("[CMD] " + narrow); }

    bool ok = CreateProcessW(nullptr, c.data(), nullptr, nullptr, TRUE,
                             CREATE_NEW_PROCESS_GROUP, nullptr,
                             wd.empty() ? nullptr : wd.c_str(), &si, &pi) != 0;
    CloseHandle(hW);
    if (!ok) {
        DWORD err = GetLastError();
        cb("[ERR] CreateProcess failed (Win32 error " + std::to_string(err) + ") - exe not found or access denied");
        CloseHandle(hR); return false;
    }

    // Store active PID so Stop/Pause buttons can act on it immediately
    g_activePid.store(pi.dwProcessId);
    {
        std::lock_guard<std::mutex> lk(g_activePiMtx);
        g_activePi = pi;
    }

    std::string buf; char ch[4096]; DWORD br = 0;
    while (ReadFile(hR, ch, (DWORD)(sizeof(ch)-1), &br, nullptr) && br > 0) {
        ch[br] = '\0'; buf += ch;
        size_t p;
        while ((p = buf.find('\n')) != std::string::npos) {
            std::string ln = buf.substr(0, p);
            buf = buf.substr(p+1);
            // Strip trailing \r (CRLF line ending)
            if (!ln.empty() && ln.back()=='\r') ln.pop_back();
            // Handle interior \r (progress overwrite): keep text after last \r
            auto cr = ln.rfind('\r');
            if (cr != std::string::npos) ln = ln.substr(cr+1);
            if (!ln.empty()) cb(ln);
        }
        if (stop.load()) {
            GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pi.dwProcessId);
            WaitForSingleObject(pi.hProcess, 3000);
            TerminateProcess(pi.hProcess, 1);
            break;
        }
    }
    if (!buf.empty()) {
        if (buf.back()=='\r') buf.pop_back();
        auto cr = buf.rfind('\r');
        if (cr != std::string::npos) buf = buf.substr(cr+1);
        if (!buf.empty()) cb(buf);
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ex=1; GetExitCodeProcess(pi.hProcess, &ex);
    if (ex != 0) cb("[ERR] Process exited with code " + std::to_string(ex));
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(hR);

    // Clear active PID
    g_activePid.store(0);
    {
        std::lock_guard<std::mutex> lk(g_activePiMtx);
        g_activePi = {};
    }

    return ex == 0 && !stop.load();
}

// ── Loss parser ───────────────────────────────────────────────────
static bool ParseLoss(const std::string& line, TrainPoint& pt) {
    auto findNum = [&](const std::string& key) -> double {
        auto i = line.find(key);
        if (i == std::string::npos) return -1.0;
        i += key.size();
        while (i < line.size() && (line[i]==' '||line[i]==':')) i++;
        try { size_t n; return std::stod(line.substr(i), &n); } catch(...){ return -1.0; }
    };
    auto findInt = [&](const std::string& key) -> int {
        auto i = line.find(key);
        if (i == std::string::npos) return -1;
        i += key.size();
        while (i < line.size() && (line[i]==' '||line[i]==':')) i++;
        try { size_t n; return std::stoi(line.substr(i), &n); } catch(...){ return -1; }
    };
    double tl = findNum("Train Loss");
    if (tl < 0) tl = findNum("train_loss");
    if (tl < 0) tl = findNum("Train:");   // train_nnue.py: "  Epoch N/M | Train: 0.123456 | Val: ..."
    if (tl < 0) return false;
    pt.train  = tl;
    int ep    = findInt("Epoch"); if (ep >= 0) pt.step = ep;
    int st2   = findInt("Step");  if (st2>= 0) pt.step = st2;
    double vl = findNum("Val Loss");
    if (vl < 0) vl = findNum("val_loss");
    if (vl < 0) vl = findNum("Val:");     // train_nnue.py: "... | Val: 0.234567 * | LR: ..."
    pt.hasVal = vl >= 0;
    pt.val    = pt.hasVal ? vl : 0.0;
    double lrv = findNum("LR");
    if (lrv < 0) lrv = findNum("lr");
    pt.hasLR = lrv >= 0;
    pt.lr    = pt.hasLR ? lrv : 0.0;
    double acc = findNum("Acc");
    if (acc < 0) acc = findNum("accuracy");
    pt.hasAcc = acc >= 0;
    pt.accuracy = pt.hasAcc ? acc : 0.0;
    return true;
}

// ── Pipeline ──────────────────────────────────────────────────────
static bool SelfPlay(const Config& cfg, int gen) {
    std::string d = exeDir();
    fs::path assetsDir = fs::path(d)/cfg.dataDir;
    fs::create_directories(assetsDir);
    fs::path outFile = assetsDir/(cfg.selfplayPrefix()+std::to_string(gen)+".bin");
    std::string weightsArg;
    if (gen > cfg.startGen + 1) {
        fs::path prev = assetsDir/(cfg.weightsBaseName()+"_gen"+std::to_string(gen-1)+".bin");
        if (fs::exists(prev)) weightsArg = " --weights \""+prev.string()+"\"";
    }
    std::string variantLabel = (cfg.variant == ChessVariant::DuckChess) ? "Duck Chess" : "Standard";
    std::wstring cmd = W(
        "\"" + (fs::path(d)/cfg.exeName).string() + "\""
        " --generate --games " + std::to_string(cfg.gamesPerGen) +
        " --depth " + std::to_string(cfg.depth) +
        " --workers " + std::to_string(cfg.workers) +
        " --output \"" + outFile.string() + "\"" +
        weightsArg + cfg.variantFlag()
    );
    g_st.setStatus("Gen "+std::to_string(gen)+": "+variantLabel+" self-play ("+std::to_string(cfg.gamesPerGen)+" games)");
    g_st.setPhase("selfplay");
    g_st.phaseStart = std::chrono::steady_clock::now();
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curEpoch=0; g_st.totalEpochs=cfg.gamesPerGen; }
    return RunProc(cmd, d, [&](const std::string& ln){
        g_st.pushLog(ln);
        // SelfPlayGen outputs: "\r[SelfPlay] done/total  pos=..."
        // Parse the leading integer after "[SelfPlay] " as games completed.
        auto p = ln.find("[SelfPlay] ");
        if (p != std::string::npos) {
            try { size_t n; int g2=std::stoi(ln.substr(p+11), &n);
                std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curEpoch=g2; }
            catch(...) {}
        }
        // Legacy fallback: some older builds printed "Games: N"
        if (p == std::string::npos) {
            auto q = ln.find("Games:");
            if (q != std::string::npos) {
                try { int g2=std::stoi(ln.substr(q+6));
                    std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curEpoch=g2; }
                catch(...) {}
            }
        }
    }, g_st.stopFlag);
}

static bool Training(const Config& cfg, int gen) {
    std::string d = exeDir();
    fs::path assetsDir = fs::path(d)/cfg.dataDir;
    fs::create_directories(assetsDir);
    fs::path baseData     = assetsDir/cfg.trainingDataName();
    fs::path selfplayData = assetsDir/(cfg.selfplayPrefix()+std::to_string(gen)+".bin");
    fs::path prevWeights  = assetsDir/(cfg.weightsBaseName()+"_gen"+std::to_string(gen-1)+".bin");
    fs::path outputWeights= assetsDir/(cfg.weightsBaseName()+".bin");
    fs::path genWeights   = assetsDir/(cfg.weightsBaseName()+"_gen"+std::to_string(gen)+".bin");
    std::string args =
        " --data \""        + baseData.string() + "\""
        " --extra-data \""  + selfplayData.string() + "\" " + dbl2s(cfg.splRatio) +
        " --max-positions " + std::to_string(cfg.maxPositions) +
        " --epochs "        + std::to_string(cfg.epochsPerGen) +
        " --batch-size "    + std::to_string(cfg.batchSize) +
        " --lr "            + dbl2s(cfg.lr, 8) +
        " --lr-min 0.00001"
        " --weight-decay "  + dbl2s(cfg.weightDecay, 8) +
        " --dropout "       + dbl2s(cfg.dropout) +
        " --grad-clip 1.0"
        " --grad-accum "    + std::to_string(cfg.gradAccum) +
        " --warmup-steps "  + std::to_string(cfg.warmupSteps) +
        " --draw-weight "   + dbl2s(cfg.drawWeight) +
        " --mate-boost "    + dbl2s(cfg.mateBoost) +
        " --early-stop "    + std::to_string(cfg.earlyStop) +
        " --enhanced --swa --no-cosine-restarts --plot";
    if (cfg.labelSmooth > 0.0)
        args += " --label-smoothing " + dbl2s(cfg.labelSmooth);
    if (fs::exists(prevWeights))
        args += " --load-weights \"" + prevWeights.string() + "\"";
    args += " --output \"" + outputWeights.string() + "\"";
    std::wstring cmd = W("py -3.10 -u \"" + (fs::path(d)/cfg.pyScript).string() + "\"" + args);
    std::string variantLabel = (cfg.variant == ChessVariant::DuckChess) ? "Duck Chess" : "Standard";
    g_st.setStatus("Gen "+std::to_string(gen)+": "+variantLabel+" training ("+std::to_string(cfg.epochsPerGen)+" epochs)");
    g_st.setPhase("training");
    g_st.phaseStart = std::chrono::steady_clock::now();
    { std::lock_guard<std::mutex> lk(g_st.mtx);
      g_st.curEpoch=0; g_st.totalEpochs=cfg.epochsPerGen;
      g_st.batchEtaSec=0; g_st.epochEtaSec=0; g_st.nextEpochSec=0; }
    // Helper: parse duration string like "5s", "1m 30s", "1h 5m 30s" -> seconds
    auto parseEtaSecs = [](const std::string& s) -> int {
        int total = 0;
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && s[i] == ' ') i++;
            if (i >= s.size() || !std::isdigit((unsigned char)s[i])) break;
            int val = 0;
            while (i < s.size() && std::isdigit((unsigned char)s[i])) { val = val*10 + (s[i]-'0'); i++; }
            while (i < s.size() && s[i] == ' ') i++;
            if (i < s.size()) {
                char u = s[i++];
                if (u=='h') total += val*3600;
                else if (u=='m') total += val*60;
                else if (u=='s') total += val;
            }
        }
        return total;
    };
    // Helper: strip ANSI escape sequences and a named token " | KEY: VALUE" from a line
    // Returns cleaned line and optionally sets *outSecs if key found
    auto stripToken = [](std::string line, const std::string& key) -> std::pair<std::string,std::string> {
        // Strip ANSI escapes first (\x1b[...m and \x1b[K etc.)
        std::string clean;
        clean.reserve(line.size());
        for (size_t i = 0; i < line.size(); ) {
            if (line[i] == '\x1b' && i+1 < line.size() && line[i+1] == '[') {
                i += 2;
                while (i < line.size() && (std::isdigit((unsigned char)line[i]) || line[i]==';')) i++;
                if (i < line.size()) i++; // skip final letter
            } else if (line[i] == '\x1b') {
                i++;
            } else {
                clean += line[i++];
            }
        }
        // Also strip bare "[K" left-overs (cursor erase without ESC prefix)
        {
            std::string tmp;
            tmp.reserve(clean.size());
            for (size_t i = 0; i < clean.size(); ) {
                if (i+1 < clean.size() && clean[i]=='[' && clean[i+1]=='K') { i+=2; }
                else tmp += clean[i++];
            }
            clean = tmp;
        }
        // Find and extract the value after key
        std::string val;
        auto pos = clean.find(key);
        if (pos != std::string::npos) {
            size_t vs = pos + key.size();
            // skip spaces/colons
            while (vs < clean.size() && (clean[vs]==' ' || clean[vs]==':' || clean[vs]=='~')) vs++;
            size_t ve = vs;
            // value ends at " |" or " [" or end
            while (ve < clean.size() && !(clean[ve]=='|' && ve>0 && clean[ve-1]==' ') && !(clean[ve]=='[' && ve>0 && clean[ve-1]==' ')) ve++;
            // trim trailing spaces
            while (ve > vs && clean[ve-1]==' ') ve--;
            val = clean.substr(vs, ve-vs);
            // remove " | KEY ..." section from line: find " | KEY" or just "KEY" at start
            size_t removeFrom = (pos >= 3 && clean.substr(pos-3,3)==" | ") ? pos-3 : pos;
            size_t removeTo   = ve;
            // also eat trailing " |" if present
            if (removeTo < clean.size()-1 && clean.substr(removeTo,2)==" |") removeTo += 2;
            clean.erase(removeFrom, removeTo - removeFrom);
            // trim trailing whitespace and trailing " |"
            while (!clean.empty() && (clean.back()==' ' || clean.back()=='|')) clean.pop_back();
        }
        return {clean, val};
    };

    bool ok = RunProc(cmd, d, [&](const std::string& ln){
        // Strip ANSI + extract ETA fields before logging
        std::string cleaned = ln;
        std::string batchEtaStr, epochEtaStr, nextEpochStr;

        // "Total ETA: Xs" appears on the epoch summary line
        {   auto [c,v] = stripToken(cleaned, "Total ETA");
            if (!v.empty()) { epochEtaStr = v; cleaned = c; }
        }
        // "Next: ~Xs" appears on the epoch summary line
        {   auto [c,v] = stripToken(cleaned, "Next");
            if (!v.empty()) { nextEpochStr = v; cleaned = c; }
        }
        // "ETA: Xs" appears on batch-level lines
        {   auto [c,v] = stripToken(cleaned, "ETA");
            if (!v.empty()) { batchEtaStr = v; cleaned = c; }
        }

        // Store parsed ETAs into state
        if (!batchEtaStr.empty() || !epochEtaStr.empty() || !nextEpochStr.empty()) {
            auto now = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lk(g_st.mtx);
            if (!batchEtaStr.empty()) {
                g_st.batchEtaSec   = parseEtaSecs(batchEtaStr);
                g_st.batchEtaStamp = now;
            }
            if (!epochEtaStr.empty()) {
                g_st.epochEtaSec   = parseEtaSecs(epochEtaStr);
                g_st.epochEtaStamp = now;
            }
            if (!nextEpochStr.empty()) {
                g_st.nextEpochSec   = parseEtaSecs(nextEpochStr);
                g_st.nextEpochStamp = now;
            }
        }

        // Log the cleaned line (ETA fields removed)
        g_st.pushLog(cleaned);

        TrainPoint pt; pt.gen = gen;
        if (ParseLoss(ln, pt)) {
            g_st.pushPt(pt);
            std::lock_guard<std::mutex> lk(g_st.mtx);
            if (pt.step > 0) g_st.curEpoch = pt.step;
        }
    }, g_st.stopFlag);
    if (ok) {
        std::error_code ec;
        fs::copy_file(outputWeights, genWeights, fs::copy_options::overwrite_existing, ec);
        if (ec) g_st.pushLog("[WARN] Failed to copy weights to " + genWeights.string() + ": " + ec.message());
    }
    return ok;
}

static void EloVal(const Config& cfg, int gen) {
    if (!cfg.eloValidate || gen <= cfg.startGen + 1) return;
    std::string d = exeDir();
    fs::path assetsDir = fs::path(d)/cfg.dataDir;
    fs::path newWt  = assetsDir/(cfg.weightsBaseName()+"_gen"+std::to_string(gen)+".bin");
    fs::path prevWt = assetsDir/(cfg.weightsBaseName()+"_gen"+std::to_string(gen-1)+".bin");
    if (!fs::exists(newWt)||!fs::exists(prevWt)) return;
    fs::path cutechess = fs::path(d)/"cutechess"/"cutechess-cli.exe";
    if (!fs::exists(cutechess)) {
        g_st.pushLog("[WARN] cutechess-cli.exe not found, skipping ELO validation");
        return;
    }
    fs::path pgnOut = assetsDir/("validation_gen"+std::to_string(gen)+".pgn");
    std::wstring cmd = W(
        "\"" + cutechess.string() + "\""
        " -engine name=Gen" + std::to_string(gen) + " cmd=uci_engine.bat"
        " option.WeightsFile=\"" + newWt.string() + "\""
        " -engine name=Gen" + std::to_string(gen-1) + " cmd=uci_engine.bat"
        " option.WeightsFile=\"" + prevWt.string() + "\""
        " -each proto=uci tc=1+0.1"
        " -rounds 50"
        " -pgnout \"" + pgnOut.string() + "\""
        " -recover"
    );
    g_st.setStatus("Gen "+std::to_string(gen)+": ELO validation");
    g_st.setPhase("elo");
    g_st.phaseStart = std::chrono::steady_clock::now();
    RunProc(cmd, d, [&](const std::string& ln){
        g_st.pushLog(ln);
        for (auto key : {"ELO:","Elo:","Elo "}) {
            auto p = ln.find(key);
            if (p != std::string::npos) {
                size_t numStart = p + std::string(key).size();
                while (numStart < ln.size() && (ln[numStart]==' '||ln[numStart]==':'||ln[numStart]=='+')) numStart++;
                try { int e=std::stoi(ln.substr(numStart));
                    std::lock_guard<std::mutex> lk(g_st.mtx); g_st.lastElo=e; }
                catch(...) {}
            }
        }
    }, g_st.stopFlag);
}

static void PipelineThread(Config cfg) {
    g_st.stopFlag.store(false);
    g_pauseFlag.store(false);
    int firstGen = cfg.startGen + 1;
    int lastGen  = cfg.startGen + cfg.generations;
    { std::lock_guard<std::mutex> lk(g_st.mtx);
      g_st.running=true; g_st.curGen=0; g_st.totalGens=cfg.generations;
      g_st.pipelineStart = std::chrono::steady_clock::now();
      g_st.phaseStart = g_st.pipelineStart;
      g_st.completedGens = 0;
      g_st.pts.clear(); }
    std::string variantName = (cfg.variant == ChessVariant::DuckChess) ? "Duck Chess" : "Standard Chess";
    g_st.pushLog("=== Pipeline start: "+variantName+" | "+std::to_string(cfg.generations)+" generations (gen "+std::to_string(firstGen)+" to "+std::to_string(lastGen)+") ===");

    for (int gen=firstGen; gen<=lastGen && !g_st.stopFlag.load(); gen++) {
        { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curGen=gen-firstGen; }
        g_st.pushLog("--- Generation "+std::to_string(gen)+" ---");
        if (!SelfPlay(cfg, gen))  { g_st.pushLog("[ERR] Self-play failed gen "+std::to_string(gen)); break; }
        if (g_st.stopFlag.load()) break;
        if (!Training(cfg, gen))  { g_st.pushLog("[ERR] Training failed gen "+std::to_string(gen)); break; }
        if (g_st.stopFlag.load()) break;
        EloVal(cfg, gen);
        { std::lock_guard<std::mutex> lk(g_st.mtx);
          std::ostringstream ss; ss<<std::fixed<<std::setprecision(5);
          ss<<"Gen "<<gen<<" done | train="<<g_st.lastTrain<<" val="<<g_st.lastVal<<" ELO="<<g_st.lastElo;
          g_st.log.push_back(ss.str()); g_st.completedGens++; }
    }
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.running=false; }
    g_st.setStatus(g_st.stopFlag.load() ? "Stopped." : "Pipeline complete!");
    g_st.pushLog("=== Pipeline "+std::string(g_st.stopFlag.load()?"stopped":"complete")+" ===");
    PostMessage(g_hWnd, WM_USER+1, 0, 0);
}

// ── Graph drawing ─────────────────────────────────────────────────
static LRESULT CALLBACK GraphProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp);

static void DrawGraph(HWND hw) {
    RECT rc; GetClientRect(hw, &rc);
    int W2 = rc.right, H2 = rc.bottom;
    PAINTSTRUCT ps; HDC hdc = BeginPaint(hw, &ps);
    HDC memDC = CreateCompatibleDC(hdc);
    HBITMAP bmp = CreateCompatibleBitmap(hdc, W2, H2);
    SelectObject(memDC, bmp);

    Graphics g(memDC);
    g.SetSmoothingMode(SmoothingModeAntiAlias);
    SolidBrush bgBr(Color(255,16,16,24));
    g.FillRectangle(&bgBr, 0, 0, W2, H2);

    std::vector<TrainPoint> pts;
    { std::lock_guard<std::mutex> lk(g_st.mtx); pts = g_st.pts; }

    if (pts.size() < 2) {
        Font fnt(L"Segoe UI", 10.0f);
        SolidBrush tb(Color(255,80,80,100));
        g.DrawString(L"No data yet", -1, &fnt, PointF((float)W2/2-40,(float)H2/2-8), &tb);
        BitBlt(hdc,0,0,W2,H2,memDC,0,0,SRCCOPY);
        DeleteObject(bmp); DeleteDC(memDC);
        EndPaint(hw, &ps); return;
    }

    // Weighted panel sizing: Loss gets 3 shares, Acc and LR get 1 share each
    float lossWeight = g_showLoss ? 3.0f : 0.0f;
    float accWeight  = g_showAcc  ? 1.0f : 0.0f;
    float lrWeight   = g_showLR   ? 1.0f : 0.0f;
    float totalWeight = lossWeight + accWeight + lrWeight;
    if (totalWeight < 0.01f) { lossWeight = 1.0f; totalWeight = 1.0f; }  // fallback

    int numPanels = (g_showLoss?1:0) + (g_showAcc?1:0) + (g_showLR?1:0);
    if (numPanels == 0) numPanels = 1;

    float ml = 52, mr = 16;
    float panelGap = 4.0f;
    float availH = (float)H2 - panelGap * (numPanels - 1);
    float lossH = availH * (lossWeight / totalWeight);
    float accH  = availH * (accWeight  / totalWeight);
    float lrH   = availH * (lrWeight   / totalWeight);
    float curY = 0;
    float gw = (float)W2 - ml - mr;

    auto xf = [&](int i) -> float {
        return ml + (float)i / (float)(pts.size()-1) * gw;
    };

    // ---- Loss panel (3x weight) ----
    if (g_showLoss || totalWeight < 0.01f) {
        float pt_h = (lossWeight > 0) ? lossH : availH;
        float pt_top = curY;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2;
        if (gh2 < 10) gh2 = 10;

        SolidBrush panelBg(Color(255,20,20,30));
        g.FillRectangle(&panelBg, 0.0f, pt_top, (float)W2, pt_h);

        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Loss", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);

        double minV=1e9, maxV=-1e9;
        for (auto& p2 : pts) {
            minV = (std::min)(minV, p2.train); maxV = (std::max)(maxV, p2.train);
            if (p2.hasVal) { minV = (std::min)(minV, p2.val); maxV = (std::max)(maxV, p2.val); }
        }
        if (maxV <= minV) maxV = minV + 0.1;
        double rng = maxV - minV;
        minV -= rng*0.05; maxV += rng*0.05; rng = maxV - minV;
        auto yf = [&](double v) -> float { return mt2 + (float)((maxV-v)/rng)*gh2; };

        Pen gridPen(Color(40,60,60,80), 1.0f);
        Font gridFnt(L"Consolas", 7.0f);
        SolidBrush gridBr(Color(255,80,80,100));
        for (int i=0; i<=4; i++) {
            float y2 = mt2 + gh2*i/4;
            g.DrawLine(&gridPen, ml, y2, ml+gw, y2);
            double val = maxV - rng*i/4;
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(4)<<val;
            g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
        }

        int bestTrainIdx=-1, bestValIdx=-1;
        double bestTrain=1e9, bestVal=1e9;
        for (size_t i=0; i<pts.size(); i++) {
            if (pts[i].train < bestTrain) { bestTrain=pts[i].train; bestTrainIdx=(int)i; }
            if (pts[i].hasVal && pts[i].val < bestVal) { bestVal=pts[i].val; bestValIdx=(int)i; }
        }

        Pen trainPen(Color(255,65,125,245), 1.8f);
        for (size_t i=1; i<pts.size(); i++)
            g.DrawLine(&trainPen, xf((int)i-1), yf(pts[i-1].train), xf((int)i), yf(pts[i].train));

        Pen valPen(Color(255,245,160,60), 1.8f);
        { bool st=false; float px2=0,py2=0;
          for (size_t i=0; i<pts.size(); i++) {
            if (!pts[i].hasVal) continue;
            float cx=xf((int)i), cy=yf(pts[i].val);
            if (st) g.DrawLine(&valPen,px2,py2,cx,cy);
            px2=cx; py2=cy; st=true;
        }}

        // Best train marker
        if (bestTrainIdx >= 0) {
            float bx = xf(bestTrainIdx), by = yf(bestTrain);
            SolidBrush mk(Color(255,65,125,245));
            PointF dm[4] = {{bx,by-5},{bx+5,by},{bx,by+5},{bx-5,by}};
            g.FillPolygon(&mk, dm, 4);
            Font mf(L"Consolas",6.5f);
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(5)<<bestTrain;
            std::wstring lbl = L"Best: " + ss.str();
            float lx2 = bx + 7; if (lx2 + 80 > W2) lx2 = bx - 90;
            g.DrawString(lbl.c_str(),-1,&mf,PointF(lx2,by-5),&mk);
        }
        // Best val marker
        if (bestValIdx >= 0) {
            float bx = xf(bestValIdx), by = yf(bestVal);
            SolidBrush mk(Color(255,245,160,60));
            PointF dm[4] = {{bx,by-5},{bx+5,by},{bx,by+5},{bx-5,by}};
            g.FillPolygon(&mk, dm, 4);
            Font mf(L"Consolas",6.5f);
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(5)<<bestVal;
            std::wstring lbl = L"Best: " + ss.str();
            float lx2 = bx + 7; if (lx2 + 80 > W2) lx2 = bx - 90;
            g.DrawString(lbl.c_str(),-1,&mf,PointF(lx2,by+2),&mk);
        }

        // Legend
        { Font lf(L"Segoe UI",7.5f);
          float lx = (float)W2 - mr - 120, ly = pt_top + 3;
          SolidBrush b1(Color(255,65,125,245)); Pen lp1(Color(255,65,125,245),2);
          g.DrawLine(&lp1,lx,ly+5,lx+12,ly+5);
          g.DrawString(L"train",-1,&lf,PointF(lx+14,ly-1),&b1);
          SolidBrush b2(Color(255,245,160,60)); Pen lp2(Color(255,245,160,60),2);
          g.DrawLine(&lp2,lx+52,ly+5,lx+64,ly+5);
          g.DrawString(L"val",-1,&lf,PointF(lx+66,ly-1),&b2);
        }
        curY += pt_h + panelGap;
    }

    // ---- Accuracy panel ----
    if (g_showAcc) {
        float pt_top = curY, pt_h = accH;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        SolidBrush panelBg(Color(255,18,22,28));
        g.FillRectangle(&panelBg, 0.0f, pt_top, (float)W2, pt_h);
        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Accuracy", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);

        bool hasAnyAcc = false;
        for (auto& p2 : pts) if (p2.hasAcc) { hasAnyAcc = true; break; }

        if (!hasAnyAcc) {
            Font nf(L"Segoe UI", 9.0f); SolidBrush nb(Color(255,80,80,100));
            g.DrawString(L"No accuracy data", -1, &nf, PointF(ml+gw/2-50, mt2+gh2/2-6), &nb);
        } else {
            double minA=1e9, maxA=-1e9;
            for (auto& p2 : pts) {
                if (!p2.hasAcc) continue;
                minA=(std::min)(minA,p2.accuracy); maxA=(std::max)(maxA,p2.accuracy);
            }
            if (maxA<=minA) maxA=minA+0.1;
            double rng=maxA-minA; minA-=rng*0.05; maxA+=rng*0.05; rng=maxA-minA;
            auto yf=[&](double v)->float{return mt2+(float)((maxA-v)/rng)*gh2;};
            Pen gridPen(Color(40,60,60,80),1.0f); Font gridFnt(L"Consolas",7.0f);
            SolidBrush gridBr(Color(255,80,80,100));
            for (int i=0;i<=4;i++){
                float y2=mt2+gh2*i/4;
                g.DrawLine(&gridPen,ml,y2,ml+gw,y2);
                double val=maxA-rng*i/4;
                std::wostringstream ss; ss<<std::fixed<<std::setprecision(3)<<val;
                g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
            }
            int bestAccIdx=-1; double bestAcc=-1;
            Pen accPen(Color(255,0,200,120),1.8f);
            { bool st=false; float px3=0,py3=0;
              for (size_t i=0;i<pts.size();i++){
                if (!pts[i].hasAcc) continue;
                float cx=xf((int)i),cy=yf(pts[i].accuracy);
                if (st) g.DrawLine(&accPen,px3,py3,cx,cy);
                px3=cx; py3=cy; st=true;
                if (pts[i].accuracy>bestAcc){bestAcc=pts[i].accuracy; bestAccIdx=(int)i;}
            }}
            if (bestAccIdx>=0){
                float bx=xf(bestAccIdx),by=yf(bestAcc);
                SolidBrush mk(Color(255,0,200,120));
                PointF dm[4]={{bx,by-5},{bx+5,by},{bx,by+5},{bx-5,by}};
                g.FillPolygon(&mk,dm,4);
                Font mf(L"Consolas",6.5f);
                std::wostringstream ss; ss<<std::fixed<<std::setprecision(4)<<bestAcc;
                g.DrawString((L"Best: "+ss.str()).c_str(),-1,&mf,PointF(bx+7,by-5),&mk);
            }
        }
        curY += pt_h + panelGap;
    }

    // ---- Learning Rate panel ----
    if (g_showLR) {
        float pt_top = curY, pt_h = lrH;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        SolidBrush panelBg(Color(255,22,18,28));
        g.FillRectangle(&panelBg, 0.0f, pt_top, (float)W2, pt_h);
        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Learning Rate", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);

        bool hasAnyLR = false;
        for (auto& p2 : pts) if (p2.hasLR) { hasAnyLR = true; break; }

        if (!hasAnyLR) {
            Font nf(L"Segoe UI", 9.0f); SolidBrush nb(Color(255,80,80,100));
            g.DrawString(L"No LR data", -1, &nf, PointF(ml+gw/2-30, mt2+gh2/2-6), &nb);
        } else {
            double minL=1e9, maxL=-1e9;
            for (auto& p2 : pts) {
                if (!p2.hasLR) continue;
                minL=(std::min)(minL,p2.lr); maxL=(std::max)(maxL,p2.lr);
            }
            if (maxL<=minL) maxL=minL+0.0001;
            double rng=maxL-minL; minL-=rng*0.05; maxL+=rng*0.05; rng=maxL-minL;
            auto yf=[&](double v)->float{return mt2+(float)((maxL-v)/rng)*gh2;};
            Pen gridPen(Color(40,60,60,80),1.0f); Font gridFnt(L"Consolas",7.0f);
            SolidBrush gridBr(Color(255,80,80,100));
            for (int i=0;i<=4;i++){
                float y2=mt2+gh2*i/4;
                g.DrawLine(&gridPen,ml,y2,ml+gw,y2);
                double val=maxL-rng*i/4;
                std::wostringstream ss; ss<<std::scientific<<std::setprecision(2)<<val;
                g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
            }
            Pen lrPen(Color(255,180,80,220),1.8f);
            { bool st=false; float px4=0,py4=0;
              for (size_t i=0;i<pts.size();i++){
                if (!pts[i].hasLR) continue;
                float cx=xf((int)i),cy=yf(pts[i].lr);
                if (st) g.DrawLine(&lrPen,px4,py4,cx,cy);
                px4=cx; py4=cy; st=true;
            }}
        }
        curY += pt_h + panelGap;
    }

    // ---- Hover crosshair + tooltip ----
    if (g_graphHoverIdx >= 0 && g_graphHoverIdx < (int)pts.size()) {
        float hx = xf(g_graphHoverIdx);
        Pen crossPen(Color(100,200,200,220), 1.0f);
        crossPen.SetDashStyle(DashStyleDash);
        g.DrawLine(&crossPen, hx, 0.0f, hx, (float)H2);

        auto& hp = pts[g_graphHoverIdx];
        std::wostringstream ss;
        ss << L"Step: " << hp.step << L"  Gen: " << hp.gen;
        ss << L"\nTrain: " << std::fixed << std::setprecision(6) << hp.train;
        if (hp.hasVal) ss << L"\nVal: " << std::fixed << std::setprecision(6) << hp.val;
        if (hp.hasLR)  ss << L"\nLR: " << std::scientific << std::setprecision(4) << hp.lr;
        if (hp.hasAcc) ss << L"\nAcc: " << std::fixed << std::setprecision(4) << hp.accuracy;
        std::wstring info = ss.str();

        Font tipFnt(L"Consolas", 7.5f);
        RectF tipRc;
        g.MeasureString(info.c_str(), -1, &tipFnt, PointF(0,0), &tipRc);
        float tipW = tipRc.Width + 14, tipH = tipRc.Height + 10;
        float tipX = (float)g_graphMousePt.x + 14;
        float tipY = (float)g_graphMousePt.y + 4;
        if (tipX + tipW > W2) tipX = (float)g_graphMousePt.x - tipW - 4;
        if (tipY + tipH > H2) tipY = (float)g_graphMousePt.y - tipH - 4;

        SolidBrush tipBg(Color(230,30,30,40));
        Pen tipBorder(Color(200,100,100,120), 1.0f);
        g.FillRectangle(&tipBg, tipX, tipY, tipW, tipH);
        g.DrawRectangle(&tipBorder, tipX, tipY, tipW, tipH);
        SolidBrush tipText(Color(255,220,220,230));
        g.DrawString(info.c_str(), -1, &tipFnt, PointF(tipX+7, tipY+5), &tipText);
    }

    BitBlt(hdc,0,0,W2,H2,memDC,0,0,SRCCOPY);
    DeleteObject(bmp); DeleteDC(memDC);
    EndPaint(hw,&ps);
}

static LRESULT CALLBACK GraphProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg==WM_PAINT) { DrawGraph(hw); return 0; }
    if (msg==WM_ERASEBKGND) return 1;
    if (msg==WM_MOUSEMOVE) {
        POINT pt = {GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
        g_graphMousePt = pt;
        RECT rc2; GetClientRect(hw, &rc2);
        float ml2=52, gw2=(float)rc2.right-ml2-16;
        std::vector<TrainPoint> pts2;
        { std::lock_guard<std::mutex> lk(g_st.mtx); pts2 = g_st.pts; }
        if (pts2.size()>=2 && gw2>0) {
            float fx = (float)(pt.x - ml2) / gw2;
            int idx2 = (int)(fx * (pts2.size()-1) + 0.5f);
            if (idx2<0) idx2=0;
            if (idx2>=(int)pts2.size()) idx2=(int)pts2.size()-1;
            g_graphHoverIdx = (fx>=0 && fx<=1.0f) ? idx2 : -1;
        } else g_graphHoverIdx = -1;
        InvalidateRect(hw, nullptr, FALSE);
        TRACKMOUSEEVENT tme{}; tme.cbSize=sizeof(tme);
        tme.dwFlags=TME_LEAVE; tme.hwndTrack=hw;
        TrackMouseEvent(&tme);
        return 0;
    }
    if (msg==WM_MOUSELEAVE) {
        g_graphHoverIdx=-1; g_graphMousePt={-1,-1};
        InvalidateRect(hw,nullptr,FALSE);
        return 0;
    }
    return DefWindowProcW(hw,msg,wp,lp);
}

// ── Config panel helpers ──────────────────────────────────────────
static HWND mkLabel(HWND parent, const wchar_t* txt, int x, int y, int w, int h) {
    HWND hw = CreateWindowExW(0,L"STATIC",txt,WS_CHILD|WS_VISIBLE|SS_LEFT|SS_NOTIFY,
                              x,y,w,h,parent,nullptr,g_hInst,nullptr);
    SendMessageW(hw, WM_SETFONT, (WPARAM)g_fUI, TRUE);
    SetWindowLongPtrW(hw, GWLP_ID, (LONG_PTR)hw); // use HWND as id for tooltip
    return hw;
}

static HWND mkEdit(HWND parent, int id, const wchar_t* def, int x, int y, int w, int h) {
    HWND hw = CreateWindowExW(WS_EX_CLIENTEDGE,L"EDIT",def,
                              WS_CHILD|WS_VISIBLE|ES_AUTOHSCROLL,
                              x,y,w,h,parent,(HMENU)(LONG_PTR)id,g_hInst,nullptr);
    SendMessageW(hw, WM_SETFONT, (WPARAM)g_fUI, TRUE);
    return hw;
}

static HWND mkCheck(HWND parent, int id, const wchar_t* txt, int x, int y, int w, int h, bool chk) {
    HWND hw = CreateWindowExW(0,L"BUTTON",txt,
                              WS_CHILD|WS_VISIBLE|BS_AUTOCHECKBOX,
                              x,y,w,h,parent,(HMENU)(LONG_PTR)id,g_hInst,nullptr);
    SendMessageW(hw, WM_SETFONT, (WPARAM)g_fUI, TRUE);
    Button_SetCheck(hw, chk ? BST_CHECKED : BST_UNCHECKED);
    return hw;
}


// ── Custom tooltip system (avoids TOOLTIPS_CLASS white-box bugs) ──────────
static HWND g_hTipWnd = nullptr;
static std::map<HWND, const wchar_t*> g_tipMap;
static HWND g_tipCurrent = nullptr;

static LRESULT CALLBACK TipWndProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hw, &ps);
        RECT rc; GetClientRect(hw, &rc);
        // Dark background
        HBRUSH br = CreateSolidBrush(RGB(50, 50, 55));
        FillRect(hdc, &rc, br);
        DeleteObject(br);
        // Border
        HPEN pen = CreatePen(PS_SOLID, 1, RGB(100, 100, 110));
        HPEN oldPen = (HPEN)SelectObject(hdc, pen);
        HBRUSH oldBr = (HBRUSH)SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Rectangle(hdc, rc.left, rc.top, rc.right, rc.bottom);
        SelectObject(hdc, oldPen);
        SelectObject(hdc, oldBr);
        DeleteObject(pen);
        // Text
        const wchar_t* text = (const wchar_t*)GetWindowLongPtrW(hw, GWLP_USERDATA);
        if (text) {
            SetBkMode(hdc, TRANSPARENT);
            SetTextColor(hdc, RGB(220, 220, 220));
            HFONT oldFont = (HFONT)SelectObject(hdc, g_fUI);
            RECT textRc = {6, 4, rc.right - 6, rc.bottom - 4};
            DrawTextW(hdc, text, -1, &textRc, DT_LEFT | DT_WORDBREAK | DT_NOPREFIX);
            SelectObject(hdc, oldFont);
        }
        EndPaint(hw, &ps);
        return 0;
    }
    if (msg == WM_NCHITTEST) return HTTRANSPARENT; // clicks pass through
    return DefWindowProcW(hw, msg, wp, lp);
}

static void ShowCustomTip(HWND hCtrl) {
    auto it = g_tipMap.find(hCtrl);
    if (it == g_tipMap.end()) return;
    if (g_tipCurrent == hCtrl) return;
    g_tipCurrent = hCtrl;

    const wchar_t* text = it->second;
    SetWindowLongPtrW(g_hTipWnd, GWLP_USERDATA, (LONG_PTR)text);

    // Calculate size needed
    HDC hdc = GetDC(g_hTipWnd);
    HFONT oldFont = (HFONT)SelectObject(hdc, g_fUI);
    RECT calcRc = {0, 0, 280, 0};
    DrawTextW(hdc, text, -1, &calcRc, DT_CALCRECT | DT_WORDBREAK | DT_NOPREFIX);
    SelectObject(hdc, oldFont);
    ReleaseDC(g_hTipWnd, hdc);

    int tipW = calcRc.right + 14;
    int tipH = calcRc.bottom + 10;
    if (tipW < 80) tipW = 80;

    // Position below the control
    RECT ctrlRc;
    GetWindowRect(hCtrl, &ctrlRc);
    int tipX = ctrlRc.left;
    int tipY = ctrlRc.bottom + 2;

    // Keep on screen
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    if (tipX + tipW > screenW) tipX = screenW - tipW;
    if (tipY + tipH > screenH) tipY = ctrlRc.top - tipH - 2;

    SetWindowPos(g_hTipWnd, HWND_TOPMOST, tipX, tipY, tipW, tipH, SWP_NOACTIVATE);
    ShowWindow(g_hTipWnd, SW_SHOWNOACTIVATE);
    InvalidateRect(g_hTipWnd, nullptr, TRUE);
}

static void HideCustomTip() {
    ShowWindow(g_hTipWnd, SW_HIDE);
    g_tipCurrent = nullptr;
}

static LRESULT CALLBACK TipSubclassProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp,
                                         UINT_PTR uIdSubclass, DWORD_PTR dwRefData) {
    if (msg == WM_MOUSEMOVE) {
        ShowCustomTip(hw);
        TRACKMOUSEEVENT tme{};
        tme.cbSize = sizeof(tme);
        tme.dwFlags = TME_LEAVE;
        tme.hwndTrack = hw;
        TrackMouseEvent(&tme);
    } else if (msg == WM_MOUSELEAVE) {
        HideCustomTip();
    }
    return DefSubclassProc(hw, msg, wp, lp);
}

static void AddTooltip(HWND hCtrl, const wchar_t* text) {
    g_tipMap[hCtrl] = text;
    SetWindowSubclass(hCtrl, TipSubclassProc, 2, 0);
}

static void BuildConfigPane(HWND pane, int PW) {
    int lw = 110, ew = PW - lw - 24, ex = lw + 12, lx = 8;
    int y = 8, dy = 24;


    // ── Section: Variant ─────────────────────────────────────────
    {
        HWND lbl = mkLabel(pane, L"Variant", lx, y+2, lw, 18);
        g_hVariant = CreateWindowExW(0, L"COMBOBOX", nullptr,
                                     WS_CHILD|WS_VISIBLE|CBS_DROPDOWNLIST|WS_VSCROLL,
                                     ex, y, ew, 120, pane,
                                     (HMENU)(LONG_PTR)ID_COMBO_VARIANT, g_hInst, nullptr);
        SendMessageW(g_hVariant, WM_SETFONT, (WPARAM)g_fUI, TRUE);
        SendMessageW(g_hVariant, CB_ADDSTRING, 0, (LPARAM)L"Standard Chess");
        SendMessageW(g_hVariant, CB_ADDSTRING, 0, (LPARAM)L"Duck Chess");
        SendMessageW(g_hVariant, CB_SETCURSEL, 0, 0);
        AddTooltip(lbl, L"Select which chess variant to train. Standard Chess uses 768-feature NNUE. Duck Chess uses 832-feature DuckNNUE with duck-square encoding.");
        y += dy;
    }

    // ── Section: Presets ──────────────────────────────────────────
    {
        HWND lbl = mkLabel(pane, L"Preset", lx, y+2, lw, 18);
        g_hPreset = CreateWindowExW(0, L"COMBOBOX", nullptr,
                                    WS_CHILD|WS_VISIBLE|CBS_DROPDOWNLIST|WS_VSCROLL,
                                    ex, y, ew, 200, pane,
                                    (HMENU)(LONG_PTR)ID_COMBO_PRESET, g_hInst, nullptr);
        SendMessageW(g_hPreset, WM_SETFONT, (WPARAM)g_fUI, TRUE);
        AddTooltip(lbl, L"Select a built-in or custom preset. Custom presets can be saved and deleted.");
        y += dy;
    }
    // Save As / Delete buttons
    {
        int halfW = (ew - 4) / 2;
        g_hBtnSave = CreateWindowExW(0, L"BUTTON", L"Save As...",
                                     WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                                     ex, y, halfW, 22, pane,
                                     (HMENU)(LONG_PTR)ID_BTN_SAVE_PRESET, g_hInst, nullptr);
        SendMessageW(g_hBtnSave, WM_SETFONT, (WPARAM)g_fUI, TRUE);
        g_hBtnDel = CreateWindowExW(0, L"BUTTON", L"Delete",
                                    WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                                    ex + halfW + 4, y, halfW, 22, pane,
                                    (HMENU)(LONG_PTR)ID_BTN_DEL_PRESET, g_hInst, nullptr);
        SendMessageW(g_hBtnDel, WM_SETFONT, (WPARAM)g_fUI, TRUE);
        EnableWindow(g_hBtnDel, FALSE);
        y += dy + 2;
    }

    // ── Section: Graph Toggles ───────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Graph Panels ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_fUI,TRUE);
        y += 20;
        int chkW = (PW - 24) / 3;
        g_hChkGLoss = mkCheck(pane, ID_CHK_GRAPH_LOSS, L"Loss", lx, y, chkW, 20, true);
        g_hChkGAcc  = mkCheck(pane, ID_CHK_GRAPH_ACC,  L"Accuracy", lx+chkW, y, chkW, 20, false);
        g_hChkGLR   = mkCheck(pane, ID_CHK_GRAPH_LR,   L"LR", lx+chkW*2, y, chkW, 20, false);
        AddTooltip(g_hChkGLoss, L"Toggle the Loss curve panel (Train + Val loss). Best values are marked with diamond markers.");
        AddTooltip(g_hChkGAcc,  L"Toggle the Accuracy panel. Shows accuracy if reported by the training script.");
        AddTooltip(g_hChkGLR,   L"Toggle the Learning Rate schedule panel.");
        y += dy;
    }

    // ── Section header: Self-Play ─────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Self-Play ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_fUI,TRUE);
        y += 20;
    }

    // Generations
    {
        HWND lbl = mkLabel(pane, L"Generations", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_GENS, L"10", ex, y, ew, 20);
        g_edits[ID_EDIT_GENS] = ed;
        AddTooltip(lbl, L"Number of self-play \u2192 train cycles to run. More generations = stronger engine over time.");
        y += dy;
    }

    // Start Gen
    {
        HWND lbl = mkLabel(pane, L"Start Gen", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_STARTGEN, L"0", ex, y, ew, 20);
        g_edits[ID_EDIT_STARTGEN] = ed;
        AddTooltip(lbl, L"Generation number to resume from. 0 = start fresh. Set to last completed gen to continue a previous run.");
        y += dy;
    }

    // Games per Gen
    {
        HWND lbl = mkLabel(pane, L"Games per Gen", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_GAMES, L"5000", ex, y, ew, 20);
        g_edits[ID_EDIT_GAMES] = ed;
        AddTooltip(lbl, L"Number of self-play games generated each generation. More games = more training data but takes longer.");
        y += dy;
    }

    // Workers
    {
        HWND lbl = mkLabel(pane, L"Workers", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WORKERS, L"12", ex, y, ew, 20);
        g_edits[ID_EDIT_WORKERS] = ed;
        AddTooltip(lbl, L"Parallel threads used for self-play game generation. Set to your CPU thread count (e.g. 12) for best speed.");
        y += dy;
    }

    // Depth
    {
        HWND lbl = mkLabel(pane, L"Depth", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DEPTH, L"5", ex, y, ew, 20);
        g_edits[ID_EDIT_DEPTH] = ed;
        AddTooltip(lbl, L"Search depth for self-play move selection. Higher depth = stronger, more instructive games but much slower generation.");
        y += dy;
    }

    // ── Section header: Training ──────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Training ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_fUI,TRUE);
        y += 20;
    }

    // Epochs per Gen
    {
        HWND lbl = mkLabel(pane, L"Epochs per Gen", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_EPOCHS, L"10", ex, y, ew, 20);
        g_edits[ID_EDIT_EPOCHS] = ed;
        AddTooltip(lbl, L"How many full passes through the training data each generation. More epochs = better fitting but risk of overfitting.");
        y += dy;
    }

    // Batch Size
    {
        HWND lbl = mkLabel(pane, L"Batch Size", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_BATCHSZ, L"2048", ex, y, ew, 20);
        g_edits[ID_EDIT_BATCHSZ] = ed;
        AddTooltip(lbl, L"Number of positions processed together per gradient update. Larger = more stable gradients, requires more VRAM.");
        y += dy;
    }

    // Learning Rate
    {
        HWND lbl = mkLabel(pane, L"Learning Rate", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_LR, L"0.001", ex, y, ew, 20);
        g_edits[ID_EDIT_LR] = ed;
        AddTooltip(lbl, L"Step size for gradient updates. Too high = unstable training. Too low = slow convergence. Typically 0.0001\u20130.01.");
        y += dy;
    }

    // Weight Decay
    {
        HWND lbl = mkLabel(pane, L"Weight Decay", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WD, L"1e-5", ex, y, ew, 20);
        g_edits[ID_EDIT_WD] = ed;
        AddTooltip(lbl, L"L2 regularization strength. Penalizes large weights to reduce overfitting. Usually a small value like 1e-5.");
        y += dy;
    }

    // Dropout
    {
        HWND lbl = mkLabel(pane, L"Dropout", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DROPOUT, L"0.1", ex, y, ew, 20);
        g_edits[ID_EDIT_DROPOUT] = ed;
        AddTooltip(lbl, L"Fraction of neurons randomly disabled during training. Reduces overfitting. 0 = disabled, 0.1\u20130.3 typical.");
        y += dy;
    }

    // Label Smoothing
    {
        HWND lbl = mkLabel(pane, L"Label Smoothing", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_LSMOOTH, L"0.05", ex, y, ew, 20);
        g_edits[ID_EDIT_LSMOOTH] = ed;
        AddTooltip(lbl, L"Softens hard 0/1 targets to reduce overconfidence. 0.05 means targets become 0.05 and 0.95 instead of 0 and 1.");
        y += dy;
    }

    // Grad Accumulation
    {
        HWND lbl = mkLabel(pane, L"Grad Accumulation", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_GRADACCUM, L"4", ex, y, ew, 20);
        g_edits[ID_EDIT_GRADACCUM] = ed;
        AddTooltip(lbl, L"Accumulate gradients over N batches before updating weights. Simulates a larger effective batch size without extra memory.");
        y += dy;
    }

    // LR Warmup Steps
    {
        HWND lbl = mkLabel(pane, L"LR Warmup Steps", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WARMUP, L"500", ex, y, ew, 20);
        g_edits[ID_EDIT_WARMUP] = ed;
        AddTooltip(lbl, L"Linearly ramp learning rate from 0 to target over this many steps at the start of training. Stabilizes early training.");
        y += dy;
    }

    // ── Section header: Scoring ───────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Scoring ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_fUI,TRUE);
        y += 20;
    }

    // Draw Weight
    {
        HWND lbl = mkLabel(pane, L"Draw Weight", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAWWT, L"0.25", ex, y, ew, 20);
        g_edits[ID_EDIT_DRAWWT] = ed;
        AddTooltip(lbl, L"How much draw outcomes contribute to training signal (vs win/loss). 0.25 = draws count quarter as much as decisive games.");
        y += dy;
    }

    // Mate Boost
    {
        HWND lbl = mkLabel(pane, L"Mate Boost", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MATEBOOST, L"3.0", ex, y, ew, 20);
        g_edits[ID_EDIT_MATEBOOST] = ed;
        AddTooltip(lbl, L"Multiplier applied to positions near checkmate. Higher values make the engine prioritize mating patterns more strongly.");
        y += dy;
    }

    // Self-Play Ratio
    {
        HWND lbl = mkLabel(pane, L"Self-Play Ratio", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SPLRATIO, L"0.4", ex, y, ew, 20);
        g_edits[ID_EDIT_SPLRATIO] = ed;
        AddTooltip(lbl, L"Fraction of training data from current generation vs older generations. Higher = more recent data emphasis.");
        y += dy;
    }

    // ── Section header: Validation ────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Validation ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_fUI,TRUE);
        y += 20;
    }

    // ELO Validation checkbox
    {
        g_hChkElo = mkCheck(pane, ID_CHK_ELO, L"ELO Validation", lx, y, PW-16, 20, false);
        AddTooltip(g_hChkElo, L"After each generation, run 50 games between new and previous model to measure ELO improvement.");
        y += dy;
    }

    // Overfitting Detection checkbox
    {
        g_hChkOvfit = mkCheck(pane, ID_CHK_OVERFIT, L"Overfitting Detection", lx, y, PW-16, 20, true);
        AddTooltip(g_hChkOvfit, L"Monitor validation loss and stop early if training loss diverges from val loss significantly.");
        y += dy;
    }
}

// ── Read config from UI ───────────────────────────────────────────
static Config ReadConfig() {
    Config c;
    auto e = [&](int id) -> std::string {
        auto it = g_edits.find(id);
        if (it == g_edits.end()) return "";
        return getEdit(it->second);
    };
    c.generations  = pInt(e(ID_EDIT_GENS),   10);
    c.gamesPerGen  = pInt(e(ID_EDIT_GAMES), 5000);
    c.epochsPerGen = pInt(e(ID_EDIT_EPOCHS),  10);
    c.batchSize    = pInt(e(ID_EDIT_BATCHSZ),2048);
    c.lr           = pDbl(e(ID_EDIT_LR),    0.001);
    c.weightDecay  = pDbl(e(ID_EDIT_WD),    1e-5);
    c.dropout      = pDbl(e(ID_EDIT_DROPOUT),0.1);
    c.labelSmooth  = pDbl(e(ID_EDIT_LSMOOTH),0.05);
    c.gradAccum    = pInt(e(ID_EDIT_GRADACCUM),4);
    c.warmupSteps  = pInt(e(ID_EDIT_WARMUP),500);
    c.drawWeight   = pDbl(e(ID_EDIT_DRAWWT), 0.25);
    c.mateBoost    = pDbl(e(ID_EDIT_MATEBOOST),3.0);
    c.splRatio     = pDbl(e(ID_EDIT_SPLRATIO),0.4);
    c.workers      = pInt(e(ID_EDIT_WORKERS), 12);
    c.depth        = pInt(e(ID_EDIT_DEPTH),    5);
    c.startGen     = pInt(e(ID_EDIT_STARTGEN), 0);
    c.eloValidate   = Button_GetCheck(g_hChkElo)   == BST_CHECKED;
    c.overfitDetect = Button_GetCheck(g_hChkOvfit) == BST_CHECKED;
    // Read variant selection
    int variantSel = (int)SendMessageW(g_hVariant, CB_GETCURSEL, 0, 0);
    c.variant = (variantSel == 1) ? ChessVariant::DuckChess : ChessVariant::Standard;
    return c;
}

// ── Apply preset from g_allPresets ────────────────────────────────
static void ApplyPreset(int idx) {
    if (idx < 0 || idx >= (int)g_allPresets.size()) return;
    g_currentPresetIdx = idx;
    Preset& p = g_allPresets[idx];
    auto se = [&](int id, const std::string& v) {
        auto it = g_edits.find(id);
        if (it != g_edits.end()) setEdit(it->second, v);
    };
    se(ID_EDIT_GENS,      std::to_string(p.generations));
    se(ID_EDIT_GAMES,     std::to_string(p.gamesPerGen));
    se(ID_EDIT_EPOCHS,    std::to_string(p.epochsPerGen));
    se(ID_EDIT_BATCHSZ,   std::to_string(p.batchSize));
    se(ID_EDIT_WORKERS,   std::to_string(p.workers));
    se(ID_EDIT_DEPTH,     std::to_string(p.depth));
    se(ID_EDIT_STARTGEN,  std::to_string(p.startGen));
    se(ID_EDIT_GRADACCUM, std::to_string(p.gradAccum));
    se(ID_EDIT_WARMUP,    std::to_string(p.warmupSteps));
    se(ID_EDIT_LR,        dbl2s(p.lr,6));
    se(ID_EDIT_WD,        dbl2s(p.weightDecay,8));
    se(ID_EDIT_DROPOUT,   dbl2s(p.dropout,4));
    se(ID_EDIT_LSMOOTH,   dbl2s(p.labelSmooth,4));
    se(ID_EDIT_DRAWWT,    dbl2s(p.drawWeight,4));
    se(ID_EDIT_MATEBOOST, dbl2s(p.mateBoost,4));
    se(ID_EDIT_SPLRATIO,  dbl2s(p.splRatio,4));
    Button_SetCheck(g_hChkElo,  p.eloValidate  ? BST_CHECKED : BST_UNCHECKED);
    Button_SetCheck(g_hChkOvfit,p.overfitDetect ? BST_CHECKED : BST_UNCHECKED);
    if (g_hBtnDel) EnableWindow(g_hBtnDel, !p.isBuiltin);
}

// ── Save As dialog helper ─────────────────────────────────────────
struct SavePresetDlgData { wchar_t name[128]; bool ok; };

static INT_PTR CALLBACK SavePresetDlgProc(HWND hDlg, UINT msg, WPARAM wp2, LPARAM lp2) {
    if (msg == WM_INITDIALOG) {
        SetWindowLongPtrW(hDlg, GWLP_USERDATA, lp2);
        SavePresetDlgData* d2 = (SavePresetDlgData*)lp2;
        SetDlgItemTextW(hDlg, 200, d2->name);
        SendDlgItemMessageW(hDlg, 200, EM_SETSEL, 0, -1);
        return TRUE;
    }
    if (msg == WM_COMMAND) {
        if (LOWORD(wp2)==IDOK) {
            SavePresetDlgData* d2=(SavePresetDlgData*)GetWindowLongPtrW(hDlg, GWLP_USERDATA);
            GetDlgItemTextW(hDlg, 200, d2->name, 128);
            d2->ok = true;
            EndDialog(hDlg, IDOK); return TRUE;
        }
        if (LOWORD(wp2)==IDCANCEL) { EndDialog(hDlg, IDCANCEL); return TRUE; }
    }
    if (msg==WM_CLOSE) { EndDialog(hDlg, IDCANCEL); return TRUE; }
    return FALSE;
}

// ── Save As dialog ───────────────────────────────────────────────
static void SavePresetAs() {
    // Build a dialog template in memory
    SavePresetDlgData dd = {}; wcscpy_s(dd.name, L"My Custom Preset"); dd.ok = false;


    alignas(4) unsigned char buf[1024] = {};
    DLGTEMPLATE* pDlg = (DLGTEMPLATE*)buf;
    pDlg->style = DS_MODALFRAME | DS_CENTER | WS_POPUP | WS_CAPTION | WS_SYSMENU;
    pDlg->cdit = 3;
    pDlg->cx = 180; pDlg->cy = 60;
    WORD* pw = (WORD*)(pDlg + 1);
    *pw++ = 0; *pw++ = 0;
    // Title
    *pw++ = L'S'; *pw++ = L'a'; *pw++ = L'v'; *pw++ = L'e';
    *pw++ = L' '; *pw++ = L'P'; *pw++ = L'r'; *pw++ = L'e';
    *pw++ = L's'; *pw++ = L'e'; *pw++ = L't'; *pw++ = L' ';
    *pw++ = L'A'; *pw++ = L's'; *pw++ = 0;

    // Edit control (id=200)
    pw = (WORD*)(((ULONG_PTR)pw + 3) & ~3);
    DLGITEMTEMPLATE* pIt = (DLGITEMTEMPLATE*)pw;
    pIt->style = WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL;
    pIt->x=8; pIt->y=8; pIt->cx=164; pIt->cy=14; pIt->id=200;
    pw = (WORD*)(pIt+1);
    *pw++ = 0xFFFF; *pw++ = 0x0081; *pw++ = 0; *pw++ = 0;

    // OK button
    pw = (WORD*)(((ULONG_PTR)pw + 3) & ~3);
    pIt = (DLGITEMTEMPLATE*)pw;
    pIt->style = WS_CHILD|WS_VISIBLE|BS_DEFPUSHBUTTON;
    pIt->x=50; pIt->y=30; pIt->cx=40; pIt->cy=14; pIt->id=IDOK;
    pw = (WORD*)(pIt+1);
    *pw++ = 0xFFFF; *pw++ = 0x0080;
    *pw++ = L'O'; *pw++ = L'K'; *pw++ = 0; *pw++ = 0;

    // Cancel button
    pw = (WORD*)(((ULONG_PTR)pw + 3) & ~3);
    pIt = (DLGITEMTEMPLATE*)pw;
    pIt->style = WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON;
    pIt->x=96; pIt->y=30; pIt->cx=40; pIt->cy=14; pIt->id=IDCANCEL;
    pw = (WORD*)(pIt+1);
    *pw++ = 0xFFFF; *pw++ = 0x0080;
    *pw++ = L'C'; *pw++ = L'a'; *pw++ = L'n'; *pw++ = L'c';
    *pw++ = L'e'; *pw++ = L'l'; *pw++ = 0; *pw++ = 0;

    INT_PTR result = DialogBoxIndirectParamW(g_hInst, (DLGTEMPLATE*)buf, g_hWnd,
                                              SavePresetDlgProc, (LPARAM)&dd);
    if (result != IDOK || !dd.ok) return;
    std::string name = N(std::wstring(dd.name));
    if (name.empty()) return;

    Config c = ReadConfig();
    Preset np;
    np.name = name; np.isBuiltin = false;
    np.generations = c.generations; np.gamesPerGen = c.gamesPerGen;
    np.epochsPerGen = c.epochsPerGen; np.batchSize = c.batchSize;
    np.lr = c.lr; np.weightDecay = c.weightDecay;
    np.dropout = c.dropout; np.labelSmooth = c.labelSmooth;
    np.gradAccum = c.gradAccum; np.warmupSteps = c.warmupSteps;
    np.drawWeight = c.drawWeight; np.mateBoost = c.mateBoost;
    np.splRatio = c.splRatio; np.workers = c.workers;
    np.depth = c.depth; np.startGen = c.startGen;
    np.eloValidate = c.eloValidate; np.overfitDetect = c.overfitDetect;

    bool found = false;
    for (size_t i = 0; i < g_allPresets.size(); i++) {
        if (!g_allPresets[i].isBuiltin && g_allPresets[i].name == name) {
            g_allPresets[i] = np; g_currentPresetIdx = (int)i; found = true; break;
        }
    }
    if (!found) {
        g_allPresets.push_back(np);
        g_currentPresetIdx = (int)g_allPresets.size() - 1;
    }
    SaveCustomPresets();
    PopulatePresetCombo();
    if (g_hBtnDel) EnableWindow(g_hBtnDel, TRUE);
}

static void DeleteCurrentPreset() {
    if (g_currentPresetIdx < 0 || g_currentPresetIdx >= (int)g_allPresets.size()) return;
    if (g_allPresets[g_currentPresetIdx].isBuiltin) return;
    std::string name = g_allPresets[g_currentPresetIdx].name;
    int res = MessageBoxW(g_hWnd,
        (L"Delete preset \"" + W(name) + L"\"?").c_str(),
        L"Confirm Delete", MB_YESNO | MB_ICONQUESTION);
    if (res != IDYES) return;
    g_allPresets.erase(g_allPresets.begin() + g_currentPresetIdx);
    SaveCustomPresets();
    g_currentPresetIdx = 1;
    PopulatePresetCombo();
    ApplyPreset(g_currentPresetIdx);
}

// ── Config panel subclass (dark background + label colors) ────────
static LRESULT CALLBACK PanelProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp,
                                   UINT_PTR, DWORD_PTR) {
    if (msg == WM_CTLCOLORSTATIC) {
        HDC hdc = (HDC)wp;
        SetBkColor(hdc, C_PANEL);
        SetTextColor(hdc, C_TEXT);
        return (LRESULT)g_brPanel;
    }
    if (msg == WM_CTLCOLOREDIT) {
        HDC hdc = (HDC)wp;
        SetBkColor(hdc, RGB(30,30,46));
        SetTextColor(hdc, C_TEXT);
        static HBRUSH editBr = CreateSolidBrush(RGB(30,30,46));
        return (LRESULT)editBr;
    }
    if (msg == WM_ERASEBKGND) {
        RECT rc; GetClientRect(hw, &rc);
        FillRect((HDC)wp, &rc, g_brPanel);
        return 1;
    }
    // Forward WM_COMMAND from child controls (buttons, combos, checkboxes)
    // to the main window so they can be handled in WndProc
    if (msg == WM_COMMAND) {
        PostMessage(g_hWnd, WM_COMMAND, wp, lp);
        return 0;
    }
    return DefSubclassProc(hw, msg, wp, lp);
}

// ── Update log listbox ────────────────────────────────────────────
static size_t g_logSent = 0;
static void FlushLog() {
    std::deque<std::string> snap;
    bool replaced = false;
    { std::lock_guard<std::mutex> lk(g_st.mtx);
      snap = g_st.log;
      replaced = g_st.logLastReplaced;
      g_st.logLastReplaced = false;
    }
    // If the last line was replaced (running update), remove the old listbox
    // entry so the updated text gets re-added below.
    if (replaced && g_logSent > 0 && snap.size() <= g_logSent) {
        int cnt = (int)SendMessageW(g_hLog, LB_GETCOUNT, 0, 0);
        if (cnt > 0) {
            SendMessageW(g_hLog, LB_DELETESTRING, (WPARAM)(cnt - 1), 0);
            g_logSent--;
        }
    }
    if (snap.size() <= g_logSent) return;
    for (size_t i = g_logSent; i < snap.size(); i++) {
        SendMessageW(g_hLog, LB_ADDSTRING, 0, (LPARAM)W(snap[i]).c_str());
    }
    g_logSent = snap.size();
    int cnt = (int)SendMessageW(g_hLog, LB_GETCOUNT, 0, 0);
    if (cnt > 0) SendMessageW(g_hLog, LB_SETTOPINDEX, (WPARAM)(cnt - 1), 0);
}

// ── Main window proc ──────────────────────────────────────────────
static LRESULT CALLBACK WndProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {

    case WM_MEASUREITEM: {
        MEASUREITEMSTRUCT* mis = (MEASUREITEMSTRUCT*)lp;
        if (mis->CtlID == ID_LOG_BOX) {
            mis->itemHeight = 16;
            return TRUE;
        }
        break;
    }

    case WM_DRAWITEM: {
        DRAWITEMSTRUCT* dis = (DRAWITEMSTRUCT*)lp;
        if (dis->CtlID == ID_LOG_BOX && dis->itemID != (UINT)-1) {
            int len = (int)SendMessageW(dis->hwndItem, LB_GETTEXTLEN, dis->itemID, 0);
            if (len <= 0) break;
            std::wstring txt(len + 1, 0);
            SendMessageW(dis->hwndItem, LB_GETTEXT, dis->itemID, (LPARAM)&txt[0]);
            txt.resize(wcslen(txt.c_str()));

            std::string narrow = N(txt);
            LogColor lc = ClassifyLogLine(narrow);
            COLORREF textColor = LogColorTable[(int)lc];

            COLORREF bgColor = RGB(12,12,20);
            if (dis->itemState & ODS_SELECTED) bgColor = RGB(40,40,60);

            HBRUSH hbr = CreateSolidBrush(bgColor);
            FillRect(dis->hDC, &dis->rcItem, hbr);
            DeleteObject(hbr);

            SetBkMode(dis->hDC, TRANSPARENT);
            SetTextColor(dis->hDC, textColor);
            HFONT oldFont = (HFONT)SelectObject(dis->hDC, g_fMono);
            RECT textRc = dis->rcItem;
            textRc.left += 4;
            DrawTextW(dis->hDC, txt.c_str(), (int)txt.size(), &textRc,
                      DT_LEFT | DT_SINGLELINE | DT_VCENTER | DT_NOPREFIX);
            SelectObject(dis->hDC, oldFont);

            if (dis->itemState & ODS_FOCUS)
                DrawFocusRect(dis->hDC, &dis->rcItem);
            return TRUE;
        }
        break;
    }

    case WM_COMMAND: {
        int id = LOWORD(wp);
        if (id == ID_BTN_START && !g_st.running) {
            g_cfg = ReadConfig();
            g_logSent = 0;
            SendMessageW(g_hLog, LB_RESETCONTENT, 0, 0);
            EnableWindow(g_hStart, FALSE);
            EnableWindow(g_hStop,  TRUE);
            EnableWindow(g_hPause, TRUE);
            g_pauseFlag.store(false);
            SetWindowTextW(g_hPause, L"\u23F8 Pause");
            if (g_worker.joinable()) g_worker.join();
            g_worker = std::thread(PipelineThread, g_cfg);
        }
        else if (id == ID_BTN_STOP && g_st.running) {
            g_st.stopFlag.store(true);
            g_st.setStatus("Stopping...");
            // Resume if paused so process can receive signal
            if (g_pauseFlag.load()) {
                g_pauseFlag.store(false);
                ResumeProcessThreads(g_activePid.load());
                SetWindowTextW(g_hPause, L"\u23F8 Pause");
            }
            SuspendOrTerminateActive();
        }
        else if (id == ID_BTN_PAUSE && g_st.running) {
            if (!g_pauseFlag.load()) {
                // Pause
                g_pauseFlag.store(true);
                SetWindowTextW(g_hPause, L"\u25B6 Resume");
                SuspendProcessThreads(g_activePid.load());
            } else {
                // Resume
                g_pauseFlag.store(false);
                SetWindowTextW(g_hPause, L"\u23F8 Pause");
                ResumeProcessThreads(g_activePid.load());
            }
        }
        else if (id == ID_COMBO_PRESET && HIWORD(wp) == CBN_SELCHANGE) {
            int sel = (int)SendMessageW(g_hPreset, CB_GETCURSEL, 0, 0);
            ApplyPreset(sel);
        }
        else if (id == ID_BTN_SAVE_PRESET) {
            SavePresetAs();
        }
        else if (id == ID_BTN_DEL_PRESET) {
            DeleteCurrentPreset();
        }
        else if (id == ID_COMBO_VARIANT && HIWORD(wp) == CBN_SELCHANGE) {
            // Variant changed – just update the stored config; pipeline reads on start
        }
        else if (id == ID_CHK_GRAPH_LOSS) {
            g_showLoss = (Button_GetCheck(g_hChkGLoss) == BST_CHECKED);
            InvalidateRect(g_hGraph, nullptr, FALSE);
        }
        else if (id == ID_CHK_GRAPH_ACC) {
            g_showAcc = (Button_GetCheck(g_hChkGAcc) == BST_CHECKED);
            InvalidateRect(g_hGraph, nullptr, FALSE);
        }
        else if (id == ID_CHK_GRAPH_LR) {
            g_showLR = (Button_GetCheck(g_hChkGLR) == BST_CHECKED);
            InvalidateRect(g_hGraph, nullptr, FALSE);
        }
        break;
    }

    case WM_USER+1: {
        // Pipeline finished
        EnableWindow(g_hStart, TRUE);
        EnableWindow(g_hStop,  FALSE);
        EnableWindow(g_hPause, FALSE);
        SetWindowTextW(g_hPause, L"\u23F8 Pause");
        g_pauseFlag.store(false);
        FlushLog();
        InvalidateRect(g_hGraph, nullptr, FALSE);
        break;
    }

    case WM_TIMER: {
        if (wp == ID_TIMER) {
            // Update status text
            std::string status, phase;
            int curGen, totalGens, curEp, totalEp, elo;
            double train, val;
            bool running;
            {
                std::lock_guard<std::mutex> lk(g_st.mtx);
                status = g_st.status; phase = g_st.phase;
                curGen = g_st.curGen; totalGens = g_st.totalGens;
                curEp = g_st.curEpoch; totalEp = g_st.totalEpochs;
                train = g_st.lastTrain; val = g_st.lastVal; elo = g_st.lastElo;
                running = g_st.running;
            }
            // Status line — shows phase description and latest loss/ELO; timing is in banner
            std::wstring stw = W(status);
            if (running && totalGens > 0) {
                std::wostringstream ss;
                ss << W(status);
                if (train > 0) ss << L"  Train Loss: " << std::fixed << std::setprecision(5) << train;
                if (val > 0)   ss << L"  Val Loss: "   << std::fixed << std::setprecision(5) << val;
                if (elo != 0)  ss << L"  ELO delta: "  << elo;
                stw = ss.str();
            }
            SetWindowTextW(g_hStatus, stw.c_str());

            // Progress bar
            if (running && totalEp > 0) {
                int pct = (int)(100.0 * curEp / totalEp);
                SendMessageW(g_hProg, PBM_SETPOS, pct, 0);
            } else if (!running) {
                SendMessageW(g_hProg, PBM_SETPOS, running ? 0 : 100, 0);
            }

            // Update ETA banner
            if (running && totalGens > 0) {
                auto now = std::chrono::steady_clock::now();
                std::chrono::steady_clock::time_point pStart, phStart;
                std::chrono::steady_clock::time_point bStamp, eStamp, nStamp;
                int doneGens, bEta, eEta, nEta;
                {
                    std::lock_guard<std::mutex> lk(g_st.mtx);
                    pStart   = g_st.pipelineStart;
                    phStart  = g_st.phaseStart;
                    doneGens = g_st.completedGens;
                    bEta     = g_st.batchEtaSec;
                    eEta     = g_st.epochEtaSec;
                    nEta     = g_st.nextEpochSec;
                    bStamp   = g_st.batchEtaStamp;
                    eStamp   = g_st.epochEtaStamp;
                    nStamp   = g_st.nextEpochStamp;
                }

                auto fmtDur = [](long long totalSec) -> std::wstring {
                    if (totalSec < 0) totalSec = 0;
                    long long h = totalSec / 3600;
                    long long m = (totalSec % 3600) / 60;
                    long long s = totalSec % 60;
                    std::wostringstream o;
                    if (h > 0) o << h << L"h " << m << L"m " << s << L"s";
                    else if (m > 0) o << m << L"m " << s << L"s";
                    else o << s << L"s";
                    return o.str();
                };
                // Countdown: stored ETA minus seconds elapsed since it was received
                auto countdown = [&](int etaSec, std::chrono::steady_clock::time_point stamp) -> long long {
                    if (etaSec <= 0) return -1;
                    long long elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - stamp).count();
                    return std::max(0LL, (long long)etaSec - elapsed);
                };

                long long phaseSec  = std::chrono::duration_cast<std::chrono::seconds>(now - phStart).count();
                long long totalSec  = std::chrono::duration_cast<std::chrono::seconds>(now - pStart).count();

                // Determine phase label and progress detail
                std::wstring phaseLabel;
                std::wstring progressDetail;
                bool isTraining = (phase == "training");
                bool isSelfPlay = (phase == "selfplay");
                bool isElo      = (phase == "elovalidation" || phase == "elo");

                if (isSelfPlay) {
                    phaseLabel = L"Self-play";
                    if (totalEp > 0)
                        progressDetail = L"  Games: " + std::to_wstring(curEp) + L"/" + std::to_wstring(totalEp);
                } else if (isTraining) {
                    phaseLabel = L"Training";
                    if (totalEp > 0)
                        progressDetail = L"  Epoch: " + std::to_wstring(curEp) + L"/" + std::to_wstring(totalEp);
                } else if (isElo) {
                    phaseLabel = L"ELO Validation";
                } else {
                    phaseLabel = W(phase);
                }

                std::wostringstream bs;
                // 1. Phase + gen + progress
                bs << L"Phase: " << phaseLabel
                   << L"  Gen: " << (curGen+1) << L"/" << totalGens
                   << progressDetail;

                // 2. Phase elapsed (how long current phase has been running)
                bs << L"  |  Phase elapsed: " << fmtDur(phaseSec);

                // 3. Training-specific countdowns (only when training phase active)
                if (isTraining) {
                    long long bLeft = countdown(bEta, bStamp);
                    long long eLeft = countdown(eEta, eStamp);
                    long long nLeft = countdown(nEta, nStamp);
                    if (bLeft >= 0)
                        bs << L"  |  Batch done in: ~" << fmtDur(bLeft);
                    if (nLeft >= 0)
                        bs << L"  |  Next epoch in: ~" << fmtDur(nLeft);
                    if (eLeft >= 0)
                        bs << L"  |  Gen training done in: ~" << fmtDur(eLeft);
                }

                // 4. Total pipeline elapsed
                bs << L"  |  Pipeline elapsed: " << fmtDur(totalSec);

                // 5. Estimated pipeline remaining + finish time
                if (doneGens > 0) {
                    double avgPerGen = (double)totalSec / doneGens;
                    int remainingGens = totalGens - (curGen + 1);
                    long long estRemaining = (long long)(avgPerGen * remainingGens);
                    if (estRemaining > 0) {
                        bs << L"  |  Est. pipeline remaining: ~" << fmtDur(estRemaining);
                        auto etaTime = std::chrono::system_clock::now() + std::chrono::seconds(estRemaining);
                        std::time_t etaT = std::chrono::system_clock::to_time_t(etaTime);
                        std::tm etaTm;
                        localtime_s(&etaTm, &etaT);
                        wchar_t timeBuf[32];
                        wcsftime(timeBuf, 32, L"%H:%M", &etaTm);
                        bs << L"  |  Pipeline finishes at: " << timeBuf;
                    } else if (remainingGens <= 0) {
                        bs << L"  |  Final generation";
                    }
                } else {
                    bs << L"  |  Est. remaining: calculating...";
                }

                SetWindowTextW(g_hBanner, bs.str().c_str());
            } else if (!running) {
                SetWindowTextW(g_hBanner, L"Ready");
            }

            FlushLog();
            InvalidateRect(g_hGraph, nullptr, FALSE);
        }
        break;
    }

    case WM_CTLCOLORSTATIC: {
        HDC hdc = (HDC)wp;
        HWND hCtrl = (HWND)lp;
        if (hCtrl == g_hStatus) {
            SetBkColor(hdc, C_BG);
            SetTextColor(hdc, C_ACCENT);
            return (LRESULT)g_brBg;
        }
        if (hCtrl == g_hBanner) {
            SetBkColor(hdc, RGB(25, 25, 40));
            SetTextColor(hdc, C_ACCENT);
            static HBRUSH brBanner = CreateSolidBrush(RGB(25, 25, 40));
            return (LRESULT)brBanner;
        }
        SetBkColor(hdc, C_BG);
        SetTextColor(hdc, C_TEXT);
        return (LRESULT)g_brBg;
    }

    case WM_CTLCOLOREDIT: {
        HDC hdc = (HDC)wp;
        static HBRUSH brEdit = CreateSolidBrush(RGB(20,20,32));
        SetBkColor(hdc, RGB(20,20,32));
        SetTextColor(hdc, C_TEXT);
        return (LRESULT)brEdit;
    }

    case WM_CTLCOLORLISTBOX: {
        HDC hdc = (HDC)wp;
        static HBRUSH brLB = CreateSolidBrush(RGB(12,12,20));
        SetBkColor(hdc, RGB(12,12,20));
        SetTextColor(hdc, RGB(160,200,160));
        return (LRESULT)brLB;
    }

    case WM_CTLCOLORBTN: {
        // Handled by owner-draw or default
        break;
    }

    case WM_SIZE: {
        int W2 = LOWORD(lp), H2 = HIWORD(lp);
        if (W2 < 10 || H2 < 10) break;

        int rightX = PANEL_W + 4;
        int rightW = W2 - rightX - 4;

        // Graph area
        int graphH = H2 - LOG_H - PROG_H - BANNER_H - TITLE_H - 80;
        if (graphH < 80) graphH = 80;
        SetWindowPos(g_hGraph,  nullptr, rightX, TITLE_H, rightW, graphH, SWP_NOZORDER);

        // Log box
        int logY = TITLE_H + graphH + 4;
        SetWindowPos(g_hLog, nullptr, rightX, logY, rightW, LOG_H, SWP_NOZORDER);

        // Progress bar
        int progY = logY + LOG_H + 4;
        SetWindowPos(g_hProg, nullptr, rightX, progY, rightW, PROG_H, SWP_NOZORDER);

        // ETA Banner
        int bannerY = progY + PROG_H + 4;
        SetWindowPos(g_hBanner, nullptr, rightX, bannerY, rightW, BANNER_H, SWP_NOZORDER);

        // Status
        int stY = bannerY + BANNER_H + 4;
        SetWindowPos(g_hStatus, nullptr, rightX, stY, rightW, 20, SWP_NOZORDER);

        // Buttons (bottom of left panel)
        int btnY2 = H2 - 46;
        int btnW2 = (PANEL_W - 24) / 3;
        if (g_hStart) SetWindowPos(g_hStart, nullptr, 8, btnY2, btnW2, 28, SWP_NOZORDER);
        if (g_hStop)  SetWindowPos(g_hStop,  nullptr, 8+btnW2+4, btnY2, btnW2, 28, SWP_NOZORDER);
        if (g_hPause) SetWindowPos(g_hPause, nullptr, 8+btnW2*2+8, btnY2, btnW2, 28, SWP_NOZORDER);

        // Config pane – stop above buttons
        int paneH = btnY2 - TITLE_H - 8;
        SetWindowPos(g_hCfgPane, nullptr, 0, TITLE_H, PANEL_W, paneH, SWP_NOZORDER);
        break;
    }

    case WM_ERASEBKGND: {
        RECT rc; GetClientRect(hw, &rc);
        FillRect((HDC)wp, &rc, g_brBg);
        return 1;
    }

    case WM_PAINT: {
        PAINTSTRUCT ps; HDC hdc = BeginPaint(hw, &ps);
        // Draw title bar area
        RECT titleRc = { 0, 0, 2000, TITLE_H };
        FillRect(hdc, &titleRc, CreateSolidBrush(RGB(18,18,28)));
        // Title text
        SelectObject(hdc, g_fUI);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, C_ACCENT);
        TextOutW(hdc, 12, 10, L"NNUE Training Runner", 20);
        // Divider
        HPEN pen = CreatePen(PS_SOLID, 1, C_ACCENT);
        SelectObject(hdc, pen);
        MoveToEx(hdc, 0, TITLE_H-1, nullptr);
        LineTo(hdc, 2000, TITLE_H-1);
        DeleteObject(pen);
        EndPaint(hw, &ps);
        break;
    }

    case WM_DESTROY:
        g_st.stopFlag.store(true);
        SuspendOrTerminateActive();
        if (g_worker.joinable()) g_worker.join();
        KillTimer(hw, ID_TIMER);
        GdiplusShutdown(g_gdip);
        PostQuitMessage(0);
        break;
    }
    return DefWindowProcW(hw, msg, wp, lp);
}

// ── Entry point ───────────────────────────────────────────────────
int main(int, char**) {
    g_hInst = GetModuleHandleW(nullptr);

    // Init GDI+
    GdiplusStartupInput gdi; GdiplusStartup(&g_gdip, &gdi, nullptr);

    // Init common controls
    INITCOMMONCONTROLSEX icc{}; icc.dwSize=sizeof(icc);
    icc.dwICC = ICC_BAR_CLASSES | ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&icc);

    // Fonts
    g_fUI   = CreateFontW(-13,0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,
                           OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,
                           DEFAULT_PITCH|FF_SWISS, L"Segoe UI");
    g_fMono = CreateFontW(-12,0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,
                           OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,
                           DEFAULT_PITCH|FF_MODERN, L"Consolas");

    // Brushes
    g_brBg    = CreateSolidBrush(C_BG);
    g_brPanel = CreateSolidBrush(C_PANEL);

    // Register graph class
    WNDCLASSEXW wgc{}; wgc.cbSize=sizeof(wgc);
    wgc.lpszClassName = L"NNUEGraph";
    wgc.hInstance     = g_hInst;
    wgc.lpfnWndProc   = GraphProc;
    wgc.hbrBackground = CreateSolidBrush(C_BG);
    RegisterClassExW(&wgc);

    // Register custom tooltip class
    WNDCLASSEXW wtip{}; wtip.cbSize = sizeof(wtip);
    wtip.lpszClassName = L"TipWnd";
    wtip.hInstance     = g_hInst;
    wtip.lpfnWndProc   = TipWndProc;
    wtip.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassExW(&wtip);

    // Register main class
    WNDCLASSEXW wc{}; wc.cbSize=sizeof(wc);
    wc.lpszClassName = L"NNUETrainer";
    wc.hInstance     = g_hInst;
    wc.lpfnWndProc   = WndProc;
    wc.hbrBackground = g_brBg;
    wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);

    // Create main window
    int W2=1100, H2=700;
    g_hWnd = CreateWindowExW(0, L"NNUETrainer", L"NNUE Training Runner",
                              WS_OVERLAPPEDWINDOW,
                              CW_USEDEFAULT,CW_USEDEFAULT, W2,H2,
                              nullptr, nullptr, g_hInst, nullptr);

    // Custom tooltip popup (owned by main window, floats above everything)
    g_hTipWnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
                                L"TipWnd", nullptr, WS_POPUP,
                                0, 0, 10, 10,
                                g_hWnd, nullptr, g_hInst, nullptr);

    // Config pane (scrollable child)
    g_hCfgPane = CreateWindowExW(0, L"STATIC", nullptr,
                                  WS_CHILD|WS_VISIBLE|SS_LEFT,
                                  0, TITLE_H, PANEL_W, H2-TITLE_H-60,
                                  g_hWnd, nullptr, g_hInst, nullptr);
    SetWindowSubclass(g_hCfgPane, PanelProc, 1, 0);
    BuildConfigPane(g_hCfgPane, PANEL_W);

    // Graph
    int rightX = PANEL_W + 4;
    int rightW = W2 - rightX - 8;
    int graphH = H2 - LOG_H - PROG_H - TITLE_H - 80;
    g_hGraph = CreateWindowExW(0, L"NNUEGraph", nullptr,
                                WS_CHILD|WS_VISIBLE,
                                rightX, TITLE_H, rightW, graphH,
                                g_hWnd, nullptr, g_hInst, nullptr);

    // Log listbox
    int logY = TITLE_H + graphH + 4;
    g_hLog = CreateWindowExW(WS_EX_CLIENTEDGE, L"LISTBOX", nullptr,
                              WS_CHILD|WS_VISIBLE|LBS_NOINTEGRALHEIGHT|WS_VSCROLL|LBS_NOTIFY|LBS_OWNERDRAWFIXED|LBS_HASSTRINGS,
                              rightX, logY, rightW, LOG_H,
                              g_hWnd, (HMENU)(LONG_PTR)ID_LOG_BOX, g_hInst, nullptr);
    SendMessageW(g_hLog, WM_SETFONT, (WPARAM)g_fMono, TRUE);

    // Progress bar
    int progY = logY + LOG_H + 4;
    g_hProg = CreateWindowExW(0, PROGRESS_CLASSW, nullptr,
                               WS_CHILD|WS_VISIBLE|PBS_SMOOTH,
                               rightX, progY, rightW, PROG_H,
                               g_hWnd, (HMENU)(LONG_PTR)ID_PROGRESS, g_hInst, nullptr);
    SendMessageW(g_hProg, PBM_SETRANGE, 0, MAKELPARAM(0,100));
    SendMessageW(g_hProg, PBM_SETBARCOLOR, 0, (LPARAM)C_ACCENT);
    SendMessageW(g_hProg, PBM_SETBKCOLOR,  0, (LPARAM)C_BG);

    // ETA Banner
    int bannerY = progY + PROG_H + 4;
    g_hBanner = CreateWindowExW(0, L"STATIC", L"Ready",
        WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOTIFY,
        0, 0, 400, BANNER_H, g_hWnd, nullptr, g_hInst, nullptr);
    SendMessageW(g_hBanner, WM_SETFONT, (WPARAM)g_fMono, TRUE);

    // Status text
    int stY = bannerY + BANNER_H + 4;
    g_hStatus = CreateWindowExW(0, L"STATIC", L"Ready",
                                 WS_CHILD|WS_VISIBLE|SS_LEFT,
                                 rightX, stY, rightW, 20,
                                 g_hWnd, (HMENU)(LONG_PTR)ID_STATUS_TXT, g_hInst, nullptr);
    SendMessageW(g_hStatus, WM_SETFONT, (WPARAM)g_fUI, TRUE);

    // Buttons (bottom of left panel)
    int btnY = H2 - 46;
    int btnW = (PANEL_W - 24) / 3;
    g_hStart = CreateWindowExW(0, L"BUTTON", L"\u25B6 Start",
                                WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                                8, btnY, btnW, 28,
                                g_hWnd, (HMENU)(LONG_PTR)ID_BTN_START, g_hInst, nullptr);
    SendMessageW(g_hStart, WM_SETFONT, (WPARAM)g_fUI, TRUE);

    g_hStop = CreateWindowExW(0, L"BUTTON", L"\u25A0 Stop",
                               WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                               8+btnW+4, btnY, btnW, 28,
                               g_hWnd, (HMENU)(LONG_PTR)ID_BTN_STOP, g_hInst, nullptr);
    SendMessageW(g_hStop, WM_SETFONT, (WPARAM)g_fUI, TRUE);
    EnableWindow(g_hStop, FALSE);

    g_hPause = CreateWindowExW(0, L"BUTTON", L"\u23F8 Pause",
                                WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                                8+btnW*2+8, btnY, btnW, 28,
                                g_hWnd, (HMENU)(LONG_PTR)ID_BTN_PAUSE, g_hInst, nullptr);
    SendMessageW(g_hPause, WM_SETFONT, (WPARAM)g_fUI, TRUE);
    EnableWindow(g_hPause, FALSE);

    // Initialize preset system
    InitBuiltinPresets();
    LoadCustomPresets();
    PopulatePresetCombo();
    ApplyPreset(1);  // default: Standard

    // Timer for UI updates
    SetTimer(g_hWnd, ID_TIMER, 500, nullptr);

    ShowWindow(g_hWnd, SW_SHOWDEFAULT);
    UpdateWindow(g_hWnd);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    if (g_worker.joinable()) g_worker.join();
    DeleteObject(g_fUI);
    DeleteObject(g_fMono);
    DeleteObject(g_brBg);
    DeleteObject(g_brPanel);
    return (int)msg.wParam;
}
