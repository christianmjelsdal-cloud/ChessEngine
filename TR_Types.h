#pragma once
// TR_Types.h  --  Types, constants, and utilities for NNUE Training GUI
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
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "wbemuuid.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")

#include <windows.h>
#include <cassert>
#include <mmsystem.h>
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
#include <ctime>
#include <cmath>
#include <memory>
#include <comdef.h>
#include <Wbemidl.h>

namespace fs = std::filesystem;

// ── Layout constants ──────────────────────────────────────────────
static const int PANEL_W  = 295;
static const int TITLE_H  = 36;
static const int LOG_H    = 170;
static const int PROG_H   = 8;
static const int BANNER_H = 52;

// ── Control IDs ───────────────────────────────────────────────────
enum {
    ID_BTN_START        = 1001,
    ID_BTN_STOP         = 1002,
    ID_BTN_PAUSE        = 1003,
    ID_BTN_SKIP         = 1004,
    ID_COMBO_PRESET     = 1043,  // FIX C-3: was 1004, collided with ID_BTN_SKIP
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
    ID_BTN_LATEST_GEN   = 1007,
    ID_BTN_BEST_GEN     = 1008,
    ID_BTN_BENCHMARK    = 1009,
    ID_EDIT_MAXPOS      = 1026,
    ID_EDIT_EARLYSTOP   = 1027,
    ID_CHK_COSINELR     = 1032,
    ID_EDIT_COSINET0    = 1033,
    ID_CHK_SWA          = 1034,
    ID_EDIT_SWASTART    = 1035,
    ID_EDIT_DRAWPCT     = 1036,
    ID_EDIT_FRCMIX      = 1037,
    ID_EDIT_REPLAYWIN   = 1038,
    ID_EDIT_REPLAYDECAY = 1039,
    ID_EDIT_ELOGAMES    = 1044,  // FIX C-3: was 1040, collided with ID_LOG_BOX
    ID_EDIT_SWAGAMES    = 1045,  // FIX C-3: was 1041, collided with ID_STATUS_TXT
    ID_EDIT_WDLALPHA    = 1046,
    ID_EDIT_WDLDRAWELO  = 1047,
    ID_CHK_GRAPH_LOSS   = 1050,
    ID_CHK_GRAPH_ACC    = 1051,
    ID_CHK_GRAPH_LR     = 1052,
    ID_CHK_GRAPH_PHASE  = 1053,
    ID_CHK_GRAPH_NPS    = 1063,
    ID_CHK_MUTE_SOUNDS  = 1060,  // BUG FIX: was 1054, collided with ID_EDIT_MIXDEPTH_PCT
    ID_EDIT_MIXDEPTH_PCT = 1054,
    ID_EDIT_MIXDEPTH_LOW = 1055,
    ID_EDIT_RESIGNCP     = 1056,
    ID_EDIT_CONTEMPT     = 1057,
    ID_EDIT_MAXPLIES     = 1058,
    ID_EDIT_DRAWCP       = 1059,
    ID_CHK_DEPTH_SHUFFLE = 1061,
    ID_EDIT_DEPTH_SHUFFLE_BIAS = 1062,
    // --- Self-play diversity settings ---
    ID_EDIT_OPENING_TEMP    = 1063,
    ID_EDIT_OPENING_PLIES   = 1064,
    ID_EDIT_SOFTMAX_PLIES   = 1065,
    ID_EDIT_SOFTMAX_TEMP    = 1066,
    ID_EDIT_ROOT_NOISE      = 1067,
    // --- Position recording filters ---
    ID_EDIT_RECORD_MIN_PLY  = 1068,
    ID_EDIT_RECORD_MAX_EVAL = 1069,
    // --- Adjudication fine-tuning ---
    ID_EDIT_RESIGN_COUNT    = 1070,
    ID_EDIT_DRAW_COUNT      = 1071,
    ID_EDIT_DRAW_MIN_PLY    = 1072,
    ID_EDIT_DRAW_ADJ_MOVES  = 1073,
    ID_EDIT_DRAW_ADJ_THRESH = 1074,
    ID_EDIT_DRAW_ADJ_MIN_MOVE = 1075,
    ID_TIMER            = 2001,
};

// ── Dark theme colors ─────────────────────────────────────────────
static const COLORREF C_BG     = RGB(16, 16, 24);
static const COLORREF C_PANEL  = RGB(22, 22, 32);
static const COLORREF C_TEXT   = RGB(215, 215, 228);
static const COLORREF C_DIM    = RGB(100, 100, 120);
static const COLORREF C_ACCENT = RGB(65, 125, 245);

// ── Data structures ───────────────────────────────────────────────
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
    double drawWeight    = 0.5;
    double mateBoost     = 3.0;
    double splRatio      = 0.4;
    double frcMix        = 0.0;
    int    replayWindow  = 3;
    double replayDecay   = 0.7;
    int    workers       = 12;
    int    depth         = 5;
    int    mixedDepthLow  = 4;
    double mixedDepthRatio= 0.0;  // FIX 6.33: use double for consistency
    bool   depthShuffle   = false; // When true, shuffled games sample depth from [mixedDepthLow, depth) with geometric weighting
    double depthShuffleBias = 2.0; // Geometric weight: P(d) ∝ bias^(d - low). Higher = more high-depth games.
    int    maxPositions  = 300000;
    int    earlyStop     = 10;
    bool   cosineLr      = true;
    int    cosineT0      = 0;      // 0 = plain cosine decay; >0 = warm restart every T0 epochs
    bool   swa           = true;
    int    swaStart      = 3;
    double drawPct       = 10.0;   // % of max-positions allocated to draws
    double wdlAlpha      = 0.5;    // WDL cross-entropy blend (0=MSE only, 1=CE only)
    double wdlDrawElo    = 100.0;  // Draw bandwidth in centipawns for WDL targets
    bool   eloValidate   = false;
    int    eloGames      = 100;
    int    swaGames      = 50;
    bool   overfitDetect = true;
    int    resignCp      = 500;
    int    contemptCp    = 25;
    int    maxPlies      = 250;
    int    drawCp        = 8;
    // --- Self-play diversity settings ---
    double openingTemp     = 1.5;    // softmax temperature for opening plies
    int    openingPlies    = 4;      // plies using softmax move selection at game start
    int    softmaxPlies    = 8;      // additional post-opening plies using softmax selection
    double softmaxTemp     = 0.5;    // temperature for post-opening softmax phase
    double rootNoiseEps    = 0.0;    // probability of replacing best move with random alt (0=off)
    // --- Position recording filters ---
    int    recordMinPly    = 10;     // don't record positions before this ply
    int    recordMaxEval   = 2500;   // don't record positions where |eval| > this
    // --- Adjudication fine-tuning ---
    int    resignCount     = 3;      // consecutive plies above resign threshold before resigning
    int    drawCount       = 6;      // consecutive plies below draw threshold before draw
    int    drawMinPly      = 40;     // earliest ply for standard draw adjudication
    int    drawAdjMoves    = 12;     // plies of near-zero eval for secondary draw adj
    int    drawAdjThreshold = 4;     // centipawn threshold for "dead equal"
    int    drawAdjMinMove  = 50;     // minimum move number for secondary draw adj
};

// Config extends Preset: inherits all training parameters, adds runtime paths.
// This eliminates the 39 duplicate fields that Config and Preset used to maintain separately.
struct Config : Preset {
    int    startGen      = 0;
    std::string // INFO [6.18]: exeName/pyScript/dataDir are hardcoded — safe from path traversal.
    // If made user-configurable in future, add path validation (reject ".." components).
    exeName  = "ChessEngine.exe";
    std::string pyScript = "train_nnue.py";
    std::string dataDir  = "assets";
    std::string modelDir = "assets";

    Config() { eloValidate = true; }  // Override Preset default (false → true for full runs)
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
    double openingLoss = 0.0;
    double middlegameLoss = 0.0;
    double endgameLoss = 0.0;
    bool   hasPhase = false;
    double nps = 0.0;   // nodes per second from self-play (stored per gen, same value for all steps in gen)
    bool   hasNps = false;
    bool   hasLoss = true;  // false for NPS-only points (train=0, no loss data)
};

struct GraphPanelBounds {
    float top    = 0;
    float bottom = 0;
    bool  active = false;
};

struct GraphState {
    bool showLoss = true, showAcc = true, showLR = true, showPhase = true, showNPS = true;
    int  hoverIdx = -1;
    POINT mousePt = {-1, -1};
    int  hoverPanel = -1;   // 0=Loss 1=Acc 2=LR 3=Phase 4=NPS  -1=none
    GraphPanelBounds panelBounds[5]; // indexed same as above
    std::atomic<bool> dirty{true};   // FIX 7: set when data changes, cleared after graph redraws
};

struct AppState {
    std::mutex              mtx;
    std::vector<TrainPoint> pts;
    std::deque<std::string> log;
    std::string             status   = "Ready";
    std::string             phase;
    int    curGen    = 0, totalGens  = 0;
    int    curEpoch  = 0, totalEpochs= 0;
    double lastTrain = 0.0, lastVal  = 0.0;
    int    lastElo   = 0;
    std::chrono::steady_clock::time_point pipelineStart;
    std::chrono::steady_clock::time_point phaseStart;
    long long pipelineTotalSec = 0;   // persists after run ends
    // Pause-aware timing: accumulate paused duration so elapsed timers freeze
    long long pausedPipelineSec = 0;  // total seconds spent paused (pipeline)
    long long pausedPhaseSec    = 0;  // total seconds spent paused (current phase)
    std::chrono::steady_clock::time_point pauseStart; // when pause began
    int completedGens = 0;
    long long lastGenCompletedSec = 0; // pipeline elapsed seconds at the moment the last gen finished
    double emaGenSec = 0.0;             // exponential moving average of per-gen duration
    long long prevGenCompletedSec = 0;  // pipeline elapsed seconds when the gen *before* last finished
    int batchEtaSec  = 0;   // countdown: batch ETA from Python ("ETA: Xs")
    int epochEtaSec  = 0;   // countdown: gen training ETA from Python ("Total ETA: Xs")
    int nextEpochSec = 0;   // countdown: next epoch ETA from Python ("Next: ~Xs")
    int selfPlayEtaSec = 0; // countdown: self-play ETA parsed from "[SelfPlay] ... ETA HH:MM:SS"
    std::chrono::steady_clock::time_point batchEtaStamp;
    std::chrono::steady_clock::time_point epochEtaStamp;
    std::chrono::steady_clock::time_point nextEpochStamp;
    std::chrono::steady_clock::time_point selfPlayEtaStamp;
    double curNps = 0.0;  // latest NPS parsed from self-play output
    std::atomic<bool> running{false};  // FIX 6.7: atomic for thread safety (read by UI without lock)
    std::atomic<bool> stopFlag{false};

    static const size_t MAX_LOG = 800;

    void pushLog(const std::string& s) {
        std::lock_guard<std::mutex> lk(mtx);
        if (!s.empty() && s[0] == '\r') {
            // \r-prefixed → overwrite last line (single updating progress line)
            std::string clean = s.substr(1);
            if (!log.empty()) log.back() = clean;
            else               log.push_back(clean);
        } else {
            log.push_back(s);
            if (log.size() > MAX_LOG) log.pop_front();
        }
        // Also write to the structured file log (g_fileLog).
        // Auto-classify level from message content for file logging.
        // NOTE: g_fileLog is defined in TR_Globals.cpp; we forward-declare
        // access via the extern in TR_Globals.h (included by callers).
        // This inline body is compiled in translation units that include
        // TR_Globals.h, so g_fileLog is visible.  For the rare case where
        // TR_Types.h is included without TR_Globals.h, the file-log call
        // is guarded by a check.
        _writeToFileLog(s);
    }

    // Helper: classify and forward a message to the file logger.
    // Defined out-of-line in TR_Pipeline.cpp to avoid header dependency on TR_Logger.h.
    void _writeToFileLog(const std::string& s);
    void setStatus(const std::string& s) { std::lock_guard<std::mutex> lk(mtx); status = s; }
    void setPhase (const std::string& s) { std::lock_guard<std::mutex> lk(mtx); phase  = s; }
    void pushPt(TrainPoint p) {
        std::lock_guard<std::mutex> lk(mtx);
        // Merge with existing point at same (gen, step) — preserves NPS when loss arrives
        // and preserves loss when NPS arrives, instead of wholesale replacement.
        bool merged = false;
        for (auto& existing : pts) {
            if (existing.gen == p.gen && existing.step == p.step) {
                if (p.hasLoss) {
                    existing.train          = p.train;
                    existing.val            = p.val;
                    existing.hasVal         = p.hasVal;
                    existing.accuracy       = p.accuracy;
                    existing.hasAcc         = p.hasAcc;
                    existing.lr             = p.lr;
                    existing.hasLR          = p.hasLR;
                    existing.openingLoss    = p.openingLoss;
                    existing.middlegameLoss = p.middlegameLoss;
                    existing.endgameLoss    = p.endgameLoss;
                    existing.hasPhase       = p.hasPhase;
                    existing.hasLoss        = true;
                }
                if (p.hasNps) {
                    existing.nps    = p.nps;
                    existing.hasNps = true;
                }
                merged = true;
                break;
            }
        }
        if (!merged) pts.push_back(p);
        if (p.hasLoss) {
            lastTrain = p.train;
            if (p.hasVal) lastVal = p.val;
        }
        // FIX 7: Signal graph needs redraw
        extern GraphState g_graph;
        g_graph.dirty.store(true, std::memory_order_relaxed);
    }
};


struct UIHandles {
    HWND hWnd = nullptr, hGraph = nullptr, hLog = nullptr, hStatus = nullptr;
    HWND hProg = nullptr, hBanner = nullptr;
    HWND hStart = nullptr, hStop = nullptr, hPause = nullptr, hSkip = nullptr;
    HWND hPreset = nullptr, hCfgPane = nullptr;
    HWND hChkElo = nullptr, hChkOvfit = nullptr;
    HWND hChkCosineLR = nullptr, hChkSWA = nullptr;
    HWND hChkGLoss = nullptr, hChkGAcc = nullptr, hChkGLR = nullptr, hChkGPhase = nullptr, hChkGNPS = nullptr;
    HWND hChkMute = nullptr;
    HWND hChkDepthShuffle = nullptr;
    HWND hBtnSave = nullptr, hBtnDel = nullptr;
    HWND hBenchmark = nullptr;
    HFONT fUI = nullptr, fMono = nullptr;
    HBRUSH brPanel = nullptr, brBg = nullptr;
    ULONG_PTR gdip = 0;
    std::map<int,HWND> edits;
};

struct BenchmarkResult {
    int    depth       = 0;
    double gps         = 0.0;   // games per second (EWMA from SelfPlay output)
    int    sampleGames = 0;
    double elapsedSec  = 0.0;
    bool   valid       = false;
};

struct TrainBenchResult {
    double posPerSec = 0.0;
    int    batchSize = 0;
    bool   valid     = false;
};

// (GraphPanelBounds and GraphState moved above AppState)

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

struct ProcessState {
    std::atomic<DWORD> activePid{0};
    std::atomic<bool>  pauseFlag{false};
    std::atomic<bool>  skipPhaseFlag{false};
    PROCESS_INFORMATION activePi{};
    std::mutex activePiMtx;
};

struct TipState {
    HWND hWnd = nullptr;
    HWND current = nullptr;
    std::map<HWND, const wchar_t*> tipMap;
};

struct SavePresetDlgData { wchar_t name[128]; bool ok; };

// ── Utilities ─────────────────────────────────────────────────────
inline std::wstring W(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring r(n - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &r[0], n);
    return r;
}
inline std::string N(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string r(n - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &r[0], n, nullptr, nullptr);
    return r;
}
inline std::string getEdit(HWND h) {
    int len = GetWindowTextLengthW(h) + 1;
    std::wstring w(len, 0); GetWindowTextW(h, &w[0], len);
    w.resize(wcslen(w.c_str())); return N(w);
}
inline void setEdit(HWND h, const std::string& s) { SetWindowTextW(h, W(s).c_str()); }
inline double pDbl(const std::string& s, double d=0){ try{return std::stod(s);}catch(...){return d;} }
inline int    pInt (const std::string& s, int    d=0){ try{return std::stoi(s);}catch(...){return d;} }

inline std::string exeDir() {
    // FIX 6.31: Use a growing buffer to support long paths (>MAX_PATH).
    DWORD sz = MAX_PATH;
    std::wstring buf(sz, L'\0');
    for (;;) {
        DWORD len = GetModuleFileNameW(nullptr, &buf[0], sz);
        if (len == 0) return ".";          // API failure fallback
        if (len < sz) { buf.resize(len); break; }
        sz *= 2;
        buf.resize(sz, L'\0');
    }
    auto i = buf.find_last_of(L"\\/");
    return N(i != std::wstring::npos ? buf.substr(0, i) : L".");
}
inline std::string dbl2s(double v, int p=6) {
    std::ostringstream ss; ss << std::fixed << std::setprecision(p) << v;
    std::string s = ss.str();
    if (s.find('.') != std::string::npos) {
        s.erase(s.find_last_not_of('0') + 1);
        if (s.back() == '.') s += '0';  // preserve at least one decimal place (e.g. "1.0")
    }
    return s;
}
