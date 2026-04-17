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
#pragma comment(lib, "winmm.lib")

#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <tlhelp32.h>
#include <objidl.h>
#include <mmsystem.h>
#pragma warning(push, 0)
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

// ── Data structures ───────────────────────────────────────────────
enum class ChessVariant { Standard = 0, DuckChess = 1, Automate = 2 };

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
    int    mixedDepthLow   = 4;
    double mixedDepthRatio = 0.0;
    bool   depthShuffle    = false;
    double depthShuffleBias = 2.0;
    int    maxPositions  = 300000;
    int    earlyStop     = 10;
    bool   cosineLr      = true;
    int    cosineT0      = 0;
    bool   swa           = true;
    int    swaStart      = 3;
    double drawPct       = 10.0;
    double wdlAlpha      = 0.5;
    double wdlDrawElo    = 100.0;
    bool   eloValidate   = false;
    int    eloGames      = 100;
    int    swaGames      = 50;
    bool   overfitDetect = true;
    int    resignCp      = 500;
    int    contemptCp    = 25;
    int    maxPlies      = 250;
    int    drawCp        = 8;
    double openingTemp     = 1.5;
    int    openingPlies    = 4;
    int    softmaxPlies    = 8;
    double softmaxTemp     = 0.5;
    double rootNoiseEps    = 0.0;
    int    recordMinPly    = 10;
    int    recordMaxEval   = 2500;
    int    resignCount     = 3;
    int    drawCount       = 6;
    int    drawMinPly      = 40;
    int    drawAdjMoves    = 12;
    int    drawAdjThreshold = 4;
    int    drawAdjMinMove  = 50;
};

struct Config : Preset {
    int    startGen      = 0;
    ChessVariant variant = ChessVariant::Standard;
    std::string exeName  = "ChessEngine.exe";
    std::string pyScript = "train_nnue.py";
    std::string dataDir  = "assets";
    std::string modelDir = "assets";

    Config() { eloValidate = true; }

    std::string weightsBaseName() const {
        switch (variant) {
            case ChessVariant::DuckChess: return "duck_nnue_weights";
            case ChessVariant::Automate:  return "automate_play_weights";
            default:                      return "nnue_weights";
        }
    }
    std::string trainingDataName() const {
        switch (variant) {
            case ChessVariant::DuckChess: return "duck_training_data.bin";
            case ChessVariant::Automate:  return "automate_training_data.bin";
            default:                      return "training_data.bin";
        }
    }
    std::string selfplayPrefix() const {
        switch (variant) {
            case ChessVariant::DuckChess: return "duck_selfplay_gen";
            case ChessVariant::Automate:  return "automate_selfplay_gen";
            default:                      return "selfplay_gen";
        }
    }
    std::string variantFlag() const {
        switch (variant) {
            case ChessVariant::DuckChess: return " --duck-chess";
            case ChessVariant::Automate:  return " --automate-chess";
            default:                      return "";
        }
    }
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
    double openingLoss    = 0.0;
    double middlegameLoss = 0.0;
    double endgameLoss    = 0.0;
    bool   hasPhase = false;
    double nps    = 0.0;
    bool   hasNps = false;
    bool   hasLoss = true;
};

struct AppState {
    std::mutex              mtx;
    std::vector<TrainPoint> pts;
    std::deque<std::string> log;
    std::string             lastLogSig;
    bool                    logLastReplaced = false;
    std::string             status   = "Ready";
    std::string             phase;
    int    curGen    = 0, totalGens  = 0;
    int    curEpoch  = 0, totalEpochs= 0;
    double lastTrain = 0.0, lastVal  = 0.0;
    int    lastElo   = 0;
    std::chrono::steady_clock::time_point pipelineStart;
    std::chrono::steady_clock::time_point phaseStart;
    int completedGens = 0;
    int batchEtaSec  = 0;
    int epochEtaSec  = 0;
    int nextEpochSec = 0;
    int selfPlayEtaSec = 0;   // countdown: self-play ETA from "[SelfPlay] ... ETA HH:MM:SS"
    int curBatch     = 0;
    int totalBatches = 0;
    std::chrono::steady_clock::time_point batchEtaStamp;
    std::chrono::steady_clock::time_point epochEtaStamp;
    std::chrono::steady_clock::time_point nextEpochStamp;
    std::chrono::steady_clock::time_point selfPlayEtaStamp;
    std::chrono::steady_clock::time_point lastSelfPlayPrint;  // when engine last printed a [SelfPlay] line
    bool   running   = false;
    std::atomic<bool> stopFlag{false};
    double curNps    = 0.0;  // latest NPS parsed from self-play output

    static const size_t MAX_LOG = 800;

    void pushLog(const std::string& s) {
        if (s.empty()) return;
        std::lock_guard<std::mutex> lk(mtx);

        // \r prefix = overwrite last line (running progress update)
        // No \r prefix = append as new line (distinct event)
        if (s[0] == '\r') {
            std::string clean = s.substr(1);
            if (!log.empty()) {
                log.back() = clean;
                logLastReplaced = true;
            } else {
                log.push_back(clean);
                logLastReplaced = false;
            }
        } else {
            log.push_back(s);
            if (log.size() > MAX_LOG) log.pop_front();
            logLastReplaced = false;
        }
    }
    void setStatus(const std::string& s) { std::lock_guard<std::mutex> lk(mtx); status = s; }
    void setPhase (const std::string& s) { std::lock_guard<std::mutex> lk(mtx); phase  = s; }
    void pushPt(TrainPoint p) {
        std::lock_guard<std::mutex> lk(mtx);
        // Merge with existing point at same (gen, step) rather than wholesale replace.
        // This preserves NPS fields when a training loss point arrives for the same step,
        // and preserves loss fields when an NPS point arrives for the same step.
        bool merged = false;
        for (auto& existing : pts) {
            if (existing.gen == p.gen && existing.step == p.step) {
                // Incoming has loss data → update all loss fields, keep existing NPS
                if (p.hasLoss) {
                    existing.train        = p.train;
                    existing.val          = p.val;
                    existing.hasVal       = p.hasVal;
                    existing.accuracy     = p.accuracy;
                    existing.hasAcc       = p.hasAcc;
                    existing.lr           = p.lr;
                    existing.hasLR        = p.hasLR;
                    existing.openingLoss    = p.openingLoss;
                    existing.middlegameLoss = p.middlegameLoss;
                    existing.endgameLoss    = p.endgameLoss;
                    existing.hasPhase     = p.hasPhase;
                    existing.hasLoss      = true;
                }
                // Incoming has NPS data → update NPS fields, keep existing loss
                if (p.hasNps) {
                    existing.nps    = p.nps;
                    existing.hasNps = true;
                }
                merged = true;
                break;
            }
        }
        if (!merged) pts.push_back(p);
        // Keep sorted by (gen, step) so graph always renders in order
        std::sort(pts.begin(), pts.end(), [](const TrainPoint& a, const TrainPoint& b) {
            return a.gen != b.gen ? a.gen < b.gen : a.step < b.step;
        });
        if (p.hasLoss) {
            lastTrain = p.train;
            if (p.hasVal) lastVal = p.val;
        }
    }
};

// ── Globals ───────────────────────────────────────────────────────
static AppState   g_st;
static Config     g_cfg;
static HINSTANCE  g_hInst    = nullptr;
static std::thread g_worker;
static std::vector<Preset> g_allPresets;
static int g_currentPresetIdx = 1;
static HWND g_hWnd     = nullptr;
static HWND g_hGraph   = nullptr;
static HWND g_hLog     = nullptr;
static HWND g_hStatus  = nullptr;
static HWND g_hProg    = nullptr;
static HWND g_hBanner  = nullptr;
static HWND g_hStart   = nullptr;
static HWND g_hStop    = nullptr;
static HWND g_hPause   = nullptr;
static HWND g_hSkip    = nullptr;
static HWND g_hPreset  = nullptr;
static HWND g_hCfgPane = nullptr;
static HFONT      g_fUI      = nullptr;
static HFONT      g_fMono    = nullptr;
static HBRUSH     g_brPanel  = nullptr;
static HBRUSH     g_brBg     = nullptr;
static ULONG_PTR  g_gdip     = 0;
static std::map<int,HWND> g_edits;
static HWND g_hChkElo = nullptr, g_hChkOvfit = nullptr;
static HWND g_hChkCosineLR = nullptr, g_hChkSWA = nullptr;
static HWND g_hChkDepthShuffle = nullptr;
static HWND g_hBtnSave = nullptr, g_hBtnDel = nullptr;
static HWND g_hVariant = nullptr;
static bool g_showLoss = true;
static bool g_showAcc  = true;
static bool g_showLR   = true;
static bool g_showPhase = true;
static bool g_showNPS  = true;
static HWND g_hChkGLoss = nullptr, g_hChkGAcc = nullptr, g_hChkGLR = nullptr, g_hChkGPhase = nullptr, g_hChkGNPS = nullptr;
static HWND g_hChkMute = nullptr;
static int   g_graphHoverIdx = -1;
static POINT g_graphMousePt  = {-1, -1};
static int g_cfgTotalH  = 0;
static int g_cfgScrollY = 0;

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
    ID_BTN_SKIP         = 1004,
    ID_COMBO_PRESET     = 1043,
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
    ID_EDIT_MAXPOS      = 1026,
    ID_EDIT_EARLYSTOP   = 1027,
    ID_CHK_ELO          = 1030,
    ID_CHK_OVERFIT      = 1031,
    ID_CHK_COSINELR     = 1032,
    ID_EDIT_COSINET0    = 1033,
    ID_CHK_SWA          = 1034,
    ID_EDIT_SWASTART    = 1035,
    ID_EDIT_DRAWPCT     = 1036,
    ID_EDIT_FRCMIX      = 1037,
    ID_EDIT_REPLAYWIN   = 1038,
    ID_EDIT_REPLAYDECAY = 1039,
    ID_LOG_BOX          = 1040,
    ID_STATUS_TXT       = 1041,
    ID_PROGRESS         = 1042,
    ID_EDIT_ELOGAMES    = 1044,
    ID_EDIT_SWAGAMES    = 1045,
    ID_EDIT_WDLALPHA    = 1046,
    ID_EDIT_WDLDRAWELO  = 1047,
    ID_BTN_SAVE_PRESET  = 1005,
    ID_BTN_DEL_PRESET   = 1006,
    ID_BTN_LATEST_GEN   = 1007,
    ID_BTN_BEST_GEN     = 1008,
    ID_BTN_BENCHMARK    = 1009,
    ID_CHK_GRAPH_LOSS   = 1050,
    ID_CHK_GRAPH_ACC    = 1051,
    ID_CHK_GRAPH_LR     = 1052,
    ID_CHK_GRAPH_PHASE  = 1053,
    ID_CHK_GRAPH_NPS    = 1063,
    ID_EDIT_MIXDEPTH_PCT = 1054,
    ID_EDIT_MIXDEPTH_LOW = 1055,
    ID_EDIT_RESIGNCP     = 1056,
    ID_EDIT_CONTEMPT     = 1057,
    ID_EDIT_MAXPLIES     = 1058,
    ID_EDIT_DRAWCP       = 1059,
    ID_CHK_MUTE_SOUNDS   = 1060,
    ID_CHK_DEPTH_SHUFFLE = 1061,
    ID_EDIT_DEPTH_SHUFFLE_BIAS = 1062,
    ID_EDIT_OPENING_TEMP    = 1063,
    ID_EDIT_OPENING_PLIES   = 1064,
    ID_EDIT_SOFTMAX_PLIES   = 1065,
    ID_EDIT_SOFTMAX_TEMP    = 1066,
    ID_EDIT_ROOT_NOISE      = 1067,
    ID_EDIT_RECORD_MIN_PLY  = 1068,
    ID_EDIT_RECORD_MAX_EVAL = 1069,
    ID_EDIT_RESIGN_COUNT    = 1070,
    ID_EDIT_DRAW_COUNT      = 1071,
    ID_EDIT_DRAW_MIN_PLY    = 1072,
    ID_EDIT_DRAW_ADJ_MOVES  = 1073,
    ID_EDIT_DRAW_ADJ_THRESH = 1074,
    ID_EDIT_DRAW_ADJ_MIN_MOVE = 1075,
    ID_COMBO_VARIANT    = 1076,
    ID_TIMER            = 2001,
};

// ── Dark theme colors ─────────────────────────────────────────────
static const COLORREF C_BG     = RGB(16, 16, 24);
static const COLORREF C_PANEL  = RGB(22, 22, 32);
static const COLORREF C_TEXT   = RGB(215, 215, 228);
static const COLORREF C_DIM    = RGB(100, 100, 120);
static const COLORREF C_ACCENT = RGB(65, 125, 245);




// Active process tracking for stop/pause
static std::atomic<DWORD> g_activePid{0};
static HANDLE              g_hJob = NULL;   // Job Object — kills children when TrainingRunner exits
static std::atomic<bool>  g_pauseFlag{false};
static std::atomic<bool>  g_skipFlag{false};   // skip current phase, continue pipeline
static PROCESS_INFORMATION g_activePi{};
static std::mutex          g_activePiMtx;

static bool g_muteSounds = false;  // controlled by Mute Sounds checkbox

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

// ── Sound playback (non-blocking, fire-and-forget via MCI) ────────
static void PlayMp3(const std::string& alias, const std::string& dataDir, const std::string& filename) {
    if (g_muteSounds) return;
    fs::path soundPath = fs::path(exeDir()) / dataDir / filename;
    if (!fs::exists(soundPath)) {
        soundPath = fs::path(exeDir()) / "assets" / filename;
        if (!fs::exists(soundPath)) return;
    }
    std::string close_ = "close " + alias;
    std::string open_  = "open \"" + soundPath.string() + "\" type mpegvideo alias " + alias;
    std::string play_  = "play " + alias;
    mciSendStringA(close_.c_str(), NULL, 0, NULL);
    mciSendStringA(open_.c_str(),  NULL, 0, NULL);
    mciSendStringA(play_.c_str(),  NULL, 0, NULL);
}

// -- Preset management --
static std::string presetFilePath() {
    return exeDir() + "\\custom_presets.cfg";
}

// ── File logger ───────────────────────────────────────────────────
// Writes a structured log to assets/logs/training_run_YYYYMMDD_HHMMSS.log
struct RunLogger {
    std::ofstream file;
    std::string   path;
    std::mutex    mtx;

    static std::string timestamp() {
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) % 1000;
        std::tm tm{};
        localtime_s(&tm, &time);
        std::ostringstream ss;
        ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    void open(const std::string& dataDir) {
        std::lock_guard<std::mutex> lk(mtx);
        fs::path logDir = fs::path(exeDir()) / dataDir / "logs";
        fs::create_directories(logDir);
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm{}; localtime_s(&tm, &time);
        std::ostringstream fname;
        fname << "training_run_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".log";
        path = (logDir / fname.str()).string();
        file.open(path, std::ios::out | std::ios::trunc);
        if (file.is_open()) {
            file << "# NNUE Training Runner - Structured Log\n";
            file << "# Format: [TIMESTAMP] [LEVEL] [PHASE:GEN] message\n";
            file << "# Generated: " << timestamp() << "\n#\n";
            file.flush();
        }
    }

    void close() {
        std::lock_guard<std::mutex> lk(mtx);
        if (file.is_open()) {
            file << "# END OF LOG " << timestamp() << "\n";
            file.close();
        }
    }

    bool isOpen() const { return file.is_open(); }

    void write(const std::string& level, const std::string& phase, int gen, const std::string& msg) {
        if (msg.empty() || msg[0] == '\r') return;
        std::lock_guard<std::mutex> lk(mtx);
        if (!file.is_open()) return;
        file << '[' << timestamp() << "] [" << level << "] ["
             << phase << ':' << gen << "] " << msg << '\n';
        file.flush();
    }

    void info (const std::string& phase, int gen, const std::string& msg) { write("INFO",  phase, gen, msg); }
    void warn (const std::string& phase, int gen, const std::string& msg) { write("WARN",  phase, gen, msg); }
    void error(const std::string& phase, int gen, const std::string& msg) { write("ERROR", phase, gen, msg); }
    void event(const std::string& phase, int gen, const std::string& msg) { write("EVENT", phase, gen, msg); }
    void metric(int gen, int epoch, double train, double val, bool hasVal, double lr, bool hasLR) {
        std::ostringstream ss;
        ss << "EPOCH epoch=" << epoch
           << " train_loss=" << std::fixed << std::setprecision(8) << train;
        if (hasVal) ss << " val_loss=" << std::fixed << std::setprecision(8) << val;
        if (hasLR)  ss << " lr=" << std::scientific << std::setprecision(6) << lr;
        write("METRIC", "training", gen, ss.str());
    }
};
static RunLogger g_log;

// ── Gen stats persistence ─────────────────────────────────────────
static std::string genStatsPath(const std::string& dataDir) {
    // Determine variant prefix from the dataDir name or from a separate param.
    // We use a simple convention: if dataDir contains "duck" use duck prefix.
    // The caller passes cfg.dataDir which is "assets" for both, so we rely on
    // the weights file prefix instead — but genStatsPath only gets dataDir.
    // Solution: store variant in the filename itself based on what's in dataDir.
    // Since both variants share "assets", we use a separate file per variant.
    // The variant is encoded by the caller via a separate overload below.
    return exeDir() + "\\" + dataDir + "\\gen_stats.csv";
}

static std::string genStatsPathForVariant(const std::string& dataDir, ChessVariant variant) {
    std::string prefix = (variant == ChessVariant::DuckChess) ? "duck_" :
                         (variant == ChessVariant::Automate)  ? "automate_" : "";
    return exeDir() + "\\" + dataDir + "\\" + prefix + "gen_stats.csv";
}

// Variant-specific graph CSV path — each variant has its own training history
static fs::path graphCsvPath(ChessVariant variant) {
    std::string prefix = (variant == ChessVariant::DuckChess) ? "duck_" :
                         (variant == ChessVariant::Automate)  ? "automate_" : "";
    fs::path graphDir = fs::path(exeDir()) / "training progress";
    fs::create_directories(graphDir);
    return graphDir / (prefix + "training_graph.csv");
}

// Load graph CSV into g_st.pts (caller must hold g_st.mtx or call before threads start)
static void LoadGraphCsv(ChessVariant variant) {
    fs::path graphCsv = graphCsvPath(variant);
    g_st.pts.clear();
    if (!fs::exists(graphCsv)) return;
    std::ifstream in(graphCsv.string());
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        TrainPoint p;
        std::istringstream ss(line);
        std::string tok;
        try {
            if (!std::getline(ss, tok, ',')) continue; p.gen  = std::stoi(tok);
            if (!std::getline(ss, tok, ',')) continue; p.step = std::stoi(tok);
            if (!std::getline(ss, tok, ',')) continue; p.train = std::stod(tok);
            if (std::getline(ss, tok, ',') && tok != "nan" && !tok.empty()) { p.val = std::stod(tok); p.hasVal = true; }
            if (std::getline(ss, tok, ',') && tok != "nan" && !tok.empty()) { p.lr  = std::stod(tok); p.hasLR  = true; }
            if (std::getline(ss, tok, ',') && tok != "nan" && !tok.empty()) { p.accuracy = std::stod(tok); p.hasAcc = true; }
            std::string opTok, mgTok, egTok;
            if (std::getline(ss, opTok, ',') && std::getline(ss, mgTok, ',') && std::getline(ss, egTok, ',')) {
                if (opTok != "nan" && mgTok != "nan" && egTok != "nan" &&
                    !opTok.empty() && !mgTok.empty() && !egTok.empty()) {
                    p.openingLoss    = std::stod(opTok);
                    p.middlegameLoss = std::stod(mgTok);
                    p.endgameLoss    = std::stod(egTok);
                    p.hasPhase = true;
                }
            }
            if (p.train > 10.0 || p.train < 0.0) continue;  // sanity: skip garbage entries
            g_st.pts.push_back(p);
        } catch (...) {}
    }
    // Sort by (gen, step) and deduplicate — handles replayed gens in old CSV files
    std::sort(g_st.pts.begin(), g_st.pts.end(), [](const TrainPoint& a, const TrainPoint& b) {
        return a.gen != b.gen ? a.gen < b.gen : a.step < b.step;
    });
    // Remove duplicate (gen, step) pairs — keep last occurrence (most recent run)
    auto it = std::unique(g_st.pts.begin(), g_st.pts.end(), [](const TrainPoint& a, const TrainPoint& b) {
        return a.gen == b.gen && a.step == b.step;
    });
    g_st.pts.erase(it, g_st.pts.end());
}

static int findLatestGen(const std::string& dataDir, ChessVariant variant = ChessVariant::Standard) {
    fs::path dir = fs::path(exeDir()) / dataDir;
    int best = 0;
    if (!fs::is_directory(dir)) return 0;
    // Build the prefix to search for based on variant
    std::string prefix = (variant == ChessVariant::DuckChess) ? "duck_nnue_weights_gen" :
                         (variant == ChessVariant::Automate)  ? "automate_play_weights_gen" :
                                                                 "nnue_weights_gen";
    for (auto& ent : fs::directory_iterator(dir)) {
        std::string name = ent.path().filename().string();
        if (name.rfind(prefix, 0) == 0) {
            size_t numStart = prefix.size();
            size_t dot = name.find('.', numStart);
            if (dot != std::string::npos) {
                try { int g = std::stoi(name.substr(numStart, dot - numStart));
                      if (g > best) best = g; } catch (...) {}
            }
        }
    }
    return best;
}

static int findBestGenFor(const std::string& dataDir, ChessVariant variant = ChessVariant::Standard) {
    std::map<int, double> bestVal;
    {
        std::ifstream in(genStatsPathForVariant(dataDir, variant));
        std::string line;
        while (std::getline(in, line)) {
            auto c = line.find(',');
            if (c == std::string::npos) continue;
            try {
                int g = std::stoi(line.substr(0, c));
                double v = std::stod(line.substr(c + 1));
                auto it = bestVal.find(g);
                if (it == bestVal.end() || v < it->second) bestVal[g] = v;
            } catch (...) {}
        }
    }
    { std::lock_guard<std::mutex> lk(g_st.mtx);
      for (auto& p : g_st.pts) {
        if (!p.hasVal) continue;
        auto it = bestVal.find(p.gen);
        if (it == bestVal.end() || p.val < it->second) bestVal[p.gen] = p.val;
      }
    }
    if (bestVal.empty()) return 0;
    int bestG = 0; double bestV = 1e9;
    for (auto& [g, v] : bestVal) { if (v < bestV) { bestV = v; bestG = g; } }
    return bestG;
}

// ── Preset serialization ──────────────────────────────────────────
static std::string SerializePreset(const Preset& p) {
    std::ostringstream o;
    o << p.name << "|"
      << p.generations << "|" << p.gamesPerGen << "|" << p.epochsPerGen << "|"
      << p.batchSize << "|" << p.workers << "|" << p.depth << "|"
      << p.gradAccum << "|" << p.warmupSteps << "|"
      << dbl2s(p.lr,8) << "|" << dbl2s(p.weightDecay,8) << "|"
      << dbl2s(p.dropout,4) << "|" << dbl2s(p.labelSmooth,4) << "|"
      << dbl2s(p.drawWeight,4) << "|" << dbl2s(p.mateBoost,4) << "|"
      << dbl2s(p.splRatio,4) << "|" << 0 << "|"
      << (p.eloValidate?1:0) << "|" << p.eloGames << "|" << p.swaGames << "|" << (p.overfitDetect?1:0) << "|"
      << p.maxPositions << "|" << p.earlyStop << "|"
      << (p.cosineLr?1:0) << "|" << p.cosineT0 << "|"
      << (p.swa?1:0) << "|" << p.swaStart << "|"
      << dbl2s(p.drawPct,2) << "|" << dbl2s(p.frcMix,3) << "|" << p.replayWindow << "|" << dbl2s(p.replayDecay,2) << "|"
      << p.resignCp << "|" << p.contemptCp << "|" << p.maxPlies << "|" << p.drawCp
      << "|" << p.mixedDepthLow << "|" << dbl2s(p.mixedDepthRatio,3)
      << "|" << dbl2s(p.wdlAlpha,4) << "|" << dbl2s(p.wdlDrawElo,2)
      << "|" << (p.depthShuffle?1:0) << "|" << dbl2s(p.depthShuffleBias,2)
      << "|" << dbl2s(p.openingTemp,2) << "|" << p.openingPlies
      << "|" << p.softmaxPlies << "|" << dbl2s(p.softmaxTemp,2)
      << "|" << dbl2s(p.rootNoiseEps,3)
      << "|" << p.recordMinPly << "|" << p.recordMaxEval
      << "|" << p.resignCount << "|" << p.drawCount << "|" << p.drawMinPly
      << "|" << p.drawAdjMoves << "|" << p.drawAdjThreshold << "|" << p.drawAdjMinMove;
    return o.str();
}

static Preset DeserializePreset(const std::string& line, bool builtin) {
    std::istringstream ss(line);
    Preset p; p.isBuiltin = builtin;
    auto next = [&]() -> std::string { std::string t; if (std::getline(ss, t, '|')) return t; return ""; };
    p.name = next(); if (p.name.empty()) return p;
    p.generations  = pInt(next(), 10);  p.gamesPerGen  = pInt(next(), 5000);
    p.epochsPerGen = pInt(next(), 10);  p.batchSize    = pInt(next(), 2048);
    p.workers      = pInt(next(), 12);  p.depth        = pInt(next(), 5);
    p.gradAccum    = pInt(next(), 4);   p.warmupSteps  = pInt(next(), 500);
    p.lr           = pDbl(next(), 0.001); p.weightDecay = pDbl(next(), 1e-5);
    p.dropout      = pDbl(next(), 0.1); p.labelSmooth  = pDbl(next(), 0.05);
    p.drawWeight   = pDbl(next(), 0.5); p.mateBoost    = pDbl(next(), 3.0);
    p.splRatio     = pDbl(next(), 0.4);
    next(); // startGen placeholder
    p.eloValidate  = pInt(next(), 0) != 0; p.eloGames = pInt(next(), 100);
    p.swaGames     = pInt(next(), 50);  p.overfitDetect = pInt(next(), 1) != 0;
    p.maxPositions = pInt(next(), 300000); p.earlyStop = pInt(next(), 10);
    p.cosineLr     = pInt(next(), 1) != 0; p.cosineT0 = pInt(next(), 0);
    p.swa          = pInt(next(), 1) != 0; p.swaStart = pInt(next(), 3);
    p.drawPct      = pDbl(next(), 10.0); p.frcMix = pDbl(next(), 0.0);
    p.replayWindow = pInt(next(), 3);   p.replayDecay = pDbl(next(), 0.7);
    p.resignCp     = pInt(next(), 500); p.contemptCp = pInt(next(), 25);
    p.maxPlies     = pInt(next(), 250); p.drawCp = pInt(next(), 8);
    p.mixedDepthLow   = pInt(next(), 4); p.mixedDepthRatio = pDbl(next(), 0.0);
    p.wdlAlpha        = pDbl(next(), 0.5); p.wdlDrawElo = pDbl(next(), 100.0);
    p.depthShuffle    = pInt(next(), 0) != 0; p.depthShuffleBias = pDbl(next(), 2.0);
    p.openingTemp     = pDbl(next(), 1.5); p.openingPlies = pInt(next(), 4);
    p.softmaxPlies    = pInt(next(), 8);   p.softmaxTemp  = pDbl(next(), 0.5);
    p.rootNoiseEps    = pDbl(next(), 0.0);
    p.recordMinPly    = pInt(next(), 10);  p.recordMaxEval = pInt(next(), 2500);
    p.resignCount     = pInt(next(), 3);   p.drawCount = pInt(next(), 6);
    p.drawMinPly      = pInt(next(), 40);  p.drawAdjMoves = pInt(next(), 12);
    p.drawAdjThreshold = pInt(next(), 4);  p.drawAdjMinMove = pInt(next(), 50);
    return p;
}

static void InitBuiltinPresets() {
    g_allPresets.clear();
    // Quick Test
    { Preset p; p.name="Quick Test"; p.isBuiltin=true;
      p.generations=3; p.gamesPerGen=50; p.epochsPerGen=3;
      p.batchSize=1024; p.lr=0.001; p.weightDecay=1e-5;
      p.dropout=0.0; p.labelSmooth=0.0; p.gradAccum=2;
      p.warmupSteps=3; p.drawWeight=1.0; p.mateBoost=3.0;
      p.splRatio=0.4; p.workers=12; p.depth=5;
      p.eloValidate=false; p.eloGames=20; p.swaGames=10; p.overfitDetect=false;
      p.maxPositions=100000; p.earlyStop=5;
      p.cosineLr=true; p.swa=false; p.swaStart=3;
      p.drawPct=10.0; p.wdlAlpha=0.5; p.wdlDrawElo=100.0;
      g_allPresets.push_back(p); }
    // 1 Hour
    { Preset p; p.name="1 Hour"; p.isBuiltin=true;
      p.generations=5; p.gamesPerGen=750; p.epochsPerGen=12;
      p.batchSize=2048; p.lr=0.0005; p.weightDecay=1e-4;
      p.dropout=0.04; p.labelSmooth=0.02; p.gradAccum=4;
      p.warmupSteps=20; p.drawWeight=1.0; p.mateBoost=3.0;
      p.splRatio=0.45; p.workers=12; p.depth=6;
      p.eloValidate=false; p.eloGames=30; p.swaGames=20; p.overfitDetect=true;
      p.maxPositions=100000; p.earlyStop=8;
      p.cosineLr=true; p.swa=true; p.swaStart=3;
      p.drawPct=10.0; p.wdlAlpha=0.5; p.wdlDrawElo=100.0;
      g_allPresets.push_back(p); }
    // 3 Hours
    { Preset p; p.name="3 Hours"; p.isBuiltin=true;
      p.generations=8; p.gamesPerGen=1000; p.epochsPerGen=12;
      p.batchSize=2048; p.lr=0.0005; p.weightDecay=1e-4;
      p.dropout=0.05; p.labelSmooth=0.02; p.gradAccum=4;
      p.warmupSteps=50; p.drawWeight=1.0; p.mateBoost=3.0;
      p.splRatio=0.4; p.workers=12; p.depth=7;
      p.eloValidate=true; p.eloGames=50; p.swaGames=30; p.overfitDetect=true;
      p.maxPositions=160000; p.earlyStop=8;
      p.cosineLr=true; p.swa=true; p.swaStart=3;
      p.drawPct=12.0; p.frcMix=0.10; p.wdlAlpha=0.5; p.wdlDrawElo=100.0;
      g_allPresets.push_back(p); }
    // Early Gen Training
    { Preset p; p.name="Early Gen Training"; p.isBuiltin=true;
      p.generations=12; p.gamesPerGen=3000; p.epochsPerGen=8;
      p.batchSize=2048; p.lr=0.001; p.weightDecay=1e-4;
      p.dropout=0.10; p.labelSmooth=0.05; p.gradAccum=4;
      p.warmupSteps=15; p.drawWeight=0.8; p.mateBoost=3.0;
      p.splRatio=0.50; p.workers=12; p.depth=5;
      p.eloValidate=false; p.eloGames=40; p.swaGames=20; p.overfitDetect=true;
      p.maxPositions=150000; p.earlyStop=6;
      p.cosineLr=true; p.swa=false; p.swaStart=3;
      p.drawPct=10.0; p.wdlAlpha=0.4; p.wdlDrawElo=100.0;
      g_allPresets.push_back(p); }
    // Mid Gen Training
    { Preset p; p.name="Mid Gen Training"; p.isBuiltin=true;
      p.generations=14; p.gamesPerGen=2000; p.epochsPerGen=10;
      p.batchSize=2048; p.lr=0.0006; p.weightDecay=1e-4;
      p.dropout=0.05; p.labelSmooth=0.03; p.gradAccum=4;
      p.warmupSteps=20; p.drawWeight=1.0; p.mateBoost=4.0;
      p.splRatio=0.42; p.workers=12; p.depth=7;
      p.eloValidate=true; p.eloGames=60; p.swaGames=30; p.overfitDetect=true;
      p.maxPositions=250000; p.earlyStop=7;
      p.cosineLr=true; p.swa=true; p.swaStart=5;
      p.drawPct=15.0; p.frcMix=0.10; p.wdlAlpha=0.5; p.wdlDrawElo=100.0;
      g_allPresets.push_back(p); }
    // Late Gen Training
    { Preset p; p.name="Late Gen Training"; p.isBuiltin=true;
      p.generations=15; p.gamesPerGen=4000; p.epochsPerGen=12;
      p.batchSize=2048; p.lr=0.0003; p.weightDecay=5e-5;
      p.dropout=0.02; p.labelSmooth=0.01; p.gradAccum=4;
      p.warmupSteps=25; p.drawWeight=1.3; p.mateBoost=5.0;
      p.splRatio=0.35; p.workers=12; p.depth=9;
      p.eloValidate=true; p.eloGames=100; p.swaGames=50; p.overfitDetect=true;
      p.maxPositions=400000; p.earlyStop=8;
      p.cosineLr=true; p.swa=true; p.swaStart=4;
      p.drawPct=20.0; p.frcMix=0.20; p.wdlAlpha=0.6; p.wdlDrawElo=100.0;
      g_allPresets.push_back(p); }
}

static void LoadCustomPresets() {
    std::string path = presetFilePath();
    std::ifstream f(path);
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        Preset p = DeserializePreset(line, false);
        if (!p.name.empty()) g_allPresets.push_back(p);
    }
}

static void SaveCustomPresets() {
    std::string path = presetFilePath();
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "# Custom presets\n";
    for (auto& p : g_allPresets) {
        if (p.isBuiltin) continue;
        f << SerializePreset(p) << "\n";
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
    RGB(210, 210, 210),   // LC_GENERAL  — soft white
    RGB(0,  180, 180),    // LC_SELFPLAY — teal
    RGB(100, 180, 255),   // LC_TRAINING — blue
    RGB(80,  220, 120),   // LC_SUCCESS  — soft green
    RGB(255, 80,  80),    // LC_ERROR    — soft red
    RGB(255, 210, 60),    // LC_PROGRESS — amber
    RGB(140, 140, 160),   // LC_META     — dim grey
};

static LogColor ClassifyLogLine(const std::string& line) {
    // Errors and warnings — check first (highest priority)
    if (line.find("[ERR]")   != std::string::npos ||
        line.find("[ERROR]") != std::string::npos ||
        line.find("[WARN]")  != std::string::npos ||
        line.find("WARNING") != std::string::npos ||
        line.find("failed")  != std::string::npos ||
        line.find("Failed")  != std::string::npos ||
        line.find("error:")  != std::string::npos ||
        line.find("Error:")  != std::string::npos)
        return LC_ERROR;

    // Self-play output lines
    if (line.find("[SelfPlay]") != std::string::npos ||
        line.find("SelfPlay]")  != std::string::npos ||  // handles \r prefix
        line.find("games/s")    != std::string::npos ||
        line.find("--generate") != std::string::npos)
        return LC_SELFPLAY;

    // Training epoch lines
    if (line.find("Train:") != std::string::npos ||
        line.find("Val:")   != std::string::npos ||
        line.find("Train Loss") != std::string::npos ||
        line.find("Val Loss")   != std::string::npos ||
        line.find("Epoch ")     != std::string::npos ||
        line.find("loss=")      != std::string::npos ||
        line.find("train_nnue") != std::string::npos ||
        line.find("Phase loss") != std::string::npos ||
        line.find("Acc:")       != std::string::npos ||
        (line.find("LR:") != std::string::npos && line.find("Epoch") != std::string::npos))
        return LC_TRAINING;

    // Success / completion
    if (line.find("=== Pipeline") != std::string::npos ||
        line.find("Training complete") != std::string::npos ||
        line.find("Weights saved")     != std::string::npos ||
        line.find("SWA: Applied")      != std::string::npos ||
        line.find("Done. ")            != std::string::npos ||
        line.find("complete!")         != std::string::npos)
        return LC_SUCCESS;

    // Pipeline progress milestones
    if (line.find("=== ") != std::string::npos ||
        line.find("--- ") != std::string::npos ||
        (line.find("Gen ") != std::string::npos && line.find("done") != std::string::npos) ||
        line.find("Generation ") != std::string::npos ||
        line.find("ELO:")  != std::string::npos ||
        line.find("Elo:")  != std::string::npos ||
        line.find("[LOG]") != std::string::npos)
        return LC_PROGRESS;

    // Meta / timing / commands
    if (line.find("[CMD]")   != std::string::npos ||
        line.find("elapsed") != std::string::npos ||
        line.find("Elapsed") != std::string::npos ||
        line.find("Loading") != std::string::npos ||
        line.find("Loaded")  != std::string::npos ||
        line.find("Using device") != std::string::npos ||
        line.find("CPU threads")  != std::string::npos ||
        line.find("pos/s")        != std::string::npos)
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
    // Assign to job object so child is killed if TrainingRunner crashes
    if (g_hJob) AssignProcessToJobObject(g_hJob, pi.hProcess);
    {
        std::lock_guard<std::mutex> lk(g_activePiMtx);
        g_activePi = pi;
    }

    std::string buf; char ch[4096] = {}; DWORD br = 0;
    while (ReadFile(hR, ch, (DWORD)(sizeof(ch)-1), &br, nullptr) && br > 0) {
        ch[br] = '\0'; buf += ch;
        size_t p;
        while ((p = buf.find('\n')) != std::string::npos) {
            std::string ln = buf.substr(0, p);
            buf = buf.substr(p+1);
            // Strip trailing \r (CRLF line ending)
            if (!ln.empty() && ln.back()=='\r') ln.pop_back();
            // Handle interior \r (progress overwrite): keep text after last \r,
            // but preserve a leading \r so pushLog knows to overwrite the last line.
            {
                auto cr = ln.rfind('\r');
                if (cr != std::string::npos) {
                    ln = "\r" + ln.substr(cr + 1);  // re-add leading \r as overwrite signal
                }
            }
            if (!ln.empty() && ln != "\r") cb(ln);
        }
        if (stop.load()) {
            GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pi.dwProcessId);
            WaitForSingleObject(pi.hProcess, 3000);
            TerminateProcess(pi.hProcess, 1);
            break;
        }
        if (g_skipFlag.load()) {
            GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pi.dwProcessId);
            WaitForSingleObject(pi.hProcess, 3000);
            TerminateProcess(pi.hProcess, 1);
            break;
        }
    }
    if (!buf.empty()) {
        if (buf.back()=='\r') buf.pop_back();
        auto cr = buf.rfind('\r');
        if (cr != std::string::npos) buf = "\r" + buf.substr(cr + 1);
        if (!buf.empty() && buf != "\r") cb(buf);
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ex=1; GetExitCodeProcess(pi.hProcess, &ex);
    bool userStopped = stop.load();
    bool userSkipped = g_skipFlag.exchange(false);  // consume the skip flag
    if (ex != 0) {
        if (userSkipped)
            cb("[SKIP] Phase skipped by user — continuing pipeline");
        else if (userStopped)
            cb("[STOP] Pipeline stopped by user (exit code " + std::to_string(ex) + ")");
        else
            cb("[ERR] Process exited with code " + std::to_string(ex));
    }
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(hR);

    // Clear active PID
    g_activePid.store(0);
    {
        std::lock_guard<std::mutex> lk(g_activePiMtx);
        g_activePi = {};
    }

    // Skip = success (pipeline continues); stop/crash = failure
    if (userSkipped) return true;
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
    if (tl < 0) tl = findNum("loss=");    // --train-duck: "Epoch N/M loss=0.081530"
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
    if (acc < 0) acc = findNum("acc=");
    pt.hasAcc = acc >= 0;
    pt.accuracy = pt.hasAcc ? acc : 0.0;
    // Phase losses: "Phase loss -> Opening: X  Middlegame: Y  Endgame: Z"
    // or duck chess compact: "op=X mg=Y eg=Z"
    double op = findNum("Opening:");
    if (op < 0) op = findNum("op=");
    double mg = findNum("Middlegame:");
    if (mg < 0) mg = findNum("mg=");
    double eg = findNum("Endgame:");
    if (eg < 0) eg = findNum("eg=");
    pt.hasPhase = (op >= 0 && mg >= 0 && eg >= 0);
    if (pt.hasPhase) {
        pt.openingLoss    = op;
        pt.middlegameLoss = mg;
        pt.endgameLoss    = eg;
    }
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
    std::string variantLabel = (cfg.variant == ChessVariant::DuckChess) ? "Duck Chess" : (cfg.variant == ChessVariant::Automate) ? "Automate" : "Standard";
    std::wstring cmd = W(
        "\"" + (fs::path(d)/cfg.exeName).string() + "\""
        " --generate --games " + std::to_string(cfg.gamesPerGen) +
        " --depth " + std::to_string(cfg.depth) +
        " --workers " + std::to_string(cfg.workers) +
        " --output \"" + outFile.string() + "\"" +
        weightsArg + cfg.variantFlag() +
        // Adjudication — for duck chess use higher draw threshold (evals near 0 with random weights)
        " --resign-cp " + std::to_string(cfg.resignCp) +
        " --resign-count " + std::to_string(cfg.resignCount) +
        " --draw-cp " + std::to_string(cfg.variant == ChessVariant::DuckChess ? std::max(cfg.drawCp, 50) : cfg.drawCp) +
        " --draw-count " + std::to_string(cfg.drawCount) +
        " --draw-min-ply " + std::to_string(cfg.drawMinPly) +
        " --draw-adj-moves " + std::to_string(cfg.drawAdjMoves) +
        " --draw-adj-threshold " + std::to_string(cfg.drawAdjThreshold) +
        " --draw-adj-min-move " + std::to_string(cfg.drawAdjMinMove) +
        " --maxplies " + std::to_string(cfg.maxPlies) +
        " --contempt " + std::to_string(cfg.contemptCp) +
        // Opening diversity
        " --opening-plies " + std::to_string(cfg.openingPlies) +
        " --opening-temp " + dbl2s(cfg.openingTemp, 2) +
        " --softmax-plies " + std::to_string(cfg.softmaxPlies) +
        " --softmax-temp " + dbl2s(cfg.softmaxTemp, 2) +
        (cfg.rootNoiseEps > 0.0 ? " --root-noise " + dbl2s(cfg.rootNoiseEps, 3) : "") +
        // Recording filters
        " --record-min-ply " + std::to_string(cfg.recordMinPly) +
        " --record-max-eval " + std::to_string(cfg.recordMaxEval) +
        // NPS sampling: one sample per epoch slot
        " --nps-samples " + std::to_string(cfg.epochsPerGen) +
        // Mixed depth
        (cfg.mixedDepthRatio > 0.0 ?
            " --mixed-depth-ratio " + dbl2s(cfg.mixedDepthRatio, 3) +
            " --mixed-depth-low " + std::to_string(cfg.mixedDepthLow) : "") +
        (cfg.depthShuffle ? " --depth-shuffle --depth-shuffle-bias " + dbl2s(cfg.depthShuffleBias, 2) : "") +
        // FRC mix
        (cfg.frcMix > 0.0 ? " --frc-mix " + dbl2s(cfg.frcMix, 3) : "") +
        // Openings file — only for standard chess.
        // Duck chess openings.txt contains standard chess FENs with no duck placed,
        // which are irrelevant to duck chess strategy and reduce position diversity.
        // Duck chess uses random opening plies (openingTemp/openingPlies) instead.
        [&]() -> std::string {
            if (cfg.variant != ChessVariant::Standard) return "";
            fs::path openings = fs::path(d) / cfg.dataDir / "openings.txt";
            if (!fs::exists(openings)) openings = fs::path(d) / "assets" / "openings.txt";
            if (fs::exists(openings)) return " --openings \"" + openings.string() + "\"";
            return "";
        }()
    );
    // Log the exact command so we can debug issues
    g_st.pushLog("[CMD] " + N(cmd));
    g_log.write("CMD", "selfplay", gen, N(cmd));
    g_st.setStatus("Gen "+std::to_string(gen)+": "+variantLabel+" self-play ("+std::to_string(cfg.gamesPerGen)+" games)");
    g_st.setPhase("selfplay");
    g_st.phaseStart = std::chrono::steady_clock::now();
    g_log.event("selfplay", gen, "PHASE_START games=" + std::to_string(cfg.gamesPerGen)
                + " depth=" + std::to_string(cfg.depth)
                + " workers=" + std::to_string(cfg.workers));
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curEpoch=0; g_st.totalEpochs=cfg.gamesPerGen; g_st.curNps=0.0; g_st.selfPlayEtaSec=0; }
    int lastLoggedPct = -1;  // track last logged progress percentage
    bool spOk = RunProc(cmd, d, [&](const std::string& ln){
        // NPS_SAMPLE lines are internal data for the graph — don't show in output window
        if (ln.find("NPS_SAMPLE") != std::string::npos) {
            // Parse and store, but skip pushLog
            auto stepPos = ln.find("step=");
            auto npsPos3 = ln.find("nps=");
            if (stepPos != std::string::npos && npsPos3 != std::string::npos) {
                try {
                    int step = std::stoi(ln.substr(stepPos + 5));
                    double npsVal = std::stod(ln.substr(npsPos3 + 4));
                    if (npsVal > 0.0 && step > 0) {
                        TrainPoint npt; npt.gen = gen; npt.step = step;
                        npt.nps = npsVal; npt.hasNps = true; npt.hasLoss = false;
                        g_st.pushPt(npt);
                        {
                            std::lock_guard<std::mutex> lk(g_st.mtx);
                            g_st.curNps = npsVal;
                            // Remove the step=0 live placeholder now that a real sample exists
                            auto& pts = g_st.pts;
                            pts.erase(std::remove_if(pts.begin(), pts.end(),
                                [&](const TrainPoint& p) {
                                    return p.gen == gen && p.step == 0
                                        && p.hasNps && !p.hasLoss;
                                }), pts.end());
                        }
                        if (g_hGraph) InvalidateRect(g_hGraph, nullptr, FALSE);
                    }
                } catch (...) {}
            }
            return;  // don't push to output window
        }
        g_st.pushLog(ln);
        // Always log errors, tracebacks, important lines
        if (ln.find("[ERR]")     != std::string::npos || ln.find("[STOP]")    != std::string::npos ||
            ln.find("[SKIP]")    != std::string::npos || ln.find("error")     != std::string::npos ||
            ln.find("Error")     != std::string::npos || ln.find("failed")    != std::string::npos ||
            ln.find("Traceback") != std::string::npos || ln.find("  File \"") != std::string::npos ||
            ln.find("Exception") != std::string::npos || ln.find("WARNING")   != std::string::npos ||
            ln.find("Starting")  != std::string::npos)
            g_log.write("INFO", "selfplay", gen, ln);

        // Log the final "Done." line — contains positions written
        if (ln.find("Done.") != std::string::npos)
            g_log.write("STATS", "selfplay", gen, ln);

        // Parse "[SelfPlay] done/total (pct%) pos=N W/D/B=w/d/b ... X games/s ... Y nps"
        auto p = ln.find("[SelfPlay] ");
        if (p != std::string::npos) {
            // Track when the engine last printed a [SelfPlay] line
            { std::lock_guard<std::mutex> lk(g_st.mtx);
              g_st.lastSelfPlayPrint = std::chrono::steady_clock::now(); }

            // Update games counter
            try { size_t n; int g2=std::stoi(ln.substr(p+11), &n);
                std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curEpoch=g2; }
            catch(...) {}

            // Parse self-play ETA: "ETA HH:MM:SS"
            {
                auto ep = ln.find("ETA ");
                if (ep != std::string::npos) {
                    int hh = 0, mm = 0, ss = 0;
                    if (std::sscanf(ln.c_str() + ep + 4, "%d:%d:%d", &hh, &mm, &ss) >= 2) {
                        int totalSec = hh * 3600 + mm * 60 + ss;
                        auto now = std::chrono::steady_clock::now();
                        std::lock_guard<std::mutex> lk(g_st.mtx);
                        g_st.selfPlayEtaSec   = totalSec;
                        g_st.selfPlayEtaStamp = now;
                    }
                }
            }

            // Parse NPS from every progress line for live graph (not just milestones)
            {
                auto npsPos2 = ln.find(" nps");
                if (npsPos2 != std::string::npos && npsPos2 > 0) {
                    try {
                        size_t start2 = ln.rfind(' ', npsPos2 - 1);
                        if (start2 == std::string::npos) start2 = 0; else start2++;
                        std::string numStr = ln.substr(start2, npsPos2 - start2);
                        double npsVal = 0.0;
                        if (!numStr.empty() && (numStr.back()=='K'||numStr.back()=='k'))
                            npsVal = std::stod(numStr.substr(0,numStr.size()-1)) * 1000.0;
                        else if (!numStr.empty() && (numStr.back()=='M'||numStr.back()=='m'))
                            npsVal = std::stod(numStr.substr(0,numStr.size()-1)) * 1000000.0;
                        else
                            npsVal = std::stod(numStr);
                        if (npsVal > 0.0 && npsVal < 50000000.0) {
                            std::lock_guard<std::mutex> lk(g_st.mtx);
                            g_st.curNps = npsVal;
                        }
                    } catch(...) {}
                }
            }

            // Log progress snapshots at 25%, 50%, 75%, 100%
            auto pctPos = ln.find("(");
            auto pctEnd = ln.find("%)");
            if (pctPos != std::string::npos && pctEnd != std::string::npos) {
                try {
                    int pct = std::stoi(ln.substr(pctPos+1, pctEnd-pctPos-1));
                    int milestone = (pct / 25) * 25;
                    if (milestone > lastLoggedPct && milestone > 0) {
                        lastLoggedPct = milestone;
                        // Extract key stats from the line for the log
                        auto posPos = ln.find("pos=");
                        auto wdbPos = ln.find("W/D/B=");
                        auto npsPos = ln.find("nps");
                        auto gpsPos = ln.find("games/s");
                        std::string stats;
                        if (posPos != std::string::npos) {
                            size_t end = ln.find(' ', posPos); if (end==std::string::npos) end=ln.size();
                            stats += ln.substr(posPos, end-posPos) + " ";
                        }
                        if (wdbPos != std::string::npos) {
                            size_t end = ln.find(' ', wdbPos); if (end==std::string::npos) end=ln.size();
                            stats += ln.substr(wdbPos, end-wdbPos) + " ";
                        }
                        if (gpsPos != std::string::npos) {
                            size_t start = ln.rfind(' ', gpsPos-2)+1;
                            size_t end = ln.find(' ', gpsPos+5); if (end==std::string::npos) end=ln.size();
                            stats += ln.substr(start, end-start) + " ";
                        }
                        if (npsPos != std::string::npos) {
                            size_t start = ln.rfind(' ', npsPos-2)+1;
                            size_t end = ln.find(' ', npsPos+3); if (end==std::string::npos) end=ln.size();
                            stats += ln.substr(start, end-start);
                        }
                        g_log.write("STATS", "selfplay", gen,
                            "PROGRESS pct=" + std::to_string(pct) + "% " + stats);
                    }
                } catch(...) {}
            }
        }
        // Legacy fallback
        if (p == std::string::npos) {
            auto q = ln.find("Games:");
            if (q != std::string::npos) {
                try { int g2=std::stoi(ln.substr(q+6));
                    std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curEpoch=g2; }
                catch(...) {}
            }
        }
    }, g_st.stopFlag);
    bool wasSkipped = !g_skipFlag.load() && spOk && g_activePid.load() == 0;
    // Log selfplay output file size
    {
        fs::path outFile = fs::path(d)/cfg.dataDir/(cfg.selfplayPrefix()+std::to_string(gen)+".bin");
        std::string sizeStr = "unknown";
        if (fs::exists(outFile)) {
            auto sz = fs::file_size(outFile);
            if (sz >= 1024*1024) sizeStr = dbl2s(sz/1048576.0, 2) + " MB";
            else sizeStr = std::to_string(sz/1024) + " KB";
        }
        g_log.event("selfplay", gen, std::string("PHASE_END success=") + (spOk ? "true" : "false")
                    + (g_skipFlag.load() ? " reason=skipped" : "")
                    + " output_size=" + sizeStr);
    }
    return spOk;
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

    // Pre-flight check: base training data must exist
    if (!fs::exists(baseData)) {
        g_st.pushLog("[ERR] Training ABORTED: base data file not found: " + baseData.string());
        g_st.pushLog("[ERR] For Duck Chess, you need duck_training_data.bin in the assets folder.");
        g_log.error("training", gen, "ABORTED: base data missing: " + baseData.string());
        return false;
    }
    std::string args =
        " --data \""        + baseData.string() + "\""
        " --extra-data \""  + selfplayData.string() + "\" " + dbl2s(cfg.splRatio) +
        // Replay window: add previous generations' selfplay data with exponential decay
        [&]() -> std::string {
            std::string replayArgs;
            if (cfg.replayWindow > 0 && cfg.replayDecay > 0.0) {
                double weight = cfg.splRatio * cfg.replayDecay;
                for (int back = 1; back <= cfg.replayWindow && weight > 0.001; ++back) {
                    int replayGen = gen - back;
                    if (replayGen < cfg.startGen) break;  // don't go before the run started
                    fs::path replayFile = assetsDir / (cfg.selfplayPrefix() + std::to_string(replayGen) + ".bin");
                    if (fs::exists(replayFile)) {
                        replayArgs += " --extra-data \"" + replayFile.string() + "\" " + dbl2s(weight, 4);
                    }
                    weight *= cfg.replayDecay;
                }
            }
            return replayArgs;
        }() +
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
        " --enhanced --plot";
    if (cfg.labelSmooth > 0.0)
        args += " --label-smoothing " + dbl2s(cfg.labelSmooth);
    args += " --wdl-alpha " + dbl2s(cfg.wdlAlpha, 4);
    args += " --wdl-draw-elo " + dbl2s(cfg.wdlDrawElo, 2);
    // Cosine LR: pass --no-cosine-restarts only when disabled
    if (!cfg.cosineLr) args += " --no-cosine-restarts";
    if (cfg.cosineLr && cfg.cosineT0 > 0) args += " --cosine-t0 " + std::to_string(cfg.cosineT0);
    // SWA: only pass when enabled
    if (cfg.swa) args += " --swa --swa-start " + std::to_string(cfg.swaStart);
    // Include draws dataset as extra-data with drawPct weight (mirrors TR_Pipeline behaviour)
    {
        // Use variant-specific draws file if it exists, fall back to shared one
        std::string drawsPrefix = (cfg.variant == ChessVariant::DuckChess) ? "duck_" :
                                  (cfg.variant == ChessVariant::Automate)  ? "automate_" : "";
        fs::path drawsData = assetsDir / (drawsPrefix + "training_data_draws.bin");
        if (!fs::exists(drawsData))
            drawsData = assetsDir / "training_data_draws.bin"; // shared fallback
        if (fs::exists(drawsData) && cfg.drawPct > 0.0) {
            double drawRatio = std::min(cfg.drawPct / 100.0, 0.95);
            args += " --extra-data \"" + drawsData.string() + "\" " + dbl2s(drawRatio, 4);
        }
    }
    if (fs::exists(prevWeights))
        args += " --load-weights \"" + prevWeights.string() + "\"";
    args += " --output \"" + outputWeights.string() + "\"";

    // Duck Chess uses the C++ --train-duck mode (DuckNNUE, 832 features)
    // Standard/Automate use train_nnue.py (standard NNUE, 768 features)
    std::wstring cmd;
    if (cfg.variant == ChessVariant::DuckChess) {
        // Build --train-duck command using ChessEngine.exe directly
        std::string duckArgs =
            " --train-duck"
            " --data \""   + baseData.string() + "\""
            " --extra-data \"" + selfplayData.string() + "\""
            " --epochs "   + std::to_string(cfg.epochsPerGen) +
            " --batch-size " + std::to_string(cfg.batchSize) +
            " --lr "       + dbl2s(cfg.lr, 8) +
            " --early-stop " + std::to_string(cfg.earlyStop) +
            " --weight-decay " + dbl2s(cfg.weightDecay, 8) +
            " --grad-accum " + std::to_string(cfg.gradAccum) +
            " --warmup-steps " + std::to_string(cfg.warmupSteps) +
            " --draw-weight " + dbl2s(cfg.drawWeight, 4) +
            " --mate-boost " + dbl2s(cfg.mateBoost, 4) +
            " --max-positions " + std::to_string(cfg.maxPositions);
        if (cfg.labelSmooth > 0.0)
            duckArgs += " --label-smoothing " + dbl2s(cfg.labelSmooth, 4);
        if (cfg.cosineLr) {
            duckArgs += " --cosine-lr";
            if (cfg.cosineT0 > 0) duckArgs += " --cosine-t0 " + std::to_string(cfg.cosineT0);
        }
        if (cfg.swa)
            duckArgs += " --swa --swa-start " + std::to_string(cfg.swaStart);
        // Replay window for duck chess
        if (cfg.replayWindow > 0 && cfg.replayDecay > 0.0) {
            double weight = cfg.splRatio * cfg.replayDecay;
            for (int back = 1; back <= cfg.replayWindow && weight > 0.001; ++back) {
                int replayGen = gen - back;
                if (replayGen < cfg.startGen) break;
                fs::path replayFile = assetsDir / (cfg.selfplayPrefix() + std::to_string(replayGen) + ".bin");
                if (fs::exists(replayFile))
                    duckArgs += " --extra-data \"" + replayFile.string() + "\" " + dbl2s(weight, 4);
                weight *= cfg.replayDecay;
            }
        }
        if (fs::exists(prevWeights))
            duckArgs += " --load-weights \"" + prevWeights.string() + "\"";
        duckArgs += " --output \"" + outputWeights.string() + "\"";
        cmd = W("\"" + (fs::path(d)/cfg.exeName).string() + "\"" + duckArgs);
    } else {
        cmd = W("py -3.10 -u \"" + (fs::path(d)/cfg.pyScript).string() + "\"" + args);
    }
    std::string variantLabel = (cfg.variant == ChessVariant::DuckChess) ? "Duck Chess" : (cfg.variant == ChessVariant::Automate) ? "Automate" : "Standard";
    g_st.setStatus("Gen "+std::to_string(gen)+": "+variantLabel+" training ("+std::to_string(cfg.epochsPerGen)+" epochs)");
    g_st.setPhase("training");
    g_st.phaseStart = std::chrono::steady_clock::now();
    g_log.event("training", gen, "PHASE_START epochs=" + std::to_string(cfg.epochsPerGen)
                + " lr=" + dbl2s(cfg.lr, 8) + " batch=" + std::to_string(cfg.batchSize));
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

        // Parse batch progress from --train-duck: "\r  batch N/TOTAL  loss=X  Y b/s  ETA Zm Ws"
        // Update curEpoch with batch progress so the banner shows real-time progress.
        // Also parse epoch_time= and eta= from epoch summary lines.
        {
            // Batch line: "batch N/TOTAL" — update batch counter for banner display
            auto bpos = ln.find("batch ");
            if (bpos != std::string::npos) {
                try {
                    size_t numStart = bpos + 6;
                    size_t slashPos = ln.find('/', numStart);
                    if (slashPos != std::string::npos) {
                        int bn = std::stoi(ln.substr(numStart, slashPos - numStart));
                        size_t totalEnd = slashPos + 1;
                        int tb = std::stoi(ln.substr(totalEnd));
                        if (bn > 0 && tb > 0) {
                            std::lock_guard<std::mutex> lk(g_st.mtx);
                            g_st.curBatch     = bn;
                            g_st.totalBatches = tb;
                        }
                    }
                } catch (...) {}
            }
            // Epoch line: "Epoch N/M" — update curEpoch immediately.
            // Reset curBatch only when triggered by a batch line (epoch changed mid-epoch),
            // NOT when triggered by the epoch summary line (which fires at epoch end).
            auto epos = ln.find("Epoch ");
            if (epos != std::string::npos) {
                try {
                    size_t numStart = epos + 6;
                    size_t slashPos = ln.find('/', numStart);
                    if (slashPos != std::string::npos) {
                        int en = std::stoi(ln.substr(numStart, slashPos - numStart));
                        if (en > 0) {
                            std::lock_guard<std::mutex> lk(g_st.mtx);
                            // Reset batch counter only when epoch changes AND this is a
                            // batch-level line (contains "batch "), not an epoch summary.
                            // Epoch summary fires at epoch END — curBatch should show 555/555,
                            // not reset to 0 until the first batch of the next epoch arrives.
                            bool isBatchLine = (bpos != std::string::npos);
                            if (en != g_st.curEpoch && isBatchLine) {
                                g_st.curBatch = 0;
                            }
                            g_st.curEpoch = en;
                        }
                    }
                } catch (...) {}
            }
            // Epoch summary line: "epoch_time=Xs eta=XmYs"
            auto etaPos = ln.find("eta=");
            if (etaPos != std::string::npos && ln.find("Epoch ") != std::string::npos) {
                try {
                    std::string etaStr = ln.substr(etaPos + 4);
                    // Parse "XmYs" format
                    int mins = 0, secs = 0;
                    size_t mpos = etaStr.find('m');
                    if (mpos != std::string::npos) {
                        mins = std::stoi(etaStr.substr(0, mpos));
                        size_t spos = etaStr.find('s', mpos);
                        if (spos != std::string::npos)
                            secs = std::stoi(etaStr.substr(mpos + 1, spos - mpos - 1));
                    }
                    int totalSec = mins * 60 + secs;
                    if (totalSec > 0) {
                        auto now = std::chrono::steady_clock::now();
                        std::lock_guard<std::mutex> lk(g_st.mtx);
                        g_st.epochEtaSec   = totalSec;
                        g_st.epochEtaStamp = now;
                    }
                } catch (...) {}
            }
        }

        // Log errors, tracebacks, and exceptions to file
        if (cleaned.find("Traceback") != std::string::npos ||
            cleaned.find("Error")     != std::string::npos ||
            cleaned.find("error")     != std::string::npos ||
            cleaned.find("Exception") != std::string::npos ||
            cleaned.find("  File \"") != std::string::npos ||
            cleaned.find("[ERR]")     != std::string::npos ||
            cleaned.find("[STOP]")    != std::string::npos ||
            cleaned.find("[SKIP]")    != std::string::npos ||
            cleaned.find("WARNING")   != std::string::npos)
            g_log.write("ERROR", "training", gen, cleaned);

        TrainPoint pt; pt.gen = gen;
        // Try cleaned first, fall back to raw ln — ensures val_loss is found
        // regardless of any stripping that may have occurred
        bool parsed = ParseLoss(cleaned, pt) && pt.train < 10.0 && pt.train >= 0.0;
        if (parsed && !pt.hasVal) {
            // Try raw line in case val_loss was stripped from cleaned
            TrainPoint pt2; pt2.gen = gen;
            if (ParseLoss(ln, pt2) && pt2.hasVal) {
                pt.hasVal   = true;
                pt.val      = pt2.val;
                pt.hasAcc   = pt2.hasAcc;
                pt.accuracy = pt2.accuracy;
                pt.hasPhase = pt2.hasPhase;
                pt.openingLoss    = pt2.openingLoss;
                pt.middlegameLoss = pt2.middlegameLoss;
                pt.endgameLoss    = pt2.endgameLoss;
            }
            // Last resort: search for val_loss= directly in both strings
            if (!pt.hasVal) {
                auto tryFind = [&](const std::string& s) {
                    auto pos = s.find("val_loss=");
                    if (pos == std::string::npos) return;
                    pos += 9;  // skip "val_loss="
                    try {
                        size_t n;
                        double v = std::stod(s.substr(pos), &n);
                        if (v >= 0.0 && v < 10.0) { pt.hasVal = true; pt.val = v; }
                    } catch (...) {}
                };
                tryFind(ln);
                tryFind(cleaned);
            }
        }
        if (parsed) {
            g_st.pushPt(pt);
            g_log.metric(gen, pt.step, pt.train, pt.val, pt.hasVal, pt.lr, pt.hasLR);
            if (pt.hasPhase) {
                g_log.write("METRIC", "training", gen,
                    "PHASE_LOSS opening=" + dbl2s(pt.openingLoss, 6) +
                    " middlegame=" + dbl2s(pt.middlegameLoss, 6) +
                    " endgame=" + dbl2s(pt.endgameLoss, 6));
            }
            { std::lock_guard<std::mutex> lk(g_st.mtx);
              if (pt.step > 0) g_st.curEpoch = pt.step; }
            // Immediately redraw graph when a new epoch point arrives
            if (g_hGraph) InvalidateRect(g_hGraph, nullptr, FALSE);
        }
    }, g_st.stopFlag);
    if (ok) {
        std::error_code ec;
        fs::copy_file(outputWeights, genWeights, fs::copy_options::overwrite_existing, ec);
        if (ec) g_st.pushLog("[WARN] Failed to copy weights to " + genWeights.string() + ": " + ec.message());
    }
    // Log training completion with duration and best loss
    {
        auto phaseElapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - g_st.phaseStart).count();
        int mins = (int)(phaseElapsed / 60), secs = (int)(phaseElapsed % 60);
        std::string durStr = std::to_string(mins) + "m " + std::to_string(secs) + "s";
        std::string wSize = "unknown";
        if (fs::exists(outputWeights)) {
            auto sz = fs::file_size(outputWeights);
            wSize = (sz >= 1024*1024) ? dbl2s(sz/1048576.0,2)+" MB" : std::to_string(sz/1024)+" KB";
        }
        double bestTrain = 0.0, bestVal = 0.0;
        { std::lock_guard<std::mutex> lk(g_st.mtx); bestTrain = g_st.lastTrain; bestVal = g_st.lastVal; }
        g_log.event("training", gen, std::string("PHASE_END success=") + (ok ? "true" : "false")
                    + " duration=" + durStr
                    + " best_train=" + dbl2s(bestTrain, 8)
                    + " best_val=" + dbl2s(bestVal, 8)
                    + " weights_size=" + wSize);
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
    fs::path pgnOut = assetsDir/(cfg.weightsBaseName()+"_validation_gen"+std::to_string(gen)+".pgn");
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
    g_log.event("elo", gen, "PHASE_START");
    RunProc(cmd, d, [&](const std::string& ln){
        g_st.pushLog(ln);
        for (auto key : {"ELO:","Elo:","Elo "}) {
            auto p = ln.find(key);
            if (p != std::string::npos) {
                size_t numStart = p + std::string(key).size();
                while (numStart < ln.size() && (ln[numStart]==' '||ln[numStart]==':'||ln[numStart]=='+')) numStart++;
                try { int e=std::stoi(ln.substr(numStart));
                    std::lock_guard<std::mutex> lk(g_st.mtx); g_st.lastElo=e;
                    g_log.write("METRIC", "elo", gen, "ELO_RESULT elo=" + std::to_string(e)); }
                catch(...) {}
            }
        }
    }, g_st.stopFlag);
    g_log.event("elo", gen, "PHASE_END");
}

static void PipelineThread(Config cfg) {
    g_st.stopFlag.store(false);
    g_pauseFlag.store(false);
    g_skipFlag.store(false);
    int firstGen = cfg.startGen + 1;
    int lastGen  = cfg.startGen + cfg.generations;
    { std::lock_guard<std::mutex> lk(g_st.mtx);
      g_st.running=true; g_st.curGen=0; g_st.totalGens=cfg.generations;
      g_st.pipelineStart = std::chrono::steady_clock::now();
      g_st.phaseStart = g_st.pipelineStart;
      g_st.completedGens = 0;
      // Don't clear pts — keep historical data so graph shows full training history
    }
    std::string variantName = (cfg.variant == ChessVariant::DuckChess) ? "Duck Chess" :
                              (cfg.variant == ChessVariant::Automate)  ? "Automate Chess" :
                                                                          "Standard Chess";

    // Open log file — write to assets/logs/ (flat, not variant-specific)
    g_log.open(cfg.dataDir);
    if (g_log.isOpen())
        g_st.pushLog("[LOG] Writing log to: " + g_log.path);

    // Log config
    {
        std::ostringstream cfg_ss;
        cfg_ss << "variant=" << variantName
               << " generations=" << cfg.generations
               << " startGen=" << cfg.startGen
               << " gamesPerGen=" << cfg.gamesPerGen
               << " depth=" << cfg.depth
               << " workers=" << cfg.workers
               << " epochsPerGen=" << cfg.epochsPerGen
               << " batchSize=" << cfg.batchSize
               << " lr=" << dbl2s(cfg.lr, 8)
               << " weightDecay=" << dbl2s(cfg.weightDecay, 8)
               << " maxPositions=" << cfg.maxPositions
               << " splRatio=" << dbl2s(cfg.splRatio, 4)
               << " drawWeight=" << dbl2s(cfg.drawWeight, 4)
               << " mateBoost=" << dbl2s(cfg.mateBoost, 4)
               << " eloValidate=" << (cfg.eloValidate ? "true" : "false");
        g_log.event("pipeline", 0, "PIPELINE_START first_gen=" + std::to_string(firstGen)
                    + " last_gen=" + std::to_string(lastGen)
                    + " total_gens=" + std::to_string(cfg.generations));
        g_log.write("CONFIG", "pipeline", 0, cfg_ss.str());
    }

    g_st.pushLog("=== Pipeline start: "+variantName+" | "+std::to_string(cfg.generations)+" generations (gen "+std::to_string(firstGen)+" to "+std::to_string(lastGen)+") ===");

    bool crashed = false;
    for (int gen=firstGen; gen<=lastGen && !g_st.stopFlag.load(); gen++) {
        { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curGen=gen-firstGen; }
        g_st.pushLog("--- Generation "+std::to_string(gen)+" ---");
        g_log.event("pipeline", gen, "GEN_START gen=" + std::to_string(gen));

        if (!SelfPlay(cfg, gen)) {
            if (g_st.stopFlag.load())
                g_st.pushLog("[STOP] Self-play stopped by user (gen " + std::to_string(gen) + ")");
            else {
                g_st.pushLog("[ERR] Self-play CRASHED gen " + std::to_string(gen) + " — check output above for details");
                PlayMp3("snd_crash", cfg.dataDir, "fuck-crash.mp3");
                crashed = true;
            }
            g_log.error("selfplay", gen, g_st.stopFlag.load() ? "Self-play stopped by user" : "Self-play CRASHED");
            break;
        }
        PlayMp3("snd_sp", cfg.dataDir, "ding_selfplay_complete.mp3");
        if (g_st.stopFlag.load()) break;
        if (!Training(cfg, gen)) {
            if (g_st.stopFlag.load())
                g_st.pushLog("[STOP] Training stopped by user (gen " + std::to_string(gen) + ")");
            else {
                g_st.pushLog("[ERR] Training CRASHED gen " + std::to_string(gen) + " — check output above for details");
                PlayMp3("snd_crash", cfg.dataDir, "fuck-crash.mp3");
                crashed = true;
            }
            g_log.error("training", gen, g_st.stopFlag.load() ? "Training stopped by user" : "Training CRASHED");
            break;
        }
        if (g_st.stopFlag.load()) break;
        EloVal(cfg, gen);

        { std::lock_guard<std::mutex> lk(g_st.mtx);
          std::ostringstream ss; ss<<std::fixed<<std::setprecision(5);
          ss<<"Gen "<<gen<<" done | train="<<g_st.lastTrain<<" val="<<g_st.lastVal<<" ELO="<<g_st.lastElo;
          g_st.log.push_back(ss.str()); g_st.completedGens++;
        }

        // Log gen summary with weights file size (outside lock — g_log has its own sync)
        {
          fs::path weightsFile = fs::path(exeDir())/cfg.dataDir/(cfg.weightsBaseName()+".bin");
          std::string wSize = "unknown";
          if (fs::exists(weightsFile)) {
              auto sz = fs::file_size(weightsFile);
              wSize = (sz >= 1024*1024) ? dbl2s(sz/1048576.0,2)+" MB" : std::to_string(sz/1024)+" KB";
          }
          g_log.event("pipeline", gen, "GEN_SUMMARY gen=" + std::to_string(gen)
                      + " train_loss=" + dbl2s(g_st.lastTrain, 8)
                      + " val_loss=" + dbl2s(g_st.lastVal, 8)
                      + " elo=" + std::to_string(g_st.lastElo)
                      + " weights_size=" + wSize);
        }
        PlayMp3("snd_gen", cfg.dataDir, "ding_gen_complete.mp3");

        // Persist graph data to CSV after each gen so it survives restarts
        {
            fs::path graphCsv = graphCsvPath(cfg.variant);
            std::vector<TrainPoint> pts_snap;
            { std::lock_guard<std::mutex> lk(g_st.mtx); pts_snap = g_st.pts; }
            std::ofstream out(graphCsv.string(), std::ios::trunc);
            out << "# gen,epoch,train,val,lr,acc,openingLoss,middlegameLoss,endgameLoss\n";
            for (auto& p : pts_snap) {
                out << p.gen << ',' << p.step << ','
                    << std::fixed << std::setprecision(8) << p.train << ',';
                if (p.hasVal) out << p.val; else out << "nan";
                out << ',';
                if (p.hasLR) out << std::fixed << std::setprecision(8) << p.lr; else out << "nan";
                out << ',';
                if (p.hasAcc) out << std::fixed << std::setprecision(4) << p.accuracy; else out << "nan";
                out << ',';
                if (p.hasPhase) out << std::fixed << std::setprecision(8)
                    << p.openingLoss << ',' << p.middlegameLoss << ',' << p.endgameLoss;
                else out << "nan,nan,nan";
                out << '\n';
            }
        }
    }
    int finalGens = 0;
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.running=false; finalGens = g_st.completedGens; }
    bool stopped = g_st.stopFlag.load();
    g_st.setStatus(crashed ? "Crashed." : stopped ? "Stopped by user." : "Pipeline complete!");
    g_st.pushLog("=== Pipeline " + std::string(crashed ? "CRASHED" : stopped ? "STOPPED BY USER" : "complete") + " ===");
    if (!stopped && !crashed) PlayMp3("snd_pipe", cfg.dataDir, "ding_gen_complete.mp3");
    g_log.event("pipeline", 0, std::string("PIPELINE_END reason=") + (crashed ? "crash" : stopped ? "user_stop" : "complete")
                + " completed_gens=" + std::to_string(finalGens));
    g_log.close();
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
    HBITMAP oldBmp = (HBITMAP)SelectObject(memDC, bmp);

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
        SelectObject(memDC, oldBmp);
        DeleteObject(bmp); DeleteDC(memDC);
        EndPaint(hw, &ps); return;
    }

    // Weighted panel sizing: Loss gets 3 shares, Acc and LR get 1 share each
    float lossWeight  = g_showLoss  ? 3.0f : 0.0f;
    float accWeight   = g_showAcc   ? 1.0f : 0.0f;
    float lrWeight    = g_showLR    ? 1.0f : 0.0f;
    float phaseWeight = g_showPhase ? 1.5f : 0.0f;
    float npsWeight   = g_showNPS   ? 1.0f : 0.0f;
    float totalWeight = lossWeight + accWeight + lrWeight + phaseWeight + npsWeight;
    if (totalWeight < 0.01f) { lossWeight = 1.0f; totalWeight = 1.0f; }

    int numPanels = (g_showLoss?1:0) + (g_showAcc?1:0) + (g_showLR?1:0) + (g_showPhase?1:0) + (g_showNPS?1:0);
    if (numPanels == 0) numPanels = 1;

    float ml = 52, mr = 16;
    float panelGap = 4.0f;
    float availH = (float)H2 - panelGap * (numPanels - 1);
    float lossH  = availH * (lossWeight  / totalWeight);
    float accH   = availH * (accWeight   / totalWeight);
    float npsH   = availH * (npsWeight   / totalWeight);
    float lrH   = availH * (lrWeight   / totalWeight);
    float phaseH = availH * (phaseWeight / totalWeight);
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
            if (!p2.hasLoss) continue;
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
            if (!pts[i].hasLoss) continue;
            if (pts[i].train < bestTrain) { bestTrain=pts[i].train; bestTrainIdx=(int)i; }
            if (pts[i].hasVal && pts[i].val < bestVal) { bestVal=pts[i].val; bestValIdx=(int)i; }
        }

        Pen trainPen(Color(255,65,125,245), 1.8f);
        { bool st=false; float px_t=0,py_t=0;
          for (size_t i=0; i<pts.size(); i++) {
            if (!pts[i].hasLoss) continue;  // skip NPS-only points without breaking line
            float cx=xf((int)i), cy=yf(pts[i].train);
            if (st) g.DrawLine(&trainPen,px_t,py_t,cx,cy);
            px_t=cx; py_t=cy; st=true;
        }}

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
                std::wostringstream ss; ss<<std::fixed<<std::setprecision(6)<<val;
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

    // ---- Phase Loss panel ----
    if (g_showPhase) {
        float pt_top = curY, pt_h = phaseH;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        SolidBrush panelBg(Color(255,20,24,28));
        g.FillRectangle(&panelBg, 0.0f, pt_top, (float)W2, pt_h);
        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Phase Loss", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);

        bool hasAnyPhase = false;
        for (auto& p2 : pts) if (p2.hasPhase) { hasAnyPhase = true; break; }

        if (!hasAnyPhase) {
            Font nf(L"Segoe UI", 9.0f); SolidBrush nb(Color(255,80,80,100));
            g.DrawString(L"No phase data", -1, &nf, PointF(ml+gw/2-40, mt2+gh2/2-6), &nb);
        } else {
            double minV=1e9, maxV=-1e9;
            for (auto& p2 : pts) {
                if (!p2.hasPhase) continue;
                minV=(std::min)(minV,(std::min)({p2.openingLoss,p2.middlegameLoss,p2.endgameLoss}));
                maxV=(std::max)(maxV,(std::max)({p2.openingLoss,p2.middlegameLoss,p2.endgameLoss}));
            }
            if (maxV<=minV) maxV=minV+0.1;
            double rng=maxV-minV; minV-=rng*0.05; maxV+=rng*0.05; rng=maxV-minV;
            auto yf=[&](double v)->float{return mt2+(float)((maxV-v)/rng)*gh2;};

            Pen gridPen(Color(40,60,60,80),1.0f); Font gridFnt(L"Consolas",7.0f);
            SolidBrush gridBr(Color(255,80,80,100));
            for (int i=0;i<=4;i++){
                float y2=mt2+gh2*i/4;
                g.DrawLine(&gridPen,ml,y2,ml+gw,y2);
                double val=maxV-rng*i/4;
                std::wostringstream ss; ss<<std::fixed<<std::setprecision(4)<<val;
                g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
            }

            // Opening (cyan), Middlegame (orange), Endgame (purple)
            Pen opPen(Color(255,0,200,200),1.6f);
            Pen mgPen(Color(255,255,160,60),1.6f);
            Pen egPen(Color(255,180,80,220),1.6f);

            auto drawPhaseLine = [&](Pen& pen, auto getter) {
                bool st=false; float px5=0,py5=0;
                for (size_t i=0;i<pts.size();i++){
                    if (!pts[i].hasPhase) continue;
                    float cx=xf((int)i), cy=yf(getter(pts[i]));
                    if (st) g.DrawLine(&pen,px5,py5,cx,cy);
                    px5=cx; py5=cy; st=true;
                }
            };
            drawPhaseLine(opPen, [](const TrainPoint& p){return p.openingLoss;});
            drawPhaseLine(mgPen, [](const TrainPoint& p){return p.middlegameLoss;});
            drawPhaseLine(egPen, [](const TrainPoint& p){return p.endgameLoss;});

            // Legend
            Font lf(L"Segoe UI",7.5f);
            float lx = (float)W2 - mr - 200, ly = pt_top + 3;
            SolidBrush b1(Color(255,0,200,200)); Pen lp1(Color(255,0,200,200),2);
            g.DrawLine(&lp1,lx,ly+5,lx+12,ly+5);
            g.DrawString(L"Opening",-1,&lf,PointF(lx+14,ly-1),&b1);
            SolidBrush b2(Color(255,255,160,60)); Pen lp2(Color(255,255,160,60),2);
            g.DrawLine(&lp2,lx+68,ly+5,lx+80,ly+5);
            g.DrawString(L"Middlegame",-1,&lf,PointF(lx+82,ly-1),&b2);
            SolidBrush b3(Color(255,180,80,220)); Pen lp3(Color(255,180,80,220),2);
            g.DrawLine(&lp3,lx+152,ly+5,lx+164,ly+5);
            g.DrawString(L"Endgame",-1,&lf,PointF(lx+166,ly-1),&b3);
        }
        curY += pt_h + panelGap;
    }

    // ---- NPS panel ----
    if (g_showNPS) {
        float pt_top = curY, pt_h = npsH;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        SolidBrush panelBg(Color(255,18,22,28));
        g.FillRectangle(&panelBg, 0.0f, pt_top, (float)W2, pt_h);
        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"NPS (Self-Play)", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);

        bool hasAnyNps = false;
        for (auto& p2 : pts) if (p2.hasNps) { hasAnyNps = true; break; }

        if (!hasAnyNps) {
            Font nf(L"Segoe UI", 9.0f); SolidBrush nb(Color(255,80,80,100));
            g.DrawString(L"No NPS data", -1, &nf, PointF(ml+gw/2-35, mt2+gh2/2-6), &nb);
        } else {
            double minN=1e9, maxN=-1e9;
            for (auto& p2 : pts) {
                if (!p2.hasNps) continue;
                minN=(std::min)(minN,p2.nps); maxN=(std::max)(maxN,p2.nps);
            }
            if (maxN<=minN) maxN=minN+1.0;
            double rng=maxN-minN; minN=std::max(0.0,minN-rng*0.1); maxN+=rng*0.1; rng=maxN-minN;
            auto yf=[&](double v)->float{return mt2+(float)((maxN-v)/rng)*gh2;};

            Pen gridPen(Color(40,60,60,80),1.0f); Font gridFnt(L"Consolas",7.0f);
            SolidBrush gridBr(Color(255,80,80,100));
            for (int i=0;i<=4;i++){
                float y2=mt2+gh2*i/4;
                g.DrawLine(&gridPen,ml,y2,ml+gw,y2);
                double val=maxN-rng*i/4;
                std::wostringstream ss;
                if (val>=1000000) ss<<std::fixed<<std::setprecision(2)<<val/1000000.0<<L"M";
                else if (val>=1000) ss<<std::fixed<<std::setprecision(1)<<val/1000.0<<L"K";
                else ss<<std::fixed<<std::setprecision(0)<<val;
                g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
            }

            // Build (gen, step) -> x map from training points for aligned NPS positioning.
            // For live NPS (no training points yet), distribute evenly across the gen's x range.
            std::map<std::pair<int,int>, float> stepToX;
            for (size_t i = 0; i < pts.size(); i++) {
                if (pts[i].hasLoss && pts[i].step > 0)
                    stepToX[{pts[i].gen, pts[i].step}] = xf((int)i);
            }
            std::map<int, std::pair<float,float>> genXRange;
            {
                std::map<int, std::pair<int,int>> genIdxRange;
                for (size_t i = 0; i < pts.size(); i++) {
                    int g = pts[i].gen;
                    if (genIdxRange.find(g) == genIdxRange.end()) genIdxRange[g] = {(int)i, (int)i};
                    else genIdxRange[g].second = (int)i;
                }
                for (auto& kv : genIdxRange) genXRange[kv.first] = {xf(kv.second.first), xf(kv.second.second)};
            }
            // Pre-compute per-gen NPS counts and indices to avoid O(n²) in npsX lambda.
            std::map<int, int> genNpsCount;
            std::vector<int>   npsIdxInGen(pts.size(), 0);
            {
                std::map<int, int> genNpsCur;
                for (size_t i = 0; i < pts.size(); i++) {
                    if (!pts[i].hasNps) continue;
                    int g = pts[i].gen;
                    npsIdxInGen[i] = genNpsCur[g]++;
                }
                genNpsCount = genNpsCur;
            }
            auto npsX = [&](size_t i) -> float {
                auto it = stepToX.find({pts[i].gen, pts[i].step});
                if (it != stepToX.end()) return it->second;
                auto rng = genXRange.find(pts[i].gen);
                if (rng != genXRange.end()) {
                    int cnt = genNpsCount[pts[i].gen];
                    int idx = npsIdxInGen[i];
                    if (cnt > 1) {
                        float xStart = rng->second.first, xEnd = rng->second.second;
                        return xStart + (xEnd - xStart) * idx / (cnt - 1);
                    }
                    return (rng->second.first + rng->second.second) * 0.5f;
                }
                return xf((int)i);
            };

            // Continuous line through NPS sample points (one per epoch slot).
            Pen npsPen(Color(255,80,220,180),1.8f);
            SolidBrush dotBr(Color(255,80,220,180));
            bool started=false; float px5=0,py5=0; int lastGen=-1;
            for (size_t i=0;i<pts.size();i++){
                if (!pts[i].hasNps) continue;
                float cx=npsX(i), cy=yf(pts[i].nps);
                if (started) {
                    if (pts[i].gen==lastGen)
                        g.DrawLine(&npsPen,px5,py5,cx,cy);
                    else
                        g.FillEllipse(&dotBr,cx-3.0f,cy-3.0f,6.0f,6.0f);
                } else {
                    g.FillEllipse(&dotBr,cx-3.0f,cy-3.0f,6.0f,6.0f);
                }
                px5=cx; py5=cy; started=true; lastGen=pts[i].gen;
            }
        }
        curY += pt_h + panelGap;
    }

    // ---- Generation separator lines ----
    // Draw a vertical dashed line at each gen boundary, spanning all panels,
    // with a gen label at the bottom.
    {
        Pen genPen(Color(60, 120, 120, 140), 1.0f);
        genPen.SetDashStyle(DashStyleDash);
        Font genFnt(L"Consolas", 7.0f);
        SolidBrush genBr(Color(180, 120, 120, 140));

        int prevGen = pts.empty() ? -1 : pts[0].gen;
        for (size_t i = 1; i < pts.size(); i++) {
            if (pts[i].gen != prevGen) {
                float lx = xf((int)i);
                // Vertical line spanning full graph height
                g.DrawLine(&genPen, lx, 0.0f, lx, (float)H2);
                // Gen label at bottom
                std::wstring genLabel = L"G" + std::to_wstring(pts[i].gen);
                g.DrawString(genLabel.c_str(), -1, &genFnt,
                             PointF(lx + 2, (float)H2 - 14), &genBr);
                prevGen = pts[i].gen;
            }
        }
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
        if (hp.hasLR)  ss << L"\nLR: " << std::fixed << std::setprecision(8) << hp.lr;
        if (hp.hasAcc) ss << L"\nAcc: " << std::fixed << std::setprecision(4) << hp.accuracy;
        if (hp.hasPhase) ss << L"\nOp: " << std::fixed << std::setprecision(5) << hp.openingLoss
                            << L"  Mg: " << hp.middlegameLoss
                            << L"  Eg: " << hp.endgameLoss;
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
    SelectObject(memDC, oldBmp);
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

// Forward declaration
static void ScrollCfgTo(HWND hw, int newPos);

// Forward scroll events from child controls to the config panel
static LRESULT CALLBACK ChildScrollProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp,
                                         UINT_PTR, DWORD_PTR) {
    if (msg == WM_MOUSEWHEEL) {
        // Walk up to find the config pane and forward there
        if (g_hCfgPane) {
            int delta = GET_WHEEL_DELTA_WPARAM(wp);
            ScrollCfgTo(g_hCfgPane, g_cfgScrollY - delta / 3);
            return 0;
        }
    }
    return DefSubclassProc(hw, msg, wp, lp);
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
        SetWindowSubclass(g_hVariant, ChildScrollProc, 3, 0);
        SendMessageW(g_hVariant, CB_ADDSTRING, 0, (LPARAM)L"Standard Chess");
        SendMessageW(g_hVariant, CB_ADDSTRING, 0, (LPARAM)L"Duck Chess");
        SendMessageW(g_hVariant, CB_ADDSTRING, 0, (LPARAM)L"Automate Chess");
        SendMessageW(g_hVariant, CB_SETCURSEL, 0, 0);
        AddTooltip(lbl, L"Select which chess variant to train.\n\nStandard Chess: uses 768-feature NNUE (nnue_weights*.bin, training_data.bin, selfplay_gen*.bin).\n\nDuck Chess: uses 832-feature DuckNNUE (duck_nnue_weights*.bin, duck_training_data.bin, duck_selfplay_gen*.bin).\n\nAutomate Chess: trains the Automate Play NNUE on exotic-army positions (automate_play_weights*.bin, automate_training_data.bin, automate_selfplay_gen*.bin). The standard NNUE is never modified.\n\nAll files are fully separated — switching variants will never overwrite the other variant's data.");
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
        SetWindowSubclass(g_hPreset, ChildScrollProc, 3, 0);
        AddTooltip(lbl, L"WHAT: Selects the active configuration preset that populates all settings below. Built-in presets provide tuned starting points for common training scenarios.\n\nHOW TO USE: Pick a built-in preset to start, then adjust individual settings as needed. Use 'Save As...' to save your custom configuration. Custom presets appear in this list and can be deleted; built-in presets cannot.");
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
        SetWindowSubclass(g_hBtnSave, ChildScrollProc, 3, 0);
        g_hBtnDel = CreateWindowExW(0, L"BUTTON", L"Delete",
                                    WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                                    ex + halfW + 4, y, halfW, 22, pane,
                                    (HMENU)(LONG_PTR)ID_BTN_DEL_PRESET, g_hInst, nullptr);
        SendMessageW(g_hBtnDel, WM_SETFONT, (WPARAM)g_fUI, TRUE);
        SetWindowSubclass(g_hBtnDel, ChildScrollProc, 3, 0);
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
        int chkW = (PW - 24) / 5;
        g_hChkGLoss = mkCheck(pane, ID_CHK_GRAPH_LOSS, L"Loss", lx, y, chkW, 20, true);
        g_hChkGAcc  = mkCheck(pane, ID_CHK_GRAPH_ACC,  L"Accuracy", lx+chkW, y, chkW, 20, true);
        g_hChkGLR    = mkCheck(pane, ID_CHK_GRAPH_LR,    L"LR",     lx+chkW*2, y, chkW, 20, true);
        g_hChkGPhase = mkCheck(pane, ID_CHK_GRAPH_PHASE, L"Phases", lx+chkW*3, y, chkW, 20, true);
        g_hChkGNPS   = mkCheck(pane, ID_CHK_GRAPH_NPS,   L"NPS",   lx+chkW*4, y, chkW, 20, true);
        AddTooltip(g_hChkGLoss,  L"WHAT: Shows or hides the Loss curve panel, which plots training loss and validation loss across epochs. Best-achieved values are highlighted with diamond markers.\n\nWHY: The loss curves are your primary diagnostic tool. A healthy run shows both curves declining together. If training loss drops but validation loss rises, the model is overfitting. If both plateau early, try increasing the learning rate or adding more data.");
        AddTooltip(g_hChkGAcc,   L"WHAT: Shows or hides the Accuracy panel, which plots move prediction accuracy across epochs when reported by the training script.\n\nWHY: Accuracy measures how often the model's top move matches the best move from the training data. It provides a complementary view to loss -- a model can have low loss but poor accuracy if it spreads probability too evenly. Rising accuracy confirms the model is learning meaningful patterns.");
        AddTooltip(g_hChkGLR,    L"WHAT: Shows or hides the Learning Rate schedule panel, which plots the effective learning rate at each epoch or step.\n\nWHY: Visualizing the LR schedule helps verify that cosine annealing, warm restarts, and warmup are working as expected. Unexpected LR behavior (flat when it should decay, or spikes) often explains sudden training instability.");
        AddTooltip(g_hChkGPhase, L"WHAT: Shows or hides the Phase Loss panel, which breaks down training loss by game phase: Opening, Middlegame, and Endgame.\n\nWHY: Phase breakdown reveals where the engine struggles most. High endgame loss suggests the model needs more endgame training data or a higher mate boost. High opening loss may indicate insufficient opening book diversity. Balanced phase losses indicate a well-rounded model.");
        y += dy;
        g_hChkMute = mkCheck(pane, ID_CHK_MUTE_SOUNDS, L"\xD83D\xDD07 Mute Sounds", lx, y, PW-16, 20, false);
        AddTooltip(g_hChkMute, L"WHAT: Mutes the notification sounds that play when self-play and generation complete.\n\nWHY: Handy if you're running training overnight or just prefer silence.");
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
        AddTooltip(lbl, L"WHAT: The total number of self-play \u2192 train cycles to execute. Each generation plays a batch of games using the current model, then trains a new model on the resulting data.\n\nWHY: Each generation produces a slightly stronger model that generates higher-quality training data for the next cycle. This feedback loop is the core of the reinforcement learning process -- more generations compound improvements, but with diminishing returns as the model approaches its architectural ceiling.\n\nWHEN TO ADJUST: For initial experiments, 5-10 generations is enough to see if training is working. For serious training runs, 30-100+ generations are typical. Very high values (200+) are safe -- early stopping or manual intervention can halt the run if progress stalls.\n\nDefault: 10 generations.");
        y += dy;
    }

    // Start Gen  (with "Latest" and "Best" buttons)
    {
        int btnW = 44;                       // width of each button
        int gap  = 3;
        int edW  = ew - btnW*2 - gap*2;     // shrink edit to make room for two buttons
        HWND lbl = mkLabel(pane, L"Start Gen", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_STARTGEN, L"0", ex, y, edW, 20);
        g_edits[ID_EDIT_STARTGEN] = ed;
        int bx = ex + edW + gap;
        HWND btnLatest = CreateWindowExW(0, L"BUTTON", L"Latest",
                     WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                     bx, y, btnW, 20,
                     pane, (HMENU)(LONG_PTR)ID_BTN_LATEST_GEN, g_hInst, nullptr);
        SendMessageW(btnLatest, WM_SETFONT, (WPARAM)g_fUI, TRUE);
        SetWindowSubclass(btnLatest, ChildScrollProc, 3, 0);
        bx += btnW + gap;
        HWND btnBest = CreateWindowExW(0, L"BUTTON", L"Best",
                     WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                     bx, y, btnW, 20,
                     pane, (HMENU)(LONG_PTR)ID_BTN_BEST_GEN, g_hInst, nullptr);
        SendMessageW(btnBest, WM_SETFONT, (WPARAM)g_fUI, TRUE);
        SetWindowSubclass(btnBest, ChildScrollProc, 3, 0);
        AddTooltip(lbl, L"WHAT: The generation number to resume training from. The system will look for existing weights (nnue_weights_genN.bin) at this generation and continue the self-play \u2192 train loop from there.\n\nWHY: Training runs can be interrupted by crashes, power loss, or intentional stops. Resuming from a checkpoint avoids re-doing expensive self-play and training work. Starting from 0 begins a completely fresh run with random weights.\n\nWHEN TO ADJUST: Set to 0 for a brand-new training run. After an interruption, click 'Latest' to auto-detect the highest completed generation and resume seamlessly. Click 'Best' to resume from the generation with the lowest validation loss if recent generations regressed.\n\nDefault: 0 (fresh start).");
        AddTooltip(btnLatest, L"WHAT: Scans the assets folder for nnue_weights_genN.bin files and automatically fills in the highest generation number found.\n\nHOW TO USE: Click after an interrupted run to resume from the last completed generation without manually checking which files exist.");
        AddTooltip(btnBest, L"WHAT: Reads the training log and sets Start Gen to the generation that achieved the lowest validation loss.\n\nHOW TO USE: Click when you suspect recent generations have regressed (validation loss climbing). This lets you roll back to the strongest known checkpoint and try different settings from there.");
        y += dy;
    }

    // Games per Gen
    {
        HWND lbl = mkLabel(pane, L"Games per Gen", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_GAMES, L"5000", ex, y, ew, 20);
        g_edits[ID_EDIT_GAMES] = ed;
        AddTooltip(lbl, L"WHAT: The number of self-play games the engine plays against itself each generation to produce training data. Each game generates dozens to hundreds of labeled positions for the neural network to learn from.\n\nWHY: More games means more diverse positions and more robust gradient estimates during training, reducing overfitting and noise. However, each game costs CPU time proportional to the search depth. There is a sweet spot where additional games yield diminishing returns because the model can only absorb so much new information per generation.\n\nWHEN TO ADJUST: Start with 3000-5000 for fast iteration during development. Scale up to 10000-25000 for serious training runs. If training loss is noisy or the model oscillates between generations, increase games. If each generation takes too long and you want faster feedback, decrease games or use Mixed Depth to speed up generation.\n\nDefault: 5000 games per generation.");
        y += dy;
    }

    // Workers
    {
        HWND lbl = mkLabel(pane, L"Workers", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WORKERS, L"12", ex, y, ew, 20);
        g_edits[ID_EDIT_WORKERS] = ed;
        AddTooltip(lbl, L"WHAT: The number of parallel threads used to generate self-play games simultaneously. Each worker runs an independent game using the current engine weights.\n\nWHY: Self-play is CPU-bound and embarrassingly parallel -- each game is independent. Using more workers linearly reduces generation time up to your CPU's thread count. Beyond that, hyperthreading provides diminishing returns and can even slow down due to cache contention.\n\nWHEN TO ADJUST: Set to your CPU's physical thread count for maximum throughput (check Task Manager \u2192 Performance \u2192 Logical processors). If you want to keep your system responsive during training, set to 50-75%% of your thread count. Setting too high (beyond physical threads) wastes resources. Setting to 1 is useful for debugging but extremely slow for real training.\n\nDefault: 12 threads.");
        y += dy;
    }

    // Depth
    {
        HWND lbl = mkLabel(pane, L"Depth", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DEPTH, L"5", ex, y, ew, 20);
        g_edits[ID_EDIT_DEPTH] = ed;
        AddTooltip(lbl, L"WHAT: The search depth (in plies) the engine uses when selecting moves during self-play. At each position, the engine searches this many half-moves ahead before choosing.\n\nWHY: Deeper searches produce stronger, more realistic games with better-quality position evaluations as training labels. Shallow games (depth 3-4) are fast but noisy -- the engine makes tactical blunders that teach bad habits. Deep games (depth 8+) produce expert-level data but are exponentially slower to generate.\n\nWHEN TO ADJUST: Depth 5-6 is the sweet spot for most NNUE training -- good move quality with reasonable speed. Use depth 7-8 for late-stage refinement when the model is already strong. Depth 3-4 is acceptable for very early training or when combined with Mixed Depth. Going beyond depth 10 is rarely worth the time cost unless you have very fast hardware.\n\nDefault: 5 plies.");
        y += dy;
    }

    // Mixed Depth %
    {
        HWND lbl = mkLabel(pane, L"Mixed Depth %", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MIXDEPTH_PCT, L"0", ex, y, ew, 20);
        g_edits[ID_EDIT_MIXDEPTH_PCT] = ed;
        AddTooltip(lbl, L"WHAT: The percentage of self-play games played at the reduced Low Depth instead of the full Depth. When set to 80, 80%% of games use Low Depth (fast) and 20%% use full Depth (strong).\n\nWHY: Playing all games at full depth is slow. Research shows that mixing a majority of fast, shallow games with a minority of deep games produces training data almost as good as all-deep data, at a fraction of the time cost. The shallow games provide volume and position diversity, while the deep games anchor the quality.\n\nWHEN TO ADJUST: Set to 0 to disable (all games at full depth). 70-80%% is optimal for most runs -- massive speed boost with minimal quality loss. Going above 90%% starts degrading data quality noticeably. If your full depth is already low (4-5), mixed depth provides less benefit.\n\nDefault: 0 (disabled -- all games at full depth).");
        y += dy;
    }

    // Mixed Low Depth
    {
        HWND lbl = mkLabel(pane, L"Low Depth", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MIXDEPTH_LOW, L"4", ex, y, ew, 20);
        g_edits[ID_EDIT_MIXDEPTH_LOW] = ed;
        AddTooltip(lbl, L"WHAT: The search depth used for the fast games when Mixed Depth is enabled. Only applies to the percentage of games specified by Mixed Depth %%.\n\nWHY: This controls the speed/quality trade-off for the fast portion of your data. A low value (3-4) maximizes throughput but produces noisier evaluations. A higher value (5-6) is slower but keeps data quality closer to full depth.\n\nWHEN TO ADJUST: Depth 4 is the recommended sweet spot -- fast enough to provide a 2-3x throughput boost while still producing reasonable move choices. Depth 3 is viable but expect more tactical blunders in the data. Depth 5+ narrows the gap with full depth, reducing the throughput benefit. This setting has no effect when Mixed Depth %% is 0.\n\nDefault: 4 plies.");
        y += dy;
    }

    // Depth Shuffle checkbox
    {
        g_hChkDepthShuffle = CreateWindowExW(0, L"BUTTON", L"Depth Shuffle",
            WS_CHILD|WS_VISIBLE|BS_AUTOCHECKBOX, lx, y, lw+ew, 18, pane,
            (HMENU)(INT_PTR)ID_CHK_DEPTH_SHUFFLE, g_hInst, nullptr);
        SendMessageW(g_hChkDepthShuffle, WM_SETFONT, (WPARAM)g_fUI, TRUE);
        AddTooltip(g_hChkDepthShuffle, L"WHAT: When enabled, games assigned to the Mixed Depth pool sample their search depth from a geometric distribution over [Low Depth, Depth) instead of all playing at Low Depth.\n\nWHY: A richer diversity of search depths in training data helps the neural network see positions evaluated at varying quality levels. Higher depths are weighted more heavily (controlled by Shuffle Bias), so most shuffled games still use strong searches, while a smaller fraction of shallow games adds throughput and diversity.\n\nWHEN TO ADJUST: Enable when your Depth minus Low Depth is at least 2 (e.g. Depth=9, Low Depth=4 gives 5 tiers). Requires Mixed Depth %% > 0 to have any effect. Leave disabled for the classic binary mixed depth behavior.\n\nDefault: off.");
        y += dy;
    }

    // Depth Shuffle Bias
    {
        HWND lbl = mkLabel(pane, L"Shuffle Bias", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DEPTH_SHUFFLE_BIAS, L"2.0", ex, y, ew, 20);
        g_edits[ID_EDIT_DEPTH_SHUFFLE_BIAS] = ed;
        AddTooltip(lbl, L"WHAT: Controls the geometric weighting for Depth Shuffle. Each depth level d gets probability weight bias^(d - LowDepth). Higher bias means deeper searches are much more likely than shallow ones.\n\nWHY: Bias=2.0 means each depth tier is 2x more likely than the one below it. At bias=1.0 the distribution is uniform (equal chance for every depth). At bias=3.0+ almost all games cluster near the top depth with only rare shallow games.\n\nWHEN TO ADJUST: 2.0 is a good default -- it gives a natural exponential ramp favoring quality while still producing meaningful numbers of cheaper games. Increase to 3.0-4.0 if you want even stronger bias toward high-depth data. Decrease toward 1.0 for more even coverage across depths. Has no effect unless Depth Shuffle is enabled.\n\nDefault: 2.0");
        y += dy;
    }

    // ── Section header: Opening Diversity ──────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Opening Diversity ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_fUI,TRUE);
        y += 20;
    }

    // Opening Temp
    {
        HWND lbl = mkLabel(pane, L"Opening Temp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_OPENING_TEMP, L"1.5", ex, y, ew, 20);
        g_edits[ID_EDIT_OPENING_TEMP] = ed;
        AddTooltip(lbl, L"WHAT: Softmax temperature for the first Opening Plies of each game. Higher values = wilder, more random opening moves. 0 = always pick the best move (no randomness).\n\nWHY: Without opening randomization, self-play games between identical engines produce nearly identical games. Temperature injects diversity by making the engine sometimes choose 2nd/3rd best moves, producing a wider variety of middlegame positions for training data.\n\nWHEN TO ADJUST: 1.5 is aggressive \u2014 produces very diverse openings. Decrease to 0.5-1.0 for more sensible openings that still have variety. Increase to 2.0+ if you want maximum opening chaos (useful for very early training). Set to 0 for deterministic openings.\n\nDefault: 1.5");
        y += dy;
    }

    // Opening Plies
    {
        HWND lbl = mkLabel(pane, L"Opening Plies", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_OPENING_PLIES, L"4", ex, y, ew, 20);
        g_edits[ID_EDIT_OPENING_PLIES] = ed;
        AddTooltip(lbl, L"WHAT: Number of half-moves at the start of each game that use softmax-random move selection with Opening Temp. After these plies, the engine transitions to the Softmax Plies phase (lower temperature) or best-move play.\n\nWHY: The first few moves determine the opening structure. Randomizing 4 plies (2 full moves) ensures diverse pawn structures and piece placements without making the openings too bizarre. More plies = more diverse but potentially more unrealistic game starts.\n\nWHEN TO ADJUST: 4 (2 full moves) is a strong default. Increase to 6-8 for maximum diversity. Decrease to 2 if openings are too random and producing nonsensical positions. Set to 0 to disable opening randomization entirely (all moves from best-move search).\n\nDefault: 4 plies.");
        y += dy;
    }

    // Softmax Plies
    {
        HWND lbl = mkLabel(pane, L"Softmax Plies", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SOFTMAX_PLIES, L"8", ex, y, ew, 20);
        g_edits[ID_EDIT_SOFTMAX_PLIES] = ed;
        AddTooltip(lbl, L"WHAT: Number of additional plies after the Opening Plies phase that use softmax move selection with the (lower) Softmax Temp. This creates a gradual transition from random opening play to best-move play.\n\nWHY: An abrupt switch from random to best-move creates an artificial boundary in game quality. The softmax phase provides a smooth transition \u2014 moves are mostly strong but with occasional variety. This extends diversity past the opening into the early middlegame.\n\nWHEN TO ADJUST: 8 plies (4 full moves) provides good diversity into the middlegame. Increase to 12-16 for broader coverage. Decrease to 4 or 0 if you want to minimize randomness outside the opening. The temperature during this phase is controlled by Softmax Temp.\n\nDefault: 8 plies.");
        y += dy;
    }

    // Softmax Temp
    {
        HWND lbl = mkLabel(pane, L"Softmax Temp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SOFTMAX_TEMP, L"0.5", ex, y, ew, 20);
        g_edits[ID_EDIT_SOFTMAX_TEMP] = ed;
        AddTooltip(lbl, L"WHAT: Softmax temperature for the post-opening phase (Softmax Plies). Lower than Opening Temp for more reasonable but still slightly varied move choices.\n\nWHY: After the opening, you want most moves to be strong but with occasional deviations. A temperature of 0.5 means the engine strongly prefers the best move but will sometimes play the 2nd/3rd best, keeping games varied without introducing obvious blunders.\n\nWHEN TO ADJUST: 0.5 is well-balanced. Decrease to 0.2-0.3 for nearly best-move play with very rare deviations. Increase to 0.8-1.0 for more aggressive randomization. Set to 0 to disable (only Opening Plies will be randomized). Only applies during the Softmax Plies phase.\n\nDefault: 0.5");
        y += dy;
    }

    // Root Noise
    {
        HWND lbl = mkLabel(pane, L"Root Noise \u03B5", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_ROOT_NOISE, L"0.0", ex, y, ew, 20);
        g_edits[ID_EDIT_ROOT_NOISE] = ed;
        AddTooltip(lbl, L"WHAT: Probability of replacing the best move with a randomly-weighted alternative at each position after the opening/softmax phases. Similar to Leela Chess Zero\u2019s epsilon-greedy exploration.\n\nWHY: Even after the opening, deterministic best-move play can produce repetitive game patterns. Root noise injects occasional \u201Cmistakes\u201D that keep games varied, forcing the engine to handle suboptimal positions. This produces training data for recovery play and defensive technique.\n\nWHEN TO ADJUST: 0.0 (off) is the default. Try 0.05-0.10 for subtle diversity throughout the game. 0.15-0.25 is aggressive and introduces frequent non-optimal moves. Values above 0.3 significantly degrade game quality. Use this as a complement to (not replacement for) Opening Temp.\n\nDefault: 0.0 (disabled).");
        y += dy;
    }

    // ── Section header: Recording Filters ─────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Recording Filters ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_fUI,TRUE);
        y += 20;
    }

    // Record Min Ply
    {
        HWND lbl = mkLabel(pane, L"Record Min Ply", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_RECORD_MIN_PLY, L"10", ex, y, ew, 20);
        g_edits[ID_EDIT_RECORD_MIN_PLY] = ed;
        AddTooltip(lbl, L"WHAT: Positions before this ply number are not saved to the training data file. This filters out the very early opening moves.\n\nWHY: Early opening positions (first 5-10 plies) are dominated by randomized moves from the Opening Temp phase and don\u2019t represent real engine analysis. Training on these noisy positions teaches the model to value random moves, degrading evaluation quality. Filtering them out keeps the training set clean.\n\nWHEN TO ADJUST: 10 (5 full moves) filters the typical random opening. Increase to 16-20 if your Opening Plies + Softmax Plies extend further and you want to skip the entire random phase. Decrease to 4-6 if you want to train on opening positions (useful when Opening Temp is low). Set to 0 to record everything.\n\nDefault: 10 plies.");
        y += dy;
    }

    // Record Max Eval
    {
        HWND lbl = mkLabel(pane, L"Record Max Eval", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_RECORD_MAX_EVAL, L"2500", ex, y, ew, 20);
        g_edits[ID_EDIT_RECORD_MAX_EVAL] = ed;
        AddTooltip(lbl, L"WHAT: Positions where the absolute evaluation exceeds this centipawn threshold are not saved to the training data. These are already-decided positions.\n\nWHY: When one side has a crushing advantage (e.g. +25 pawns), the remaining moves provide little training value \u2014 the outcome is obvious and the positions are highly unusual. Filtering them reduces noise in the dataset and saves disk space for positions where evaluation accuracy actually matters.\n\nWHEN TO ADJUST: 2500cp (~25 pawns) is very permissive \u2014 only the most extreme positions are filtered. Decrease to 1500-2000 for tighter filtering that removes moderately one-sided positions. Decrease further to 1000 to focus training on competitive positions only. Increase to 5000+ to effectively disable the filter.\n\nDefault: 2500 centipawns.");
        y += dy;
    }

    // ── Section header: Self-Play Adjudication ─────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Self-Play Adjudication ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_fUI,TRUE);
        y += 20;
    }

    // Resign Threshold (cp)
    {
        HWND lbl = mkLabel(pane, L"Resign Cp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_RESIGNCP, L"500", ex, y, ew, 20);
        g_edits[ID_EDIT_RESIGNCP] = ed;
        AddTooltip(lbl, L"WHAT: The centipawn threshold at which the engine concedes a lost game. When the engine's evaluation is worse than this value for 3 consecutive moves, it resigns instead of playing on.\n\nWHY: Without resignation, hopelessly lost games drag on for dozens of extra moves, wasting time generating low-quality training data from positions where the outcome is already decided. Resignation speeds up self-play and keeps training data focused on meaningful positions.\n\nWHEN TO ADJUST: Lower the value (e.g. 300-400) to end lost games sooner and speed up data generation. Raise it (e.g. 700-1000) if you want the engine to practice defending difficult endgames. Set to 0 to disable resignation entirely -- useful for testing but significantly slows training.\n\nDefault: 500 centipawns (roughly a rook disadvantage).");
        y += dy;
    }

    // Contempt (cp)
    {
        HWND lbl = mkLabel(pane, L"Contempt Cp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_CONTEMPT, L"25", ex, y, ew, 20);
        g_edits[ID_EDIT_CONTEMPT] = ed;
        AddTooltip(lbl, L"WHAT: A bias in centipawns that the engine adds to its evaluation to discourage accepting draws. A contempt of 25 means the engine treats a drawn position as if it were 25cp worse, making it prefer to keep playing rather than repeat moves or simplify into a dead-equal endgame.\n\nWHY: Self-play between identical engines naturally produces a high draw rate because both sides evaluate positions the same way and readily agree to repetitions. Contempt forces the engine to take risks and fight for decisive results, generating richer and more varied training data with a better balance of wins, losses, and draws.\n\nWHEN TO ADJUST: Increase (e.g. 40-75) if your draw rate is too high and you want more decisive games. Decrease (e.g. 5-15) if the engine is overextending and producing unnatural positions. Set to 0 for unbiased play -- useful for Elo testing but produces excessive draws in training.\n\nDefault: 25 centipawns.");
        y += dy;
    }

    // Max Plies
    {
        HWND lbl = mkLabel(pane, L"Max Plies", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MAXPLIES, L"250", ex, y, ew, 20);
        g_edits[ID_EDIT_MAXPLIES] = ed;
        AddTooltip(lbl, L"WHAT: The maximum number of half-moves (plies) allowed in a single game before it is automatically adjudicated as a draw. One ply equals one side's move, so 250 plies is roughly 125 full moves.\n\nWHY: Without a move limit, some games -- particularly in closed positions or repetitive endgames -- can spiral well past 300 moves, eventually triggering the 50-move draw rule anyway. These ultra-long games waste computation time generating repetitive, low-value training positions. A hard cap ensures no single game consumes a disproportionate amount of resources.\n\nWHEN TO ADJUST: Lower (e.g. 150-200) to speed up data generation at the cost of cutting off some endgames early. Raise (e.g. 300-400) if you want the engine to learn longer endgame technique and are willing to spend more time per game. Very low values (below 100) will prevent many games from reaching natural conclusions and degrade training quality.\n\nDefault: 250 plies (~125 full moves).");
        y += dy;
    }

    // Draw Adjudication (cp)
    {
        HWND lbl = mkLabel(pane, L"Draw Adj Cp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAWCP, L"8", ex, y, ew, 20);
        g_edits[ID_EDIT_DRAWCP] = ed;
        AddTooltip(lbl, L"WHAT: The centipawn threshold for automatic draw adjudication. If both sides' evaluations remain within this value of 0.00 for several consecutive moves after move 50, the game is declared a draw without playing further.\n\nWHY: Many self-play games reach dead-drawn positions (e.g. opposite-colored bishops, blocked pawns) long before a formal draw by repetition or 50-move rule occurs. Without adjudication, the engine plays dozens of meaningless shuffling moves that waste time and add noise to the training data. Draw adjudication detects these positions early and ends the game cleanly.\n\nWHEN TO ADJUST: Lower (e.g. 2-5) for stricter draw detection -- only truly dead positions are adjudicated, but more games drag on. Raise (e.g. 15-30) to be more aggressive about ending close games, speeding up generation but risking premature draws in positions where one side had a slight edge. Set to 0 to disable draw adjudication entirely -- games will only end by checkmate, resignation, repetition, stalemate, or the Max Plies limit.\n\nDefault: 8 centipawns (roughly equal to a small positional edge).");
        y += dy;
    }

    // Resign Count
    {
        HWND lbl = mkLabel(pane, L"Resign Count", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_RESIGN_COUNT, L"3", ex, y, ew, 20);
        g_edits[ID_EDIT_RESIGN_COUNT] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive moves where the engine\u2019s evaluation must exceed the Resign Cp threshold before the game is resigned. Prevents premature resignation from temporary evaluation spikes.\n\nWHY: A single move with a high eval can be a search artifact \u2014 the next move might refute the threat. Requiring multiple consecutive bad evaluations ensures the position is truly lost before resigning. Lower values speed up games but risk occasional premature resignations.\n\nWHEN TO ADJUST: 3 is the standard. Increase to 4-5 for more cautious resignation (fewer false quits). Decrease to 2 for faster game completion in clearly lost positions. Setting to 1 resigns immediately on any single evaluation above threshold \u2014 not recommended.\n\nDefault: 3 consecutive plies.");
        y += dy;
    }

    // Draw Count
    {
        HWND lbl = mkLabel(pane, L"Draw Count", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_COUNT, L"6", ex, y, ew, 20);
        g_edits[ID_EDIT_DRAW_COUNT] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive moves where both sides\u2019 evaluations must stay within the Draw Adj Cp threshold before the game is adjudicated as a draw.\n\nWHY: Brief periods of equal evaluation can occur in positions with hidden tactics. Requiring 6 consecutive balanced evaluations ensures the position is genuinely drawn, not just temporarily quiet. Lower values catch draws faster but may prematurely end positions where one side has a hidden advantage.\n\nWHEN TO ADJUST: 6 is well-calibrated. Increase to 8-10 for stricter draw detection (fewer premature draws). Decrease to 4 for faster adjudication. Works in conjunction with Draw Min Ply \u2014 both conditions must be met.\n\nDefault: 6 consecutive plies.");
        y += dy;
    }

    // Draw Min Ply
    {
        HWND lbl = mkLabel(pane, L"Draw Min Ply", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_MIN_PLY, L"40", ex, y, ew, 20);
        g_edits[ID_EDIT_DRAW_MIN_PLY] = ed;
        AddTooltip(lbl, L"WHAT: The earliest half-move at which the primary draw adjudication (Draw Adj Cp + Draw Count) can trigger. Before this ply, games continue regardless of evaluation.\n\nWHY: Early in the game, positions are still developing and many lines look temporarily equal. Allowing early draw adjudication would cut off games before meaningful play develops, producing training data dominated by opening positions with no middlegame content.\n\nWHEN TO ADJUST: 40 plies (20 full moves) ensures games reach the middlegame. Increase to 60-80 for longer games with more endgame data. Decrease to 20-30 for faster generation at the cost of shorter average game length.\n\nDefault: 40 plies.");
        y += dy;
    }

    // Draw Adj Moves
    {
        HWND lbl = mkLabel(pane, L"Draw Adj Moves", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_ADJ_MOVES, L"12", ex, y, ew, 20);
        g_edits[ID_EDIT_DRAW_ADJ_MOVES] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive plies of near-zero evaluation (below Draw Adj Thresh) required for the secondary \u201Cdead position\u201D draw adjudication. This is a separate, more aggressive draw detector for completely lifeless positions.\n\nWHY: Some positions are obviously drawn (blocked pawns, insufficient material in practice) but may not trigger the primary draw check because the evaluations are slightly asymmetric. The secondary adjudicator catches these dead positions by looking for prolonged near-zero evaluation windows.\n\nWHEN TO ADJUST: 12 plies (6 full moves) of dead-equal play is a strong signal. Increase to 16-20 for more conservative detection. Decrease to 8-10 for faster detection of dead positions. This only triggers after Draw Adj Min Move.\n\nDefault: 12 plies.");
        y += dy;
    }

    // Draw Adj Threshold
    {
        HWND lbl = mkLabel(pane, L"Draw Adj Thresh", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_ADJ_THRESH, L"4", ex, y, ew, 20);
        g_edits[ID_EDIT_DRAW_ADJ_THRESH] = ed;
        AddTooltip(lbl, L"WHAT: The centipawn threshold for the secondary \u201Cdead position\u201D draw adjudication. Evaluations with |eval| <= this value are considered dead equal.\n\nWHY: This is intentionally tighter than Draw Adj Cp. A 4cp threshold means only truly dead-equal positions (both sides agree the position is completely drawn) trigger the secondary adjudication. This catches fortress-type positions and opposite-colored bishop endgames that might not trigger the primary draw check.\n\nWHEN TO ADJUST: 4cp is very tight \u2014 nearly perfect equality. Increase to 8-12 for a wider dead-position band. Decrease to 2 for only mathematically dead positions. Works in conjunction with Draw Adj Moves and Draw Adj Min Move.\n\nDefault: 4 centipawns.");
        y += dy;
    }

    // Draw Adj Min Move
    {
        HWND lbl = mkLabel(pane, L"Draw Adj Min Move", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_ADJ_MIN_MOVE, L"50", ex, y, ew, 20);
        g_edits[ID_EDIT_DRAW_ADJ_MIN_MOVE] = ed;
        AddTooltip(lbl, L"WHAT: The minimum move number (full moves, not plies) before the secondary dead-position draw adjudication can trigger. This ensures games play a substantial amount before being cut short.\n\nWHY: Even completely equal-looking positions in the early middlegame may have latent imbalances that surface later. This minimum ensures the secondary draw check only fires in the late middlegame or endgame, where dead positions are genuinely drawn.\n\nWHEN TO ADJUST: 50 (move 50) is conservative. Decrease to 30-40 for faster adjudication. Increase to 60-80 if you want very long games before any secondary draw detection. This is independent of Draw Min Ply (which controls the primary draw check).\n\nDefault: move 50.");
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
        AddTooltip(lbl, L"WHAT: The number of complete passes through the training dataset each generation. One epoch means every position in the dataset is seen exactly once by the optimizer.\n\nWHY: Multiple epochs allow the model to refine its weights by seeing each position several times with different mini-batch compositions. Too few epochs and the model underfits -- it hasn't fully absorbed the data. Too many and it overfits -- it memorizes specific positions rather than learning general patterns, causing validation loss to rise.\n\nWHEN TO ADJUST: 10-15 epochs is a good starting range. If validation loss is still dropping when training ends, increase epochs. If validation loss starts rising well before the last epoch, reduce epochs or rely on Early Stop to catch it. Larger datasets need fewer epochs (the model sees enough variety in one pass), while smaller datasets benefit from more epochs.\n\nDefault: 10 epochs per generation.");
        y += dy;
    }

    // Batch Size
    {
        HWND lbl = mkLabel(pane, L"Batch Size", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_BATCHSZ, L"2048", ex, y, ew, 20);
        g_edits[ID_EDIT_BATCHSZ] = ed;
        AddTooltip(lbl, L"WHAT: The number of training positions processed together in a single forward/backward pass before updating the model weights. The gradient is averaged over all positions in the batch.\n\nWHY: Batch size controls the noise/stability trade-off in gradient estimation. Small batches (256-512) produce noisy gradients that can help escape local minima but cause erratic training. Large batches (4096-8192) produce smooth, stable gradients but may converge to sharper (less generalizable) minima and require more VRAM.\n\nWHEN TO ADJUST: 2048 works well for most NNUE training. Increase if you have VRAM to spare and want smoother loss curves. Decrease if you run out of VRAM (CUDA out of memory errors). You can also use Grad Accumulation to simulate a larger effective batch size without increasing VRAM usage. The effective batch size is Batch Size × Grad Accumulation.\n\nDefault: 2048 positions.");
        y += dy;
    }

    // Learning Rate
    {
        HWND lbl = mkLabel(pane, L"Learning Rate", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_LR, L"0.001", ex, y, ew, 20);
        g_edits[ID_EDIT_LR] = ed;
        AddTooltip(lbl, L"WHAT: The step size for each gradient update -- how much the model weights change in response to each batch of training data. This is the single most important hyperparameter in neural network training.\n\nWHY: The learning rate controls the speed and stability of convergence. Too high and the model overshoots good solutions, causing loss to spike or diverge. Too low and training crawls, potentially getting stuck in poor local minima. The right value depends on batch size, model architecture, and data quality.\n\nWHEN TO ADJUST: 0.001 is a strong default for Adam-family optimizers with batch size 2048. If loss oscillates wildly or spikes, halve the LR. If training is very slow or plateaus early, try doubling it. When increasing batch size, consider scaling LR proportionally (e.g. double batch \u2192 double LR). Enable Cosine LR to automatically decay from this starting value.\n\nDefault: 0.001.");
        y += dy;
    }

    // Weight Decay
    {
        HWND lbl = mkLabel(pane, L"Weight Decay", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WD, L"1e-5", ex, y, ew, 20);
        g_edits[ID_EDIT_WD] = ed;
        AddTooltip(lbl, L"WHAT: L2 regularization strength applied to all model weights during training. Each update, weights are multiplied by (1 - weight_decay), gently shrinking them toward zero.\n\nWHY: Without regularization, weights can grow arbitrarily large as the model memorizes training data. Weight decay acts as a soft constraint, keeping weights small and encouraging the model to find simpler, more generalizable solutions. This reduces overfitting -- the gap between training loss and validation loss.\n\nWHEN TO ADJUST: 0.00001 (1e-5) is a conservative default. Increase to 0.0001 or 0.001 if you see significant overfitting (training loss much lower than validation loss). Decrease or set to 0 if the model underfits (both losses plateau high). Larger models and smaller datasets benefit more from stronger decay.\n\nDefault: 0.00001 (1e-5).");
        y += dy;
    }

    // Dropout
    {
        HWND lbl = mkLabel(pane, L"Dropout", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DROPOUT, L"0.1", ex, y, ew, 20);
        g_edits[ID_EDIT_DROPOUT] = ed;
        AddTooltip(lbl, L"WHAT: The probability that each neuron is randomly disabled (output set to zero) during each training forward pass. At inference time, all neurons are active but outputs are scaled accordingly.\n\nWHY: Dropout is a powerful regularization technique that prevents neurons from co-adapting -- relying too heavily on specific other neurons. By randomly disabling neurons, the network is forced to learn redundant representations, making it more robust and less prone to overfitting. It effectively trains an ensemble of sub-networks.\n\nWHEN TO ADJUST: 0.1 (10%%) is a good starting point for NNUE architectures which are relatively small. Increase to 0.2-0.3 if overfitting persists despite weight decay. Set to 0 to disable -- useful when you have abundant training data and overfitting is not a concern, or for final fine-tuning runs. Values above 0.5 are almost never beneficial and will cause underfitting.\n\nDefault: 0.1 (10%% dropout rate).");
        y += dy;
    }

    // Label Smoothing
    {
        HWND lbl = mkLabel(pane, L"Label Smoothing", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_LSMOOTH, L"0.05", ex, y, ew, 20);
        g_edits[ID_EDIT_LSMOOTH] = ed;
        AddTooltip(lbl, L"WHAT: Softens hard win/loss targets by blending them toward 0.5. With a value of 0.05, a target of 1.0 (win) becomes 0.95, and 0.0 (loss) becomes 0.05. Draw targets (0.5) are unaffected.\n\nWHY: Hard 0/1 targets encourage the model to produce extreme, overconfident predictions. In chess, even clearly winning positions have some probability of a draw or loss due to the opponent's counterplay. Label smoothing teaches the model that no outcome is 100%% certain, producing better-calibrated evaluations that correlate more closely with actual win probabilities.\n\nWHEN TO ADJUST: 0.05 is a safe default. Increase to 0.1-0.15 if the model's evaluations are too extreme (e.g. jumping to ±900cp in slightly favorable positions). Set to 0 if you want the sharpest possible evaluation distinctions, though this may reduce playing strength against varied opponents.\n\nDefault: 0.05.");
        y += dy;
    }

    // Grad Accumulation
    {
        HWND lbl = mkLabel(pane, L"Grad Accumulation", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_GRADACCUM, L"4", ex, y, ew, 20);
        g_edits[ID_EDIT_GRADACCUM] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive mini-batches whose gradients are accumulated (summed) before performing a single weight update. The effective batch size becomes Batch Size × Grad Accumulation.\n\nWHY: Some training benefits from large batch sizes (stable gradients, better convergence) but large batches require proportionally more VRAM. Gradient accumulation achieves the same mathematical result as a larger batch by spreading the computation across multiple smaller forward/backward passes, at the cost of slightly slower training due to more sequential steps.\n\nWHEN TO ADJUST: Set to 1 if your GPU can handle the desired batch size directly (most efficient). Increase to 2-8 if you want a larger effective batch but are VRAM-limited. For example, Batch Size 2048 × Grad Accumulation 4 = effective batch of 8192. Values above 8 are rarely needed. Higher values slow down training proportionally since more passes happen per weight update.\n\nDefault: 4 (effective batch size = 2048 × 4 = 8192).");
        y += dy;
    }

    // LR Warmup Steps
    {
        HWND lbl = mkLabel(pane, L"LR Warmup Steps", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WARMUP, L"500", ex, y, ew, 20);
        g_edits[ID_EDIT_WARMUP] = ed;
        AddTooltip(lbl, L"WHAT: The number of training steps over which the learning rate linearly increases from 0 to the target Learning Rate at the start of each generation's training. One step equals one batch (or one accumulated gradient update if Grad Accumulation > 1).\n\nWHY: When training begins, model weights are in a random or partially-trained state. Hitting them with the full learning rate immediately can cause destructive, oversized updates that push the model into a bad region of the loss landscape. Warmup lets the optimizer calibrate its internal momentum estimates (Adam's running averages) with small, safe steps before ramping to full speed.\n\nWHEN TO ADJUST: 500 steps is a solid default. Increase to 1000-2000 if training is unstable in the first epoch (loss spikes then recovers). Decrease to 100-200 if your dataset is small and 500 steps covers too large a fraction of the epoch. Set to 0 to disable warmup -- only recommended if you're fine-tuning an already well-trained model.\n\nDefault: 500 steps.");
        y += dy;
    }

    // Max Positions
    {
        HWND lbl = mkLabel(pane, L"Max Positions", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MAXPOS, L"300000", ex, y, ew, 20);
        g_edits[ID_EDIT_MAXPOS] = ed;
        AddTooltip(lbl, L"WHAT: The maximum number of positions loaded from each training data file per generation. When set, only the first N positions from each dataset (self-play, draws, replay) are used. 0 means no limit -- use all available positions.\n\nWHY: Self-play can generate millions of positions per generation, but training on all of them may not be necessary or practical. Capping the count speeds up each training epoch, reduces VRAM usage, and can actually improve model quality by preventing the optimizer from over-fitting to a single generation's data distribution.\n\nWHEN TO ADJUST: 300000 is a good balance for most runs. Increase to 500000-1000000 if you have fast hardware and want maximum data utilization. Decrease to 100000-200000 for faster iteration during early development. If training takes too long per generation, this is the first knob to turn. Set to 0 only if your games-per-gen is already low.\n\nDefault: 300000 positions.");
        y += dy;
    }

    // Early Stop
    {
        HWND lbl = mkLabel(pane, L"Early Stop", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_EARLYSTOP, L"10", ex, y, ew, 20);
        g_edits[ID_EDIT_EARLYSTOP] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive epochs with no improvement in validation loss before training is automatically stopped for the current generation. The best model checkpoint (lowest validation loss) is kept.\n\nWHY: After a certain point, additional epochs only improve training loss while validation loss stagnates or worsens -- classic overfitting. Early stopping detects this plateau and saves time by ending training before all epochs are exhausted. The best weights are preserved, so no progress is lost.\n\nWHEN TO ADJUST: 10 epochs of patience is a safe default -- generous enough to ride out temporary plateaus. Decrease to 3-5 for faster iteration if you're confident the model converges quickly. Increase to 15-20 if your learning rate schedule has warm restarts (Cosine T0) that cause temporary loss increases before improving. Set very high (999) to effectively disable early stopping and always train for all epochs.\n\nDefault: 10 epochs patience.");
        y += dy;
    }

    // ── Section header: LR Schedule ──────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- LR Schedule ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_fUI,TRUE);
        y += 20;
    }

    // Cosine LR checkbox
    {
        g_hChkCosineLR = mkCheck(pane, ID_CHK_COSINELR, L"Cosine LR Schedule", lx, y, PW-16, 20, true);
        AddTooltip(g_hChkCosineLR, L"WHAT: Enables cosine annealing, which smoothly decays the learning rate from the initial value down to near-zero following a cosine curve shape over the course of training.\n\nWHY: A constant learning rate is suboptimal -- early training benefits from large steps to explore the loss landscape, while later training needs small steps to fine-tune. Cosine annealing provides this naturally: the LR starts high, slowly decreases through the middle, and gently approaches zero at the end. This consistently outperforms constant LR in neural network training.\n\nWHEN TO ADJUST: Keep enabled for virtually all training runs. Disable only for debugging or if you're using a custom external LR schedule. When enabled, the initial Learning Rate value becomes the peak of the cosine curve.\n\nDefault: Enabled.");
        y += dy;
    }

    // Cosine T0
    {
        HWND lbl = mkLabel(pane, L"Cosine T0", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_COSINET0, L"50", ex, y, ew, 20);
        g_edits[ID_EDIT_COSINET0] = ed;
        AddTooltip(lbl, L"WHAT: The period (in epochs) for cosine warm restarts. Every T0 epochs, the learning rate jumps back to its initial value and begins decaying again. Set to 0 for a single smooth decay over all epochs with no restarts.\n\nWHY: Warm restarts periodically shake the model out of local minima by briefly increasing the learning rate. This can help the model discover better solutions it would miss with monotonic decay. The technique (SGDR) has been shown to improve generalization, especially when combined with SWA which averages checkpoints across restart cycles.\n\nWHEN TO ADJUST: Set to 0 for simple cosine decay without restarts (safest default). Try T0 = Epochs/2 or Epochs/3 for 2-3 restart cycles per generation. Shorter periods (5-10) create frequent restarts good for exploration but may prevent convergence. Longer periods (50+) behave more like plain cosine decay. Only effective when Cosine LR is enabled.\n\nDefault: 50 epochs (one restart if training runs 100 epochs).");
        y += dy;
    }

    // SWA checkbox
    {
        g_hChkSWA = mkCheck(pane, ID_CHK_SWA, L"SWA (Stochastic Wt Avg)", lx, y, PW-16, 20, true);
        AddTooltip(g_hChkSWA, L"WHAT: Enables Stochastic Weight Averaging, which maintains a running average of model weights collected at regular intervals during training. The averaged model is saved alongside the standard best-validation model.\n\nWHY: Individual model checkpoints can be noisy -- they happen to perform well on the validation set but may be in a sharp, fragile minimum. SWA averages multiple checkpoints, producing a model that sits in a wider, flatter minimum of the loss landscape. Flatter minima generalize better to unseen positions, often producing 10-30 Elo improvement over the single best checkpoint.\n\nWHEN TO ADJUST: Keep enabled for most training runs -- it's essentially free extra strength. Disable only if you want to simplify debugging or if SWA models are consistently weaker than best-val models (rare, but possible with very small datasets). SWA pairs especially well with cosine warm restarts.\n\nDefault: Enabled.");
        y += dy;
    }

    // SWA Start
    {
        HWND lbl = mkLabel(pane, L"SWA Start Epoch", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SWASTART, L"3", ex, y, ew, 20);
        g_edits[ID_EDIT_SWASTART] = ed;
        AddTooltip(lbl, L"WHAT: The epoch number after which SWA begins collecting model snapshots for averaging. Before this epoch, only standard training occurs.\n\nWHY: Early epochs produce rapidly-changing, unstable weights as the model makes large adjustments. Including these early snapshots in the SWA average would drag it toward poor solutions. By waiting until the model has partially converged, SWA only averages high-quality checkpoints from the refinement phase of training.\n\nWHEN TO ADJUST: 3 is a good default for typical 10-15 epoch runs. For longer runs (50+ epochs), set to 25-50%% of total epochs. If you're using warm restarts (Cosine T0), set SWA Start to at least T0 so the model completes one full cosine cycle before averaging begins. Setting too high leaves too few snapshots to average, reducing SWA's benefit.\n\nDefault: Epoch 3.");
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
        HWND ed  = mkEdit(pane, ID_EDIT_DRAWWT, L"0.5", ex, y, ew, 20);
        g_edits[ID_EDIT_DRAWWT] = ed;
        AddTooltip(lbl, L"WHAT: A multiplier applied to the training loss for positions from drawn games. A value of 0.5 means drawn positions contribute half as much to the gradient as decisive (win/loss) positions.\n\nWHY: Draws are the most common outcome in strong chess. Without down-weighting, drawn positions dominate the training data and the model learns to predict 0.50 (draw) too eagerly. De-emphasizing draws forces the model to pay more attention to the distinguishing features of winning and losing positions, producing sharper, more decisive evaluations.\n\nWHEN TO ADJUST: 0.5 is the recommended value. Decrease to 0.2-0.3 if your draw rate is very high (60%%+) and you want even more emphasis on decisive games. Increase to 0.7-1.0 if draws are rare or if the engine misjudges drawn positions (e.g. calling drawn endgames as winning). Setting to 0 completely ignores draws in training -- extreme but sometimes useful for tactical-style training.\n\nDefault: 0.5 (draws at half weight).");
        y += dy;
    }

    // Mate Boost
    {
        HWND lbl = mkLabel(pane, L"Mate Boost", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MATEBOOST, L"3.0", ex, y, ew, 20);
        g_edits[ID_EDIT_MATEBOOST] = ed;
        AddTooltip(lbl, L"WHAT: A loss multiplier applied to positions that are near checkmate (mate-in-N). Positions closer to mate receive a higher weight in the training loss, proportional to this value.\n\nWHY: Mating positions are rare in training data but critically important for playing strength. Without boosting, the model treats a position with mate-in-3 the same as a quiet middlegame advantage. The boost ensures the model develops strong pattern recognition for mating nets, back-rank threats, and king-hunt sequences -- skills that directly win games.\n\nWHEN TO ADJUST: 3.0 is a strong default. Increase to 5.0-10.0 if the engine frequently misses forced mates or mishandles winning endgames. Decrease to 1.0-2.0 if the engine becomes too tactics-focused and neglects positional play. Set to 1.0 to disable the boost entirely (all positions weighted equally regardless of proximity to mate).\n\nDefault: 3.0× weight for mating positions.");
        y += dy;
    }

    // Self-Play Ratio
    {
        HWND lbl = mkLabel(pane, L"Self-Play Ratio", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SPLRATIO, L"0.4", ex, y, ew, 20);
        g_edits[ID_EDIT_SPLRATIO] = ed;
        AddTooltip(lbl, L"WHAT: The fraction of the training dataset that comes from the current generation's self-play data versus replayed data from older generations. A value of 0.4 means 40%% current-gen self-play, with the remainder filled by replay data and other sources.\n\nWHY: Current-generation data reflects the model's latest strength level and is the most relevant for improvement. However, training only on current data causes catastrophic forgetting -- the model loses knowledge from earlier training. Mixing in older data (via Replay Window) provides stability and prevents the model from forgetting lessons learned in previous generations.\n\nWHEN TO ADJUST: 0.4 is balanced. Increase to 0.6-0.8 if you want faster adaptation to new patterns (good for early training). Decrease to 0.2-0.3 for more conservative training that preserves past knowledge (good for late-stage refinement). If Replay Window is 0 (no replay), this value has reduced effect since there's no older data to mix.\n\nDefault: 0.4 (40%% current generation data).");
        y += dy;
    }

    // Draw Ratio %
    {
        HWND lbl = mkLabel(pane, L"Draw Ratio %", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAWPCT, L"10", ex, y, ew, 20);
        g_edits[ID_EDIT_DRAWPCT] = ed;
        AddTooltip(lbl, L"WHAT: The percentage of the Max Positions budget allocated to drawn game positions (from training_data_draws.bin). The remaining budget is split between decisive positions and self-play data.\n\nWHY: Draw data contains important positional knowledge (equal structures, fortresses, theoretical draws) but is extremely abundant -- often 50-60%% of all games are draws. Without capping, draws would flood the training set and dilute the decisive games that teach the model how to win and lose. This cap ensures a controlled proportion.\n\nWHEN TO ADJUST: 10%% is a good default. Increase to 15-25%% if the engine misjudges drawn endgames or fails to recognize fortress positions. Decrease to 5%% or less if you want maximum emphasis on decisive game positions and your draw rate is very high. Note: this interacts with Draw Weight -- both together control how much influence draws have on training.\n\nDefault: 10%%.");
        y += dy;
    }

    // FRC Mix %
    {
        HWND lbl = mkLabel(pane, L"FRC Mix %", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_FRCMIX, L"0", ex, y, ew, 20);
        g_edits[ID_EDIT_FRCMIX] = ed;
        AddTooltip(lbl, L"WHAT: The percentage of self-play games that begin from random Chess960 (Fischer Random) starting positions instead of standard chess openings. The remaining games use positions from the opening book.\n\nWHY: Standard chess openings produce recurring position types -- the same pawn structures and piece placements appear repeatedly. Chess960 forces the engine to evaluate unfamiliar piece configurations from move 1, teaching it general positional principles rather than memorized opening patterns. This produces a more robust, adaptable engine.\n\nWHEN TO ADJUST: 0%% for pure standard chess training. 10-20%% for a good diversity boost without straying too far from standard play. 30-50%% if you want a highly creative, unconventional engine. Above 50%% the engine becomes very strong in random positions but may underperform in standard openings where book knowledge matters. Ensure your engine supports FRC before enabling.\n\nDefault: 0%% (standard chess only).");
        y += dy;
    }

    // WDL Alpha
    {
        HWND lbl = mkLabel(pane, L"WDL Alpha", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WDLALPHA, L"0.5", ex, y, ew, 20);
        g_edits[ID_EDIT_WDLALPHA] = ed;
        AddTooltip(lbl, L"WHAT: The blending factor between two loss functions: Mean Squared Error on raw evaluation (alpha = 0.0) and Win/Draw/Loss cross-entropy (alpha = 1.0). At 0.5, both losses contribute equally to the training gradient.\n\nWHY: MSE loss teaches the model to predict accurate centipawn evaluations -- the raw numerical score. WDL loss teaches it to predict the probability of winning, drawing, or losing. Blending both produces an engine whose evaluation is both numerically accurate and well-calibrated to actual game outcomes. Pure MSE can produce evaluations that are precise but poorly correlated with win probability; pure WDL can produce good probability estimates but imprecise centipawn values.\n\nWHEN TO ADJUST: 0.5 (equal blend) is a strong default. Shift toward 0.3 if you want more precise centipawn evaluations (good for analysis). Shift toward 0.7 if you want better win-probability calibration (good for playing strength). Extreme values (0.0 or 1.0) use only one loss function -- useful for experimentation but generally weaker than a blend.\n\nDefault: 0.5 (equal MSE and WDL blend).");
        y += dy;
    }

    // WDL Draw Elo
    {
        HWND lbl = mkLabel(pane, L"WDL Draw Elo", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WDLDRAWELO, L"100", ex, y, ew, 20);
        g_edits[ID_EDIT_WDLDRAWELO] = ed;
        AddTooltip(lbl, L"WHAT: Controls the width of the draw band when converting centipawn evaluations to Win/Draw/Loss targets. A higher value means a wider range of evaluations are mapped to high draw probability. Measured in Elo-scaled units.\n\nWHY: In real chess, positions evaluated at +0.30 (30 centipawns) are almost always draws between strong players, while +2.00 is usually a win. The Draw Elo parameter shapes this mapping curve -- it determines how much of an evaluation advantage is needed before the WDL target shifts from 'likely draw' to 'likely win'. Getting this right ensures training targets match realistic game outcomes.\n\nWHEN TO ADJUST: 100 is calibrated for typical engine play. Lower to 50-70 for a narrower draw band that pushes the model to be more decisive -- small advantages get labeled as potential wins, encouraging aggressive play. Raise to 130-180 for a wider draw band -- the model learns that even moderate advantages often result in draws, producing more conservative/positional play.\n\nDefault: 100.");
        y += dy;
    }

    // Replay Window
    {
        HWND lbl = mkLabel(pane, L"Replay Window", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_REPLAYWIN, L"3", ex, y, ew, 20);
        g_edits[ID_EDIT_REPLAYWIN] = ed;
        AddTooltip(lbl, L"WHAT: The number of past generations whose self-play data is mixed into the current generation's training set. A value of 3 means positions from the current generation plus the 3 most recent previous generations are used.\n\nWHY: Each generation's self-play data represents a snapshot of the model's strength at that point. Replaying older data provides training stability and prevents catastrophic forgetting -- where the model loses skills it learned earlier. It also increases the effective dataset size without additional self-play computation. Older positions are weighted less via Replay Decay so recent, higher-quality data dominates.\n\nWHEN TO ADJUST: 3-5 is the sweet spot. Increase to 7-10 if training is unstable between generations (large loss spikes). Set to 0 to use only current-gen data -- fast but prone to forgetting and overfitting. Very large windows (15+) dilute the training set with low-quality early-generation data, slowing improvement.\n\nDefault: 3 generations.");
        y += dy;
    }

    // Replay Decay
    {
        HWND lbl = mkLabel(pane, L"Replay Decay", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_REPLAYDECAY, L"0.7", ex, y, ew, 20);
        g_edits[ID_EDIT_REPLAYDECAY] = ed;
        AddTooltip(lbl, L"WHAT: The exponential decay factor applied to replayed positions based on their generation age. Each generation back, the sampling weight is multiplied by this value. With decay 0.7: gen-1 gets 70%%, gen-2 gets 49%% (0.7²), gen-3 gets 34%% (0.7³).\n\nWHY: Older self-play data was generated by weaker versions of the model and contains more mistakes and suboptimal evaluations. While this data still provides useful diversity, it should be down-weighted relative to recent, higher-quality data. The exponential decay naturally phases out stale data while keeping it available for stability.\n\nWHEN TO ADJUST: 0.7 is a strong default that significantly reduces old data influence while retaining it. Increase to 0.8-0.9 for more equal weighting across generations (useful if improvement between generations is small). Decrease to 0.4-0.6 to heavily favor recent data (useful when the model is improving rapidly and old data is outdated). Set to 1.0 for uniform weighting (all replayed generations treated equally).\n\nDefault: 0.7 (each generation back gets 70%% of the previous generation's weight).");
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
        AddTooltip(g_hChkElo, L"WHAT: When enabled, an automated match is played between the newly trained model and the previous generation's model after each training cycle. The result is converted to an estimated Elo difference.\n\nWHY: Loss and accuracy metrics don't always correlate with playing strength -- a model can have lower loss but play weaker chess due to evaluation blind spots. Elo validation provides ground-truth measurement of whether the new model actually plays better. It catches regressions that loss metrics would miss.\n\nWHEN TO ADJUST: Enable for production training runs where you need confidence that each generation is improving. Disable for fast iteration or early experiments -- Elo matches add significant time to each generation (proportional to ELO Games setting). Also useful to disable temporarily if you're tuning hyperparameters and using validation loss as your primary metric.\n\nDefault: Disabled (loss-based evaluation only).");
        y += dy;
    }

    // ELO Games
    {
        HWND lbl = mkLabel(pane, L"ELO Games", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_ELOGAMES, L"100", ex, y, ew, 20);
        g_edits[ID_EDIT_ELOGAMES] = ed;
        AddTooltip(lbl, L"WHAT: The number of games played in the Elo validation match between the new model and the previous generation's model. Games are split evenly between playing white and black.\n\nWHY: Elo estimation from match results has statistical uncertainty that decreases with more games. A 10-game match can easily give misleading results (±100 Elo error). A 100-game match narrows uncertainty to roughly ±30 Elo. More games = more confidence that measured improvement is real and not noise.\n\nWHEN TO ADJUST: 100 is a reasonable default balancing accuracy with time cost. Increase to 200-500 for high-confidence measurements (important for late-stage training where improvements are small). Decrease to 50 for rough estimates during development. Below 30 games, Elo estimates are essentially meaningless noise. Only applies when ELO Validation is enabled.\n\nDefault: 100 games.");
        y += dy;
    }

    // SWA Games
    {
        HWND lbl = mkLabel(pane, L"SWA Games", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SWAGAMES, L"50", ex, y, ew, 20);
        g_edits[ID_EDIT_SWAGAMES] = ed;
        AddTooltip(lbl, L"WHAT: The number of games played in a match between the SWA-averaged model and the best-validation-loss model to determine which is stronger. Games are split between white and black.\n\nWHY: SWA produces a separate model from a different optimization strategy (weight averaging vs best checkpoint). Sometimes SWA is stronger, sometimes best-val wins. This match determines which model to carry forward as the generation's final weights, ensuring you always keep the stronger one.\n\nWHEN TO ADJUST: 50 games provides a reasonable signal. Increase to 100-200 if SWA and best-val models are close in strength and you want a more reliable comparison. Decrease to 30 for speed during development. Only applies when both SWA and ELO Validation are enabled.\n\nDefault: 50 games.");
        y += dy;
    }

    // Overfitting Detection checkbox
    {
        g_hChkOvfit = mkCheck(pane, ID_CHK_OVERFIT, L"Overfitting Detection", lx, y, PW-16, 20, true);
        AddTooltip(g_hChkOvfit, L"WHAT: Monitors the gap between training loss and validation loss during each epoch. If the gap exceeds a threshold (training loss dropping while validation loss rises or stagnates), training is flagged as overfitting and may be stopped early.\n\nWHY: Overfitting means the model is memorizing training positions rather than learning generalizable patterns. An overfitting model performs worse on new positions despite appearing to improve on training data. Early detection saves time and prevents adopting a degraded model. This is complementary to Early Stop -- Early Stop watches validation loss alone, while Overfitting Detection watches the divergence between train and val.\n\nWHEN TO ADJUST: Keep enabled for most training runs -- it's a safety net with no downside. Disable only if you're intentionally training for many epochs on a small dataset and expect some train/val divergence as normal (e.g. fine-tuning on a curated position set).\n\nDefault: Enabled.");
        y += dy;
    }

    // Record total content height for scrolling
    g_cfgTotalH = y + 8;
    g_cfgScrollY = 0;
}


// ── Config panel subclass (dark background + scrolling + label colors) ────────

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
    c.drawWeight   = pDbl(e(ID_EDIT_DRAWWT), 0.5);
    c.mateBoost    = pDbl(e(ID_EDIT_MATEBOOST),3.0);
    c.splRatio     = pDbl(e(ID_EDIT_SPLRATIO),0.4);
    c.workers      = pInt(e(ID_EDIT_WORKERS), 12);
    c.depth        = pInt(e(ID_EDIT_DEPTH),    5);
    c.startGen     = pInt(e(ID_EDIT_STARTGEN), 0);
    c.maxPositions = pInt(e(ID_EDIT_MAXPOS), 300000);
    c.earlyStop    = pInt(e(ID_EDIT_EARLYSTOP), 10);
    c.eloValidate   = g_hChkElo   ? Button_GetCheck(g_hChkElo)   == BST_CHECKED : false;
    c.overfitDetect = g_hChkOvfit ? Button_GetCheck(g_hChkOvfit) == BST_CHECKED : true;
    c.cosineLr      = g_hChkCosineLR ? Button_GetCheck(g_hChkCosineLR) == BST_CHECKED : true;
    c.cosineT0      = pInt(e(ID_EDIT_COSINET0), 0);
    c.swa           = g_hChkSWA ? Button_GetCheck(g_hChkSWA) == BST_CHECKED : true;
    c.swaStart      = pInt(e(ID_EDIT_SWASTART), 3);
    c.drawPct       = pDbl(e(ID_EDIT_DRAWPCT), 10.0);
    c.frcMix        = pDbl(e(ID_EDIT_FRCMIX), 0.0);
    c.replayWindow  = pInt(e(ID_EDIT_REPLAYWIN), 3);
    c.replayDecay   = pDbl(e(ID_EDIT_REPLAYDECAY), 0.7);
    c.eloGames      = pInt(e(ID_EDIT_ELOGAMES), 100);
    c.swaGames      = pInt(e(ID_EDIT_SWAGAMES), 50);
    c.wdlAlpha      = pDbl(e(ID_EDIT_WDLALPHA), 0.5);
    c.wdlDrawElo    = pDbl(e(ID_EDIT_WDLDRAWELO), 100.0);
    c.mixedDepthLow   = pInt(e(ID_EDIT_MIXDEPTH_LOW), 4);
    c.mixedDepthRatio = pDbl(e(ID_EDIT_MIXDEPTH_PCT), 0.0) / 100.0;
    c.resignCp      = pInt(e(ID_EDIT_RESIGNCP), 500);
    c.contemptCp    = pInt(e(ID_EDIT_CONTEMPT), 25);
    c.maxPlies      = pInt(e(ID_EDIT_MAXPLIES), 250);
    c.drawCp        = pInt(e(ID_EDIT_DRAWCP), 8);
    c.depthShuffle  = g_hChkDepthShuffle ? Button_GetCheck(g_hChkDepthShuffle) == BST_CHECKED : false;
    c.depthShuffleBias = pDbl(e(ID_EDIT_DEPTH_SHUFFLE_BIAS), 2.0);
    c.openingTemp   = pDbl(e(ID_EDIT_OPENING_TEMP), 1.5);
    c.openingPlies  = pInt(e(ID_EDIT_OPENING_PLIES), 4);
    c.softmaxPlies  = pInt(e(ID_EDIT_SOFTMAX_PLIES), 8);
    c.softmaxTemp   = pDbl(e(ID_EDIT_SOFTMAX_TEMP), 0.5);
    c.rootNoiseEps  = pDbl(e(ID_EDIT_ROOT_NOISE), 0.0);
    c.recordMinPly  = pInt(e(ID_EDIT_RECORD_MIN_PLY), 10);
    c.recordMaxEval = pInt(e(ID_EDIT_RECORD_MAX_EVAL), 2500);
    c.resignCount   = pInt(e(ID_EDIT_RESIGN_COUNT), 3);
    c.drawCount     = pInt(e(ID_EDIT_DRAW_COUNT), 6);
    c.drawMinPly    = pInt(e(ID_EDIT_DRAW_MIN_PLY), 40);
    c.drawAdjMoves  = pInt(e(ID_EDIT_DRAW_ADJ_MOVES), 12);
    c.drawAdjThreshold = pInt(e(ID_EDIT_DRAW_ADJ_THRESH), 4);
    c.drawAdjMinMove   = pInt(e(ID_EDIT_DRAW_ADJ_MIN_MOVE), 50);
    int variantSel = g_hVariant ? (int)SendMessageW(g_hVariant, CB_GETCURSEL, 0, 0) : 0;
    c.variant = (variantSel == 1) ? ChessVariant::DuckChess :
                (variantSel == 2) ? ChessVariant::Automate :
                                    ChessVariant::Standard;
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
    se(ID_EDIT_GRADACCUM, std::to_string(p.gradAccum));
    se(ID_EDIT_WARMUP,    std::to_string(p.warmupSteps));
    se(ID_EDIT_MAXPOS,    std::to_string(p.maxPositions));
    se(ID_EDIT_EARLYSTOP, std::to_string(p.earlyStop));
    se(ID_EDIT_LR,        dbl2s(p.lr,6));
    se(ID_EDIT_WD,        dbl2s(p.weightDecay,8));
    se(ID_EDIT_DROPOUT,   dbl2s(p.dropout,4));
    se(ID_EDIT_LSMOOTH,   dbl2s(p.labelSmooth,4));
    se(ID_EDIT_DRAWWT,    dbl2s(p.drawWeight,4));
    se(ID_EDIT_MATEBOOST, dbl2s(p.mateBoost,4));
    se(ID_EDIT_SPLRATIO,  dbl2s(p.splRatio,4));
    se(ID_EDIT_DRAWPCT,   dbl2s(p.drawPct,2));
    se(ID_EDIT_FRCMIX,    dbl2s(p.frcMix,3));
    se(ID_EDIT_REPLAYWIN, std::to_string(p.replayWindow));
    se(ID_EDIT_REPLAYDECAY, dbl2s(p.replayDecay,2));
    se(ID_EDIT_ELOGAMES,  std::to_string(p.eloGames));
    se(ID_EDIT_SWAGAMES,  std::to_string(p.swaGames));
    se(ID_EDIT_WDLALPHA,  dbl2s(p.wdlAlpha,4));
    se(ID_EDIT_WDLDRAWELO,dbl2s(p.wdlDrawElo,2));
    se(ID_EDIT_COSINET0,  std::to_string(p.cosineT0));
    se(ID_EDIT_SWASTART,  std::to_string(p.swaStart));
    se(ID_EDIT_MIXDEPTH_PCT, dbl2s(p.mixedDepthRatio * 100.0, 0));
    se(ID_EDIT_MIXDEPTH_LOW, std::to_string(p.mixedDepthLow));
    se(ID_EDIT_RESIGNCP,  std::to_string(p.resignCp));
    se(ID_EDIT_CONTEMPT,  std::to_string(p.contemptCp));
    se(ID_EDIT_MAXPLIES,  std::to_string(p.maxPlies));
    se(ID_EDIT_DRAWCP,    std::to_string(p.drawCp));
    se(ID_EDIT_DEPTH_SHUFFLE_BIAS, dbl2s(p.depthShuffleBias,2));
    se(ID_EDIT_OPENING_TEMP,  dbl2s(p.openingTemp,2));
    se(ID_EDIT_OPENING_PLIES, std::to_string(p.openingPlies));
    se(ID_EDIT_SOFTMAX_PLIES, std::to_string(p.softmaxPlies));
    se(ID_EDIT_SOFTMAX_TEMP,  dbl2s(p.softmaxTemp,2));
    se(ID_EDIT_ROOT_NOISE,    dbl2s(p.rootNoiseEps,3));
    se(ID_EDIT_RECORD_MIN_PLY,  std::to_string(p.recordMinPly));
    se(ID_EDIT_RECORD_MAX_EVAL, std::to_string(p.recordMaxEval));
    se(ID_EDIT_RESIGN_COUNT,    std::to_string(p.resignCount));
    se(ID_EDIT_DRAW_COUNT,      std::to_string(p.drawCount));
    se(ID_EDIT_DRAW_MIN_PLY,    std::to_string(p.drawMinPly));
    se(ID_EDIT_DRAW_ADJ_MOVES,  std::to_string(p.drawAdjMoves));
    se(ID_EDIT_DRAW_ADJ_THRESH, std::to_string(p.drawAdjThreshold));
    se(ID_EDIT_DRAW_ADJ_MIN_MOVE, std::to_string(p.drawAdjMinMove));
    if (g_hChkElo)          Button_SetCheck(g_hChkElo,          p.eloValidate   ? BST_CHECKED : BST_UNCHECKED);
    if (g_hChkOvfit)        Button_SetCheck(g_hChkOvfit,        p.overfitDetect ? BST_CHECKED : BST_UNCHECKED);
    if (g_hChkCosineLR)     Button_SetCheck(g_hChkCosineLR,     p.cosineLr      ? BST_CHECKED : BST_UNCHECKED);
    if (g_hChkSWA)          Button_SetCheck(g_hChkSWA,          p.swa           ? BST_CHECKED : BST_UNCHECKED);
    if (g_hChkDepthShuffle) Button_SetCheck(g_hChkDepthShuffle, p.depthShuffle  ? BST_CHECKED : BST_UNCHECKED);
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
    np.depth = c.depth;
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
// ── Config panel scroll helpers ───────────────────────────────────
static void UpdateCfgScroll(HWND hw) {
    RECT rc; GetClientRect(hw, &rc);
    int pageH = rc.bottom;
    SCROLLINFO si{}; si.cbSize = sizeof(si);
    si.fMask  = SIF_RANGE | SIF_PAGE | SIF_POS;
    si.nMin   = 0;
    si.nMax   = g_cfgTotalH;
    si.nPage  = pageH;
    si.nPos   = g_cfgScrollY;
    SetScrollInfo(hw, SB_VERT, &si, TRUE);
}

static void ScrollCfgTo(HWND hw, int newPos) {
    RECT rc; GetClientRect(hw, &rc);
    int pageH = rc.bottom;
    int maxPos = std::max(0, g_cfgTotalH - pageH);
    newPos = std::max(0, std::min(newPos, maxPos));
    if (newPos == g_cfgScrollY) return;
    int delta = g_cfgScrollY - newPos;
    g_cfgScrollY = newPos;
    ScrollWindowEx(hw, 0, delta, nullptr, nullptr, nullptr, nullptr, SW_SCROLLCHILDREN | SW_INVALIDATE);
    UpdateCfgScroll(hw);
}

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
    if (msg == WM_MOUSEWHEEL) {
        int delta = GET_WHEEL_DELTA_WPARAM(wp);
        ScrollCfgTo(hw, g_cfgScrollY - delta / 3);
        return 0;
    }
    if (msg == WM_SIZE) {
        UpdateCfgScroll(hw);
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
            EnableWindow(g_hSkip,  TRUE);
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
        else if (id == ID_BTN_SKIP && g_st.running) {
            // Skip current phase — terminate subprocess, pipeline continues to next phase
            g_skipFlag.store(true);
            g_st.pushLog("[SKIP] Skipping current phase...");
            // Resume if paused so the process can be terminated cleanly
            if (g_pauseFlag.load()) {
                g_pauseFlag.store(false);
                ResumeProcessThreads(g_activePid.load());
                SetWindowTextW(g_hPause, L"\u23F8 Pause");
            }
            SuspendOrTerminateActive();
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
        else if (id == ID_BTN_LATEST_GEN) {
            Config cfg = ReadConfig();
            int latest = findLatestGen(cfg.dataDir, cfg.variant);
            auto it = g_edits.find(ID_EDIT_STARTGEN);
            if (it != g_edits.end()) setEdit(it->second, std::to_string(latest));
        }
        else if (id == ID_BTN_BEST_GEN) {
            Config cfg = ReadConfig();
            int best = findBestGenFor(cfg.dataDir, cfg.variant);
            auto it = g_edits.find(ID_EDIT_STARTGEN);
            if (it != g_edits.end()) setEdit(it->second, std::to_string(best));
        }
        else if (id == ID_COMBO_VARIANT && HIWORD(wp) == CBN_SELCHANGE) {
            // Variant changed — update Start Gen and reload graph for the new variant
            Config cfg = ReadConfig();
            int latest = findLatestGen(cfg.dataDir, cfg.variant);
            auto it = g_edits.find(ID_EDIT_STARTGEN);
            if (it != g_edits.end())
                setEdit(it->second, std::to_string(latest));
            // Reload graph data for the selected variant
            { std::lock_guard<std::mutex> lk(g_st.mtx); LoadGraphCsv(cfg.variant); }
            InvalidateRect(g_hGraph, nullptr, FALSE);
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
        else if (id == ID_CHK_GRAPH_PHASE) {
            g_showPhase = (Button_GetCheck(g_hChkGPhase) == BST_CHECKED);
            InvalidateRect(g_hGraph, nullptr, FALSE);
        }
        else if (id == ID_CHK_GRAPH_NPS) {
            g_showNPS = (Button_GetCheck(g_hChkGNPS) == BST_CHECKED);
            InvalidateRect(g_hGraph, nullptr, FALSE);
        }
        else if (id == ID_CHK_MUTE_SOUNDS) {
            g_muteSounds = (Button_GetCheck(g_hChkMute) == BST_CHECKED);
        }
        break;
    }

    case WM_USER+1: {
        // Pipeline finished
        EnableWindow(g_hStart, TRUE);
        EnableWindow(g_hStop,  FALSE);
        EnableWindow(g_hPause, FALSE);
        EnableWindow(g_hSkip,  FALSE);
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
            // Also read batch progress for training phase
            int curBatch = 0, totalBatches = 0;
            {
                std::lock_guard<std::mutex> lk(g_st.mtx);
                curBatch     = g_st.curBatch;
                totalBatches = g_st.totalBatches;
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
                std::chrono::steady_clock::time_point bStamp, eStamp, nStamp, spStamp;
                int doneGens, bEta, eEta, nEta, spEta;
                {
                    std::lock_guard<std::mutex> lk(g_st.mtx);
                    pStart   = g_st.pipelineStart;
                    phStart  = g_st.phaseStart;
                    doneGens = g_st.completedGens;
                    bEta     = g_st.batchEtaSec;
                    eEta     = g_st.epochEtaSec;
                    nEta     = g_st.nextEpochSec;
                    spEta    = g_st.selfPlayEtaSec;
                    bStamp   = g_st.batchEtaStamp;
                    eStamp   = g_st.epochEtaStamp;
                    nStamp   = g_st.nextEpochStamp;
                    spStamp  = g_st.selfPlayEtaStamp;
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
                    if (totalEp > 0) {
                        progressDetail = L"  Epoch: " + std::to_wstring(curEp) + L"/" + std::to_wstring(totalEp);
                        if (totalBatches > 0)
                            progressDetail += L"  Batch: " + std::to_wstring(curBatch) + L"/" + std::to_wstring(totalBatches);
                    }
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

                // 3. Phase-specific countdowns
                if (isTraining) {
                    long long bLeft = countdown(bEta, bStamp);
                    long long eLeft = countdown(eEta, eStamp);
                    long long nLeft = countdown(nEta, nStamp);

                    if (bLeft >= 0)
                        bs << L"  |  Batch ETA: ~" << fmtDur(bLeft);
                    if (nLeft >= 0)
                        bs << L"  |  Next epoch in: ~" << fmtDur(nLeft);
                    if (eLeft >= 0)
                        bs << L"  |  Training done in: ~" << fmtDur(eLeft);
                }
                if (isSelfPlay) {
                    long long spLeft = countdown(spEta, spStamp);
                    if (spLeft >= 0)
                        bs << L"  |  Self-play done in: ~" << fmtDur(spLeft);
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

            // Live training countdown: update the last log line with a countdown
            // every 500ms so the output window shows live ETA during training.
            if (running && phase == "training") {
                auto now2 = std::chrono::steady_clock::now();
                int bEta2 = 0, eEta2 = 0;
                std::chrono::steady_clock::time_point bStamp2, eStamp2;
                int cb = 0, tb = 0, ce = 0, te = 0;
                {
                    std::lock_guard<std::mutex> lk(g_st.mtx);
                    bEta2 = g_st.batchEtaSec; bStamp2 = g_st.batchEtaStamp;
                    eEta2 = g_st.epochEtaSec; eStamp2 = g_st.epochEtaStamp;
                    cb = g_st.curBatch; tb = g_st.totalBatches;
                    ce = g_st.curEpoch; te = g_st.totalEpochs;
                }
                auto cdSec = [&](int eta, std::chrono::steady_clock::time_point stamp) -> long long {
                    if (eta <= 0) return -1;
                    long long el = std::chrono::duration_cast<std::chrono::seconds>(now2 - stamp).count();
                    return std::max(0LL, (long long)eta - el);
                };
                long long bLeft2 = cdSec(bEta2, bStamp2);
                long long eLeft2 = cdSec(eEta2, eStamp2);
                if (bLeft2 >= 0 || eLeft2 >= 0) {
                    // Build a status line to inject as the last log entry
                    std::string statusLine = "[Training]";
                    if (te > 0) statusLine += " Epoch " + std::to_string(ce) + "/" + std::to_string(te);
                    if (tb > 0) statusLine += "  Batch " + std::to_string(cb) + "/" + std::to_string(tb);
                    if (bLeft2 >= 0) {
                        int bm = (int)(bLeft2 / 60), bs2 = (int)(bLeft2 % 60);
                        statusLine += "  Batch ETA: " + std::to_string(bm) + "m" + std::to_string(bs2) + "s";
                    }
                    if (eLeft2 >= 0) {
                        int em = (int)(eLeft2 / 60), es2 = (int)(eLeft2 % 60);
                        statusLine += "  Training ETA: " + std::to_string(em) + "m" + std::to_string(es2) + "s";
                    }
                    // Push as \r line so it overwrites the previous countdown
                    g_st.pushLog("\r" + statusLine);
                    FlushLog();
                }
            }

            // Self-play ETA is shown in the banner (countdown(spEta, spStamp)).
            // The engine's own [SelfPlay] progress line already contains the full stats
            // and ETA — no need to inject a separate countdown line in the output window.

            // Live NPS: during self-play, maintain a single step=0 placeholder point
            // so the NPS panel shows the current NPS in real time (every 500ms timer tick).
            // Real NPS_SAMPLE points (step > 0) are pushed by the RunProc callback and
            // must NOT be overwritten here — they carry individual per-sample values.
            if (running) {
                std::lock_guard<std::mutex> lk(g_st.mtx);
                if (g_st.curNps > 0.0 && phase == "selfplay") {
                    int liveGen = g_cfg.startGen + 1 + g_st.curGen;
                    // Find or create the step=0 placeholder for this gen
                    bool found = false;
                    for (auto& p : g_st.pts) {
                        if (p.gen == liveGen && p.step == 0) {
                            p.nps    = g_st.curNps;
                            p.hasNps = true;
                            found    = true;
                            break;
                        }
                    }
                    if (!found) {
                        TrainPoint live;
                        live.gen    = liveGen;
                        live.step   = 0;
                        live.train  = 0.0;
                        live.nps    = g_st.curNps;
                        live.hasNps = true;
                        g_st.pts.push_back(live);
                    }
                }
            }

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

        // Buttons (bottom of left panel) — 4 buttons in one row
        int btnY2 = H2 - 46;
        int btnW2 = (PANEL_W - 28) / 4;
        if (g_hStart) SetWindowPos(g_hStart, nullptr, 8,              btnY2, btnW2, 28, SWP_NOZORDER);
        if (g_hStop)  SetWindowPos(g_hStop,  nullptr, 8+btnW2+4,      btnY2, btnW2, 28, SWP_NOZORDER);
        if (g_hPause) SetWindowPos(g_hPause, nullptr, 8+btnW2*2+8,    btnY2, btnW2, 28, SWP_NOZORDER);
        if (g_hSkip)  SetWindowPos(g_hSkip,  nullptr, 8+btnW2*3+12,   btnY2, btnW2, 28, SWP_NOZORDER);

        // Config pane – stop above buttons
        int paneH = btnY2 - TITLE_H - 8;
        SetWindowPos(g_hCfgPane, nullptr, 0, TITLE_H, PANEL_W, paneH, SWP_NOZORDER);
        if (g_hCfgPane) UpdateCfgScroll(g_hCfgPane);
        break;
    }

    // Scrollbar drag on the config panel — WM_VSCROLL is sent to the panel's
    // parent (this window), not to the panel itself.
    case WM_VSCROLL: {
        if (g_hCfgPane && (HWND)lp == nullptr) {
            // Came from the panel's own scrollbar (lp == 0 for standard scrollbars)
            SCROLLINFO si{}; si.cbSize = sizeof(si); si.fMask = SIF_ALL;
            GetScrollInfo(g_hCfgPane, SB_VERT, &si);
            int newPos = g_cfgScrollY;
            switch (LOWORD(wp)) {
                case SB_LINEUP:        newPos -= 20; break;
                case SB_LINEDOWN:      newPos += 20; break;
                case SB_PAGEUP:        newPos -= si.nPage; break;
                case SB_PAGEDOWN:      newPos += si.nPage; break;
                case SB_THUMBTRACK:    newPos = si.nTrackPos; break;
                case SB_THUMBPOSITION: newPos = si.nTrackPos; break;
            }
            ScrollCfgTo(g_hCfgPane, newPos);
        }
        break;
    }

    // Forward mouse wheel to config panel when cursor is over it
    case WM_MOUSEWHEEL: {
        if (g_hCfgPane) {
            POINT pt; GetCursorPos(&pt);
            RECT rc; GetWindowRect(g_hCfgPane, &rc);
            if (PtInRect(&rc, pt)) {
                int delta = GET_WHEEL_DELTA_WPARAM(wp);
                ScrollCfgTo(g_hCfgPane, g_cfgScrollY - delta / 3);
            }
        }
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

    // Create a Job Object so all child processes (ChessEngine.exe, py) are
    // automatically killed if TrainingRunner crashes or exits unexpectedly.
    g_hJob = CreateJobObjectW(nullptr, nullptr);
    if (g_hJob) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
        jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(g_hJob, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
    }

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
                                  WS_CHILD|WS_VISIBLE|SS_LEFT|WS_VSCROLL,
                                  0, TITLE_H, PANEL_W, H2-TITLE_H-60,
                                  g_hWnd, nullptr, g_hInst, nullptr);
    SetWindowSubclass(g_hCfgPane, PanelProc, 1, 0);
    BuildConfigPane(g_hCfgPane, PANEL_W);
    UpdateCfgScroll(g_hCfgPane);

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

    // Buttons (bottom of left panel) — 4 buttons in one row
    int btnY = H2 - 46;
    int btnW = (PANEL_W - 28) / 4;  // 4 buttons with 3 gaps of 4px + 8px margins
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

    g_hSkip = CreateWindowExW(0, L"BUTTON", L"\u23ED Skip",
                               WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                               8+btnW*3+12, btnY, btnW, 28,
                               g_hWnd, (HMENU)(LONG_PTR)ID_BTN_SKIP, g_hInst, nullptr);
    SendMessageW(g_hSkip, WM_SETFONT, (WPARAM)g_fUI, TRUE);
    EnableWindow(g_hSkip, FALSE);

    // Initialize preset system
    InitBuiltinPresets();
    LoadCustomPresets();
    PopulatePresetCombo();
    ApplyPreset(0);  // default: Quick Test

    // Auto-fill Start Gen with the latest completed generation
    {
        Config cfg = ReadConfig();
        int latest = findLatestGen(cfg.dataDir, cfg.variant);
        if (latest > 0) {
            auto it = g_edits.find(ID_EDIT_STARTGEN);
            if (it != g_edits.end())
                setEdit(it->second, std::to_string(latest));
        }
    }

    // Load persisted graph data so the graph shows history from previous runs
    {
        Config cfg = ReadConfig();
        std::lock_guard<std::mutex> lk(g_st.mtx);
        LoadGraphCsv(cfg.variant);
    }

    // Timer for UI updates
    SetTimer(g_hWnd, ID_TIMER, 500, nullptr);

    ShowWindow(g_hWnd, SW_SHOWMAXIMIZED);
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

