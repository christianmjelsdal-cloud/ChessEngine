#include "VisualGame.h"
#include "UCI.h"
#include "AssetPath.h"
#include "SelfPlayGen.h"
#include "Bitboard.h"
#include "NNUETrainer.h"
#include "DuckNNUE.h"
#include <memory>
#include <string>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include "Syzygy.h"

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <cstdio>
#endif
#ifdef _MSC_VER
#include <intrin.h>  // FIX 13.9: __cpuid / __cpuidex intrinsics
#endif

// ============================================================
//  Pipe I/O initialization for Windows GUI-subsystem builds
//
//  When ChessEngine.exe is built with /SUBSYSTEM:WINDOWS, the
//  CRT may not automatically wire up stdin/stdout to the pipe
//  handles inherited from a parent process (e.g. cutechess-cli).
//  This function detects inherited pipe handles and connects
//  the C and C++ standard streams to them so UCI works normally.
// ============================================================
#ifdef _WIN32
static void initPipeIO() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
    HANDLE hIn  = GetStdHandle(STD_INPUT_HANDLE);

    // Wire stdout if a valid handle exists (e.g. pipe from TrainingRunner).
    // NOTE: Do NOT gate stdout/stderr on stdin — the parent process may only
    // redirect output without providing an input handle.
    if (hOut != NULL && hOut != INVALID_HANDLE_VALUE) {
        int fdOut = _open_osfhandle(reinterpret_cast<intptr_t>(hOut), _O_WRONLY);
        if (fdOut != -1) {
            FILE* fpOut = _fdopen(fdOut, "w");
            if (fpOut) {
                // FIX 3.19: Use _dup2 instead of *stdout = *fpOut which is
                // implementation-defined (assumes FILE is trivially copyable).
                _dup2(_fileno(fpOut), _fileno(stdout));
                setvbuf(stdout, NULL, _IONBF, 0);
            } else {
                _close(fdOut);  // FIX 13.13: prevent fd leak when _fdopen fails
            }
        }
    }

    if (hErr != NULL && hErr != INVALID_HANDLE_VALUE) {
        int fdErr = _open_osfhandle(reinterpret_cast<intptr_t>(hErr), _O_WRONLY);
        if (fdErr != -1) {
            FILE* fpErr = _fdopen(fdErr, "w");
            if (fpErr) {
                _dup2(_fileno(fpErr), _fileno(stderr));
                setvbuf(stderr, NULL, _IONBF, 0);
            } else {
                _close(fdErr);  // FIX 13.13: prevent fd leak when _fdopen fails
            }
        }
    }

    // Wire stdin only if available (not provided by --generate launches)
    if (hIn != NULL && hIn != INVALID_HANDLE_VALUE) {
        int fdIn = _open_osfhandle(reinterpret_cast<intptr_t>(hIn), _O_RDONLY);
        if (fdIn != -1) {
            FILE* fpIn = _fdopen(fdIn, "r");
            if (fpIn) {
                _dup2(_fileno(fpIn), _fileno(stdin));
                setvbuf(stdin, NULL, _IONBF, 0);
            }
        }
    }

    // Sync C++ streams with the re-wired C streams
    std::ios::sync_with_stdio(true);
    std::cin.clear();
    std::cout.clear();
    std::cerr.clear();
}
#endif

// ============================================================
//  AUDIT FIX 4: Runtime CPU feature detection
//  The NNUE inference uses AVX2 + FMA intrinsics unconditionally.
//  Crash with SIGILL on CPUs that lack these features.
//  This check runs at startup and prints a clear error message.
// ============================================================
static bool checkCPUFeatures() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _MSC_VER
        int cpuInfo[4] = {};
        __cpuid(cpuInfo, 0);
        if (cpuInfo[0] < 7) return false;
        __cpuidex(cpuInfo, 7, 0);
        bool hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;
        __cpuid(cpuInfo, 1);
        bool hasFMA  = (cpuInfo[2] & (1 << 12)) != 0;
        bool hasSSE41 = (cpuInfo[2] & (1 << 19)) != 0;
    #else
        // GCC / Clang
        unsigned int eax, ebx, ecx, edx;
        __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(7), "c"(0));
        bool hasAVX2 = (ebx & (1 << 5)) != 0;
        __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1), "c"(0));
        bool hasFMA   = (ecx & (1 << 12)) != 0;
        bool hasSSE41 = (ecx & (1 << 19)) != 0;
    #endif
    if (!hasSSE41 || !hasAVX2 || !hasFMA) {
        std::cerr << "FATAL: This build requires SSE4.1 + AVX2 + FMA instructions.\n"
                  << "  SSE4.1: " << (hasSSE41 ? "OK" : "MISSING") << "\n"
                  << "  AVX2:   " << (hasAVX2  ? "OK" : "MISSING") << "\n"
                  << "  FMA:    " << (hasFMA   ? "OK" : "MISSING") << "\n"
                  << "Your CPU does not support all required features. Exiting.\n";
        return false;
    }
    return true;
#else
    // Non-x86 (ARM, etc.) — SIMD intrinsics will not work
    std::cerr << "FATAL: This build uses x86 SIMD intrinsics (SSE4.1/AVX2/FMA).\n"
              << "Running on a non-x86 CPU is not supported. Exiting.\n";
    return false;
#endif
}

// Helper: check whether argv contains a flag string
static bool hasFlag(int argc, char* argv[], const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

// Helper: return value after a flag, or nullptr if not found / no value
// FIX 13.5: Handle --flag=value syntax and edge case where flag is last arg
// Note: if the flag is the last argument (no value follows), prints a warning and
// returns nullptr. This is intentional — the flag requires a value argument.
static const char* flagValue(int argc, char* argv[], const char* flag) {
    size_t flagLen = std::strlen(flag);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            if (i + 1 < argc) return argv[i + 1];
            std::cerr << "Warning: " << flag << " requires a value" << std::endl;
            return nullptr;
        }
        // Support --flag=value syntax
        if (std::strncmp(argv[i], flag, flagLen) == 0 && argv[i][flagLen] == '=') {
            return argv[i] + flagLen + 1;
        }
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    // AUDIT FIX 4: Abort early if CPU lacks required SIMD features
    if (!checkCPUFeatures()) return 1;

    // Initialize bitboard attack tables (must be called before any board operations)
    BB::initBitboards();

    // =========================================================
    //  --generate mode
    //
    //  Native C++ self-play generator. Replaces generate_selfplay.py.
    //  Eliminates all Python/pipe overhead for 5-10x speed improvement.
    //
    //  Usage:
    //    ChessEngine.exe --generate [options]
    //
    //  Options:
    //    --games        N      number of games                    (default: 1000)
    //    --depth        N      search depth per move              (default: 5)
    //    --workers      N      parallel threads                   (default: 8)
    //    --output       path   output binary file path            (default: assets/selfplay.bin)
    //    --maxplies     N      max half-moves per game            (default: 300)
    //    --resign-cp    N      resign threshold in centipawns     (default: 500)
    //    --resign-count N      consecutive plies above threshold  (default: 3)
    //    --draw-cp      N      draw threshold in centipawns       (default: 15)
    //    --draw-count   N      consecutive plies below threshold  (default: 5)
    //    --draw-min-ply N      earliest ply for draw adjudication (default: 30)
    //    --draw-adj-moves N    plies of near-zero eval for dead-draw adj (default: 10)
    //    --draw-adj-threshold N  centipawn threshold for dead equal (default: 5)
    //    --draw-adj-min-move N   min move number for dead-draw adj (default: 40)
    //    --opening-plies N     random moves at start for variety  (default: 4)
    //    --weights      path   NNUE weights file path             (default: assets/nnue_weights.bin)
    //    --openings     path   opening book file (one FEN/line)   (default: none)
    //    --help                show this help message
    //
    //  Example (PowerShell overnight pipeline):
    //    .\x64\Release\ChessEngine.exe --generate --games 15000 --depth 5 --workers 12 --output assets\selfplay_gen1.bin
    //    .\x64\Release\ChessEngine.exe --generate --games 5000 --depth 6 --workers 12 --openings assets\openings.txt --output assets\selfplay_gen2.bin
    // =========================================================
    if (hasFlag(argc, argv, "--generate")) {

#ifdef _WIN32
        initPipeIO();   // ensure stdout works for progress output
#endif

        // --- Help -------------------------------------------------------
        if (hasFlag(argc, argv, "--help")) {
            std::cout <<
                "Usage: ChessEngine.exe --generate [options]\n\n"
                "Options:\n"
                "  --games        N      number of games                    (default: 1000)\n"
                "  --depth        N      search depth per move              (default: 5)\n"
                "  --movetime     N      milliseconds per move (overrides depth)  (default: 0=use depth)\n"
                "  --workers      N      parallel threads                   (default: 8)\n"
                "  --output       path   output binary file path            (default: assets/selfplay.bin)\n"
                "  --maxplies     N      max half-moves per game            (default: 300)\n"
                "  --contempt     N      contempt in centipawns (draw aversion) (default: 25)\n"
                "  --resign-cp    N      resign threshold in centipawns     (default: 500)\n"
                "  --resign-count N      consecutive plies above threshold  (default: 3)\n"
                "  --draw-cp      N      draw threshold in centipawns       (default: 15)\n"
                "  --draw-count   N      consecutive plies below threshold  (default: 5)\n"
                "  --draw-min-ply N      earliest ply for draw adjudication (default: 30)\n"
                "  --draw-adj-moves N    plies of near-zero eval for dead-draw adj (default: 10)\n"
                "  --draw-adj-threshold N  centipawn threshold for dead equal (default: 5)\n"
                "  --draw-adj-min-move N   min move number for dead-draw adj (default: 40)\n"
                "  --opening-plies N     random moves at start for variety  (default: 4)\n"
                "  --opening-temp  F     softmax temperature for opening     (default: 1.5)\n"
                "  --softmax-plies N     post-opening softmax plies          (default: 8)\n"
                "  --softmax-temp  F     softmax temperature for post-opening (default: 0.5)\n"
                "  --root-noise    F     prob of random move replacement      (default: 0.0)\n"
                "  --record-min-ply N    don't record positions before ply N  (default: 10)\n"
                "  --record-max-eval N   skip positions with |eval| > N       (default: 2500)\n"
                "  --weights      path   NNUE weights file path             (default: assets/nnue_weights.bin)\n"
                "  --openings     path   opening book file (one FEN/line)   (default: none)\n"
                "  --frc-mix      F      fraction of games using Chess960 starts  (default: 0.0)\n"
                "  --help                show this help message\n";
            return 0;
        }

        Engine::initZobrist();

        // --- Parse all config flags -------------------------------------
        SelfPlayGen::Config cfg;

        // FIX 13.6: Return -1 on parse failure instead of 0 (which causes invalid configs)
        auto safeStoi = [](const char* v, const char* name) -> int {
            try {
                size_t pos = 0;
                int result = std::stoi(v, &pos);
                if (pos != std::strlen(v)) {
                    std::cerr << "Warning: trailing characters in " << name << ": " << v << std::endl;
                }
                return result;
            }
            catch (const std::exception&) {
                std::cerr << "Error: invalid value for " << name << ": " << v << std::endl;
                return -1;
            }
        };

        if (const char* v = flagValue(argc, argv, "--games"))         cfg.games        = safeStoi(v, "--games");
        if (const char* v = flagValue(argc, argv, "--depth"))         cfg.searchDepth = safeStoi(v, "--depth");
        if (const char* v = flagValue(argc, argv, "--movetime"))      cfg.moveTimeMs  = safeStoi(v, "--movetime");
        if (const char* v = flagValue(argc, argv, "--workers"))       cfg.workers      = safeStoi(v, "--workers");
        if (const char* v = flagValue(argc, argv, "--output"))        cfg.outputPath   = v;
        if (const char* v = flagValue(argc, argv, "--maxplies"))      cfg.maxPlies     = safeStoi(v, "--maxplies");
        if (const char* v = flagValue(argc, argv, "--contempt"))       cfg.contemptCp   = safeStoi(v, "--contempt");
        if (const char* v = flagValue(argc, argv, "--resign-cp"))     cfg.resignCp     = safeStoi(v, "--resign-cp");
        if (const char* v = flagValue(argc, argv, "--resign-count"))  cfg.resignCount  = safeStoi(v, "--resign-count");
        if (const char* v = flagValue(argc, argv, "--draw-cp"))       cfg.drawCp       = safeStoi(v, "--draw-cp");
        if (const char* v = flagValue(argc, argv, "--draw-count"))    cfg.drawCount    = safeStoi(v, "--draw-count");
        if (const char* v = flagValue(argc, argv, "--draw-min-ply"))  cfg.drawMinPly   = safeStoi(v, "--draw-min-ply");
        if (const char* v = flagValue(argc, argv, "--draw-adj-moves"))     cfg.drawAdjMoves     = safeStoi(v, "--draw-adj-moves");
        if (const char* v = flagValue(argc, argv, "--draw-adj-threshold")) cfg.drawAdjThreshold = safeStoi(v, "--draw-adj-threshold");
        if (const char* v = flagValue(argc, argv, "--draw-adj-min-move"))  cfg.drawAdjMinMove   = safeStoi(v, "--draw-adj-min-move");
        if (const char* v = flagValue(argc, argv, "--game-time-limit")) cfg.gameTimeLimitSec = safeStoi(v, "--game-time-limit");
        if (const char* v = flagValue(argc, argv, "--opening-plies")) cfg.openingPlies = safeStoi(v, "--opening-plies");
        if (const char* v = flagValue(argc, argv, "--opening-temp")) {
            try { cfg.openingTemp = std::stof(v); }
            catch (...) { std::cerr << "Warning: invalid value for --opening-temp: " << v << std::endl; }
        }
        if (const char* v = flagValue(argc, argv, "--softmax-plies")) cfg.softmaxPlies = safeStoi(v, "--softmax-plies");
        if (const char* v = flagValue(argc, argv, "--softmax-temp")) {
            try { cfg.softmaxTemp = std::stof(v); }
            catch (...) { std::cerr << "Warning: invalid value for --softmax-temp: " << v << std::endl; }
        }
        if (const char* v = flagValue(argc, argv, "--root-noise")) {
            try { cfg.rootNoiseEps = std::stof(v); }
            catch (...) { std::cerr << "Warning: invalid value for --root-noise: " << v << std::endl; }
        }
        if (const char* v = flagValue(argc, argv, "--record-min-ply")) cfg.recordMinPly = safeStoi(v, "--record-min-ply");
        if (const char* v = flagValue(argc, argv, "--record-max-eval")) cfg.recordMaxEval = safeStoi(v, "--record-max-eval");
        if (const char* v = flagValue(argc, argv, "--nps-samples"))   cfg.npsSamples   = safeStoi(v, "--nps-samples");
        if (const char* v = flagValue(argc, argv, "--openings"))      cfg.openingsFile = v;
        if (const char* v = flagValue(argc, argv, "--frc-mix")) {
            try { cfg.frcMix = std::stof(v); }
            catch (...) { std::cerr << "Warning: invalid value for --frc-mix: " << v << std::endl; }
        }
        if (const char* v = flagValue(argc, argv, "--mixed-depth-ratio")) {
            try { cfg.mixedDepthRatio = std::stof(v); }
            catch (...) { std::cerr << "Warning: invalid value for --mixed-depth-ratio: " << v << std::endl; }
        }
        if (const char* v = flagValue(argc, argv, "--mixed-depth-low"))
            cfg.mixedDepthLow = safeStoi(v, "--mixed-depth-low");
        if (flagValue(argc, argv, "--depth-shuffle"))
            cfg.depthShuffle = true;
        if (const char* v = flagValue(argc, argv, "--depth-shuffle-bias")) {
            try { cfg.depthShuffleBias = std::stof(v); }
            catch (...) { std::cerr << "Warning: invalid value for --depth-shuffle-bias: " << v << std::endl; }
        }

        const char* weightsArg = flagValue(argc, argv, "--weights");
        static std::string defaultWeights = assetPath("assets/nnue_weights.bin");
        const char* weightsPath = weightsArg ? weightsArg : defaultWeights.c_str();

        // Duck chess: set isDuckChess on all boards in self-play
        bool isDuckChess = hasFlag(argc, argv, "--duck-chess");

        // FIX: Allocate on the heap - NNUE::Network is ~2 MB and exceeds the
        //      default 1 MB Windows stack, causing a silent crash at startup.
        auto nnue = std::make_unique<NNUE::Network>();
        bool nnueLoaded = false;

        // Duck chess: load DuckNNUE weights if available
        std::unique_ptr<DuckNNUE::Network> duckNnue;
        if (isDuckChess) {
            static std::string duckWeightsPath = assetPath("assets/duck_nnue_weights.bin");
            const char* duckWeightsArg = flagValue(argc, argv, "--weights");
            const char* duckPath = duckWeightsArg ? duckWeightsArg : duckWeightsPath.c_str();
            duckNnue = std::make_unique<DuckNNUE::Network>();
            if (duckNnue->loadWeights(duckPath)) {
                std::cerr << "[SelfPlay] Duck Chess mode enabled — loaded DuckNNUE from " << duckPath << "\n";
            } else {
                std::cerr << "[SelfPlay] Duck Chess mode enabled — DuckNNUE weights not found, using handcrafted eval\n";
                duckNnue.reset();  // don't pass invalid weights
            }
        } else {
            // Standard chess: try to load NNUE weights
            nnueLoaded = nnue->loadWeights(weightsPath);
            if (!nnueLoaded) {
                std::cerr << "[SelfPlay] WARNING: Could not load " << weightsPath
                          << " - falling back to handcrafted evaluation.\n";
            }
            if (nnueLoaded) {
                nnue->releaseFloatWeights();  // free ~160 MB — evaluateQ() uses quantized weights
            }
        }

        // Auto-detect Syzygy tablebases next to the exe
        if (Syzygy::maxPieces() == 0) {
            std::string tbPath = assetPath("Syzygy345");
            if (std::filesystem::is_directory(tbPath)) {
                if (Syzygy::init(tbPath)) {
                    std::cerr << "[SelfPlay] Syzygy tablebases auto-loaded ("
                              << Syzygy::maxPieces() << "-piece) from: "
                              << tbPath << std::endl;
                }
            }
        }

        cfg.isDuckChess = isDuckChess;
        SelfPlayGen gen(nnue.get(), duckNnue.get());
        int result = gen.generate(cfg);

        Syzygy::free();
        return (result >= 0) ? 0 : 1;
    }

    // =========================================================
    //  --train-duck mode
    //
    //  Trains the DuckNNUE (832-feature) network using C++ trainer.
    //  Called by TrainingRunner for Duck Chess variant.
    //
    //  Usage:
    //    ChessEngine.exe --train-duck --data path.bin [--extra-data path.bin ratio]
    //                    --output weights.bin --epochs N --lr F
    // =========================================================
    if (hasFlag(argc, argv, "--train-duck")) {
#ifdef _WIN32
        initPipeIO();
#endif
        BB::initBitboards();
        Engine::initZobrist();

        const char* outputPath = flagValue(argc, argv, "--output");
        if (!outputPath) {
            std::cerr << "[TrainDuck] ERROR: --output required\n";
            return 1;
        }

        NNUE::TrainingConfig tcfg;
        tcfg.isDuckChess = true;
        if (const char* v = flagValue(argc, argv, "--epochs"))       tcfg.epochs            = std::stoi(v);
        if (const char* v = flagValue(argc, argv, "--batch-size"))   tcfg.batchSize         = std::stoi(v);
        if (const char* v = flagValue(argc, argv, "--lr"))           tcfg.learningRate      = std::stof(v);
        if (const char* v = flagValue(argc, argv, "--early-stop"))   tcfg.earlyStopPatience = std::stoi(v);
        if (const char* v = flagValue(argc, argv, "--lr-decay"))     tcfg.lrDecay           = std::stof(v);
        if (const char* v = flagValue(argc, argv, "--weight-decay")) tcfg.weightDecay       = std::stof(v);
        if (const char* v = flagValue(argc, argv, "--label-smoothing")) tcfg.labelSmoothing = std::stof(v);
        if (const char* v = flagValue(argc, argv, "--grad-accum"))   tcfg.gradAccum         = std::stoi(v);
        if (const char* v = flagValue(argc, argv, "--warmup-steps")) tcfg.warmupSteps       = std::stoi(v);
        if (hasFlag(argc, argv, "--cosine-lr"))                      tcfg.cosineLr          = true;
        if (const char* v = flagValue(argc, argv, "--cosine-t0"))    tcfg.cosineT0          = std::stoi(v);
        if (hasFlag(argc, argv, "--swa")) {
            tcfg.swa = true;
            if (const char* v = flagValue(argc, argv, "--swa-start")) tcfg.swaStart = std::stoi(v);
        }
        // draw-weight: fraction of loss from game result (1-lambda = draw weight, lambda = eval weight)
        if (const char* v = flagValue(argc, argv, "--draw-weight"))  tcfg.lambda = 1.0f - std::stof(v);
        if (const char* v = flagValue(argc, argv, "--mate-boost"))   tcfg.mateBoost     = std::stof(v);
        if (const char* v = flagValue(argc, argv, "--max-positions")) tcfg.maxPositions  = std::stoi(v);
        tcfg.outputPath = outputPath;

        // Collect all --data and --extra-data files
        std::vector<std::string> dataFiles;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--data") == 0 && i+1 < argc)
                dataFiles.push_back(argv[++i]);
            else if (std::strcmp(argv[i], "--extra-data") == 0 && i+1 < argc) {
                dataFiles.push_back(argv[++i]);
                if (i+1 < argc && argv[i+1][0] != '-') ++i; // skip ratio arg
            }
        }

        if (dataFiles.empty()) {
            std::cerr << "[TrainDuck] ERROR: no --data files specified\n";
            return 1;
        }

        // Load and merge all training data
        NNUE::Trainer trainer;
        std::vector<NNUE::TrainingPosition> allData;
        for (const auto& f : dataFiles) {
            if (!std::filesystem::exists(f)) {
                std::cerr << "[TrainDuck] WARNING: data file not found: " << f << "\n";
                continue;
            }
            auto d = trainer.loadTrainingData(f);
            std::cerr << "[TrainDuck] Loaded " << d.size() << " positions from " << f << "\n";
            allData.insert(allData.end(), d.begin(), d.end());
        }

        if (allData.empty()) {
            std::cerr << "[TrainDuck] ERROR: no training data loaded\n";
            return 1;
        }
        std::cerr << "[TrainDuck] Total positions: " << allData.size() << "\n";

        // Color-swap mirror: double the dataset by adding a vertically-mirrored
        // copy of every position with colors swapped. Balances white/black
        // perspective coverage regardless of W/D/B ratio in self-play.
        allData = NNUE::Trainer::colorSwapMirrorData(allData);
        std::cerr << "[TrainDuck] After color-swap mirror: " << allData.size() << " positions\n";

        // Load or create DuckNNUE network
        auto net = std::make_unique<DuckNNUE::Network>();
        const char* loadWeights = flagValue(argc, argv, "--load-weights");
        if (loadWeights && std::filesystem::exists(loadWeights)) {
            net->loadWeights(loadWeights);
            std::cerr << "[TrainDuck] Loaded weights from " << loadWeights << "\n";
        } else {
            net->randomizeWeights();
            std::cerr << "[TrainDuck] Starting with random weights\n";
        }

        // Split data 90/10 train/val for validation loss tracking
        std::mt19937 splitRng(42);
        std::shuffle(allData.begin(), allData.end(), splitRng);
        size_t valSize = std::max(size_t(1), allData.size() / 10);
        std::vector<NNUE::TrainingPosition> valData(allData.end() - valSize, allData.end());
        allData.resize(allData.size() - valSize);
        std::cerr << "[TrainDuck] Train: " << allData.size() << "  Val: " << valData.size() << "\n";

        // Track LR across epochs (mirrors trainDuck's decay)
        float currentLr = tcfg.learningRate;

        // Timing state for epoch ETA
        using Clock = std::chrono::steady_clock;
        auto epochStart = Clock::now();
        double lastEpochSecs = 0.0;

        trainer.trainDuck(*net, allData, tcfg,
            [&](int ep, float loss) {
                // Compute epoch duration and ETA
                auto now = Clock::now();
                lastEpochSecs = std::chrono::duration<double>(now - epochStart).count();
                epochStart = now;
                int remaining = tcfg.epochs - ep;
                double etaSecs = lastEpochSecs * remaining;
                int etaMin = (int)(etaSecs / 60);
                int etaSec = (int)(etaSecs) % 60;

                auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
                auto screlu  = [](float x) { float c = x < 0.f ? 0.f : x > 1.f ? 1.f : x; return c*c; };

                float valLoss = 0.0f;
                float opLoss = 0.0f, mgLoss = 0.0f, egLoss = 0.0f;
                int opCnt = 0, mgCnt = 0, egCnt = 0;
                int correct = 0;

                for (const auto& pos : valData) {
                    // Forward pass
                    std::array<float, DuckNNUE::L1_SIZE> wAcc, bAcc;
                    for (int j = 0; j < DuckNNUE::L1_SIZE; ++j) { wAcc[j] = bAcc[j] = net->L1_biases[j]; }
                    for (int feat : pos.activeFeatures) {
                        int mir = DuckNNUE::mirrorDuckFeature(feat);
                        for (int j = 0; j < DuckNNUE::L1_SIZE; ++j) {
                            wAcc[j] += net->L1_weights[feat][j];
                            bAcc[j] += net->L1_weights[mir][j];
                        }
                    }
                    const auto& stm = (pos.sideToMove == Color::White) ? wAcc : bAcc;
                    const auto& opp = (pos.sideToMove == Color::White) ? bAcc : wAcc;
                    std::array<float, DuckNNUE::L1_SIZE*2> l1;
                    for (int i = 0; i < DuckNNUE::L1_SIZE; ++i) { l1[i] = screlu(stm[i]); l1[DuckNNUE::L1_SIZE+i] = screlu(opp[i]); }
                    std::array<float, DuckNNUE::L2_SIZE> l2;
                    for (int j = 0; j < DuckNNUE::L2_SIZE; ++j) {
                        float s = net->L2_biases[j];
                        for (int i = 0; i < DuckNNUE::L1_SIZE*2; ++i) s += l1[i] * net->L2_weights[i][j];
                        l2[j] = screlu(s);
                    }
                    std::array<float, DuckNNUE::L3_SIZE> l3;
                    for (int j = 0; j < DuckNNUE::L3_SIZE; ++j) {
                        float s = net->L3_biases[j];
                        for (int i = 0; i < DuckNNUE::L2_SIZE; ++i) s += l2[i] * net->L3_weights[i][j];
                        l3[j] = screlu(s);
                    }
                    float raw = net->output_bias;
                    for (int i = 0; i < DuckNNUE::L3_SIZE; ++i) raw += l3[i] * net->output_weights[i];
                    float whitePred = (pos.sideToMove == Color::White) ? raw * 400.f : -raw * 400.f;

                    float sp = sigmoid(whitePred / tcfg.evalScale);
                    float st = sigmoid(pos.searchEval / tcfg.evalScale);
                    float el = (sp - st) * (sp - st);
                    float rl = (sp - pos.gameResult) * (sp - pos.gameResult);
                    float posLoss = tcfg.lambda * el + (1.f - tcfg.lambda) * rl;
                    valLoss += posLoss;

                    // Accuracy: sign of predicted cp vs sign of target eval
                    // Only count non-zero targets to avoid noise near zero
                    if (std::abs(pos.searchEval) > 10.f) {
                        bool predWinning = whitePred > 0.f;
                        bool targetWinning = pos.searchEval > 0.f;
                        if (predWinning == targetWinning) ++correct;
                    }

                    // Phase loss
                    NNUE::Trainer::GamePhase phase = NNUE::Trainer::classifyPhase(pos.activeFeatures);
                    switch (phase) {
                        case NNUE::Trainer::GamePhase::Opening:    opLoss += posLoss; ++opCnt; break;
                        case NNUE::Trainer::GamePhase::Middlegame: mgLoss += posLoss; ++mgCnt; break;
                        case NNUE::Trainer::GamePhase::Endgame:    egLoss += posLoss; ++egCnt; break;
                    }
                }

                int valN = static_cast<int>(valData.size());
                valLoss /= static_cast<float>(valN);
                float accuracy = (valN > 0) ? static_cast<float>(correct) / static_cast<float>(valN) : 0.f;
                if (opCnt > 0) opLoss /= static_cast<float>(opCnt);
                if (mgCnt > 0) mgLoss /= static_cast<float>(mgCnt);
                if (egCnt > 0) egLoss /= static_cast<float>(egCnt);

                std::cout << "Epoch " << ep << "/" << tcfg.epochs
                          << " loss=" << std::fixed << std::setprecision(8) << loss
                          << " val_loss=" << std::fixed << std::setprecision(8) << valLoss
                          << " acc=" << std::fixed << std::setprecision(4) << accuracy
                          << " op=" << std::fixed << std::setprecision(8) << opLoss
                          << " mg=" << std::fixed << std::setprecision(8) << mgLoss
                          << " eg=" << std::fixed << std::setprecision(8) << egLoss
                          << " lr=" << std::fixed << std::setprecision(8) << currentLr
                          << " epoch_time=" << std::fixed << std::setprecision(1) << lastEpochSecs << "s"
                          << " eta=" << etaMin << "m" << etaSec << "s"
                          << "\n";
                std::cout.flush();
                currentLr *= tcfg.lrDecay;
            },
            nullptr,  // cancelFlag passed separately below
            [&](int batch, int totalBatches, float batchLoss) {
                // Overwrite the same line with batch progress
                // Format: "  batch 12/97 loss=0.01234  ETA 1m23s"
                double elapsed = std::chrono::duration<double>(Clock::now() - epochStart).count();
                double batchesPerSec = (elapsed > 0.0) ? batch / elapsed : 0.0;
                double etaBatch = (batchesPerSec > 0.0) ? (totalBatches - batch) / batchesPerSec : 0.0;
                int etaMin2 = (int)(etaBatch / 60);
                int etaSec2 = (int)(etaBatch) % 60;
                char buf[128];
                std::snprintf(buf, sizeof(buf),
                    "\r  batch %d/%d  loss=%.5f  %.1f b/s  ETA %dm%02ds   ",
                    batch, totalBatches, batchLoss, batchesPerSec, etaMin2, etaSec2);
                std::cout << buf;
                std::cout.flush();
            });

        net->saveWeights(outputPath);
        std::cerr << "[TrainDuck] Saved weights to " << outputPath << "\n";
        return 0;
    }

    // =========================================================
    //  UCI mode
    //
    //  Launch when:
    //    1. --uci flag is passed, OR
    // NOTE [3.25]: A previous comment described stdin auto-detection but it was
    // never implemented. Only --uci flag works. TODO: implement stdin auto-detect.
    // =========================================================
    bool uciMode = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--uci") == 0) {
            uciMode = true;
            break;
        }
    }

    if (uciMode) {
#ifdef _WIN32
        initPipeIO();   // wire up stdin/stdout for GUI-subsystem pipe I/O
#endif
        auto uci = std::make_unique<UCI>();
        uci->loop();
        return 0;
    }

    // =========================================================
    //  --help mode
    // =========================================================
    if (hasFlag(argc, argv, "--help")) {
#ifdef _WIN32
        initPipeIO();
#endif
        std::cout << "Usage: ChessEngine [options]\n\n"
                  << "Modes:\n"
                  << "  --uci            Launch in UCI protocol mode\n"
                  << "  --generate       Self-play data generation mode\n"
                  << "  (no mode flag)   Visual/GUI mode (requires display)\n\n"
                  << "Use --generate --help for self-play generation options.\n";
        return 0;
    }

    // =========================================================
    //  Default: visual/GUI mode (original behavior)
    // =========================================================
    if (argc > 1) {
        std::cerr << "Warning: unrecognized flags. Use --uci, --generate, or --help." << std::endl;
    }
    auto game = std::make_unique<VisualGame>();
    game->run();
    return 0;
}
