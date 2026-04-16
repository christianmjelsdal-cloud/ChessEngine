// TR_Config.cpp  --  Configuration, preset management, calibration
#include "TR_Types.h"
#include "TR_Globals.h"
#include "TR_Fwd.h"

// Scan assets dir for highest nnue_weights_genN.bin → return N (or 0)
int findLatestGen(const std::string& dataDir) {
    fs::path dir = fs::path(exeDir()) / dataDir;
    int best = 0;
    if (!fs::is_directory(dir)) return 0;
    for (auto& ent : fs::directory_iterator(dir)) {
        std::string name = ent.path().filename().string();
        // match "nnue_weights_gen<N>.bin"
        if (name.rfind("nnue_weights_gen", 0) == 0 && name.size() > 20) {
            size_t numStart = 16; // length of "nnue_weights_gen"
            size_t dot = name.find('.', numStart);
            if (dot != std::string::npos) {
                std::string numStr = name.substr(numStart, dot - numStart);
                try { int g = std::stoi(numStr); if (g <= 0) continue;  // FIX 6.3: skip non-positive
                      if (g > best) best = g; }
                catch (...) {}
            }
        }
    }
    return best;
}


// ── Persistent gen stats (survives app restart) ──────────────────
// File format: one line per gen, "gen,best_val_loss\n"
// FIX 6.9: Note — this calls ReadConfig() which reads from UI controls.
// It must not be called from within ReadConfig() to avoid recursion.
static std::string genStatsPath(const std::string& dataDir) {
    // BUG FIX: Accept dataDir directly to avoid coupling with ReadConfig()
    return exeDir() + "\\" + dataDir + "\\gen_stats.csv";
}

// Append or update a gen's best val_loss in the persistent file.
void saveGenStat(int gen, double bestValLoss, const std::string& dataDir) {
    std::string path = genStatsPath(dataDir);
    // Read existing entries
    std::map<int, double> stats;
    {   std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            auto c = line.find(',');
            if (c == std::string::npos) continue;
            try {
                int g = std::stoi(line.substr(0, c));
                if (g <= 0) continue;  // FIX 6.3: skip non-positive gen numbers
                double v = std::stod(line.substr(c + 1));
                stats[g] = v;
            } catch (...) {}
        }
    }
    // Update this gen (keep lowest)
    auto it = stats.find(gen);
    if (it == stats.end() || bestValLoss < it->second)
        stats[gen] = bestValLoss;
    // Rewrite atomically via temp file  (FIX 6.10)
    std::string tmpPath = path + ".tmp";
    {
        std::ofstream out(tmpPath, std::ios::trunc);
        for (auto& [g, v] : stats)
            out << g << "," << std::fixed << std::setprecision(8) << v << "\n";
    }  // out closes here, flushing all data
    // Atomic rename: replace real file with temp  (FIX 6.10, BUG-1 fix)
    // Windows rename() fails if destination exists; use platform-appropriate replacement
#ifdef _WIN32
    // MoveFileExW with MOVEFILE_REPLACE_EXISTING handles the existing-dest case
    MoveFileExW(
        fs::path(tmpPath).wstring().c_str(),
        fs::path(path).wstring().c_str(),
        MOVEFILE_REPLACE_EXISTING);
#else
    std::rename(tmpPath.c_str(), path.c_str());
#endif
}

// Read the best gen from the persistent file.  Returns 0 if none.
static int loadBestGenFromFile(const std::string& dataDir) {
    std::string path = genStatsPath(dataDir);
    std::ifstream in(path);
    if (!in) return 0;
    int bestG = 0; double bestV = 1e9;
    std::string line;
    while (std::getline(in, line)) {
        auto c = line.find(',');
        if (c == std::string::npos) continue;
        try {
            int g = std::stoi(line.substr(0, c));
            if (g <= 0) continue;  // FIX 6.3: skip non-positive gen numbers
            double v = std::stod(line.substr(c + 1));
            if (v < bestV) { bestV = v; bestG = g; }
        } catch (...) {}
    }
    return bestG;
}

// ── Graph data persistence ──────────────────────────────────────

static std::string graphDataPath(const std::string& dataDir) {
    return exeDir() + "\\training progress\\training_graph.csv";
}

// Load all graph points from the persistent CSV.
// Format: gen,epoch,train,val,lr,acc,openLoss,midLoss,endLoss
std::vector<TrainPoint> loadGraphData(const std::string& dataDir) {
    std::vector<TrainPoint> result;
    std::string path = graphDataPath(dataDir);
    std::ifstream in(path);
    if (!in) return result;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        TrainPoint p;
        std::istringstream ss(line);
        std::string tok;
        try {
            if (!std::getline(ss, tok, ',')) continue; p.gen = std::stoi(tok);
            if (!std::getline(ss, tok, ',')) continue; p.step = std::stoi(tok);
            if (!std::getline(ss, tok, ',')) continue; p.train = std::stod(tok);
            if (std::getline(ss, tok, ',')) {
                if (tok != "nan") { p.val = std::stod(tok); p.hasVal = true; }
            }
            if (std::getline(ss, tok, ',')) {
                if (tok != "nan") { p.lr = std::stod(tok); p.hasLR = true; }
            }
            if (std::getline(ss, tok, ',')) {
                if (tok != "nan") { p.accuracy = std::stod(tok); p.hasAcc = true; }
            }
            if (std::getline(ss, tok, ',')) {
                std::string midTok, endTok;
                if (tok != "nan" && std::getline(ss, midTok, ',') && std::getline(ss, endTok, ',')) {
                    p.openingLoss = std::stod(tok);
                    p.middlegameLoss = std::stod(midTok);
                    p.endgameLoss = std::stod(endTok);
                    p.hasPhase = true;
                }
            }
            // NPS column (optional — older CSVs won't have it)
            if (std::getline(ss, tok, ',')) {
                if (tok != "nan" && !tok.empty()) { p.nps = std::stod(tok); p.hasNps = true; }
            }
            // NPS-only points have train=0 — mark hasLoss=false so loss panel skips them
            if (p.train == 0.0 && !p.hasVal) p.hasLoss = false;
        } catch (...) { continue; }
        result.push_back(p);
    }
    return result;
}

// Save all graph points to the persistent CSV (atomic write via temp file).
void saveGraphData(const std::string& dataDir, const std::vector<TrainPoint>& pts) {
    std::string path = graphDataPath(dataDir);
    std::string tmpPath = path + ".tmp";
    {
        std::ofstream out(tmpPath, std::ios::trunc);
        out << "# gen,epoch,train,val,lr,acc,openLoss,midLoss,endLoss,nps\n";
        out << std::fixed;
        for (auto& p : pts) {
            out << p.gen << ',' << p.step << ','
                << std::setprecision(8) << p.train << ',';
            if (p.hasVal) out << std::setprecision(8) << p.val; else out << "nan";
            out << ',';
            if (p.hasLR) out << std::scientific << std::setprecision(6) << p.lr; else out << "nan";
            out << ',';
            out << std::fixed;
            if (p.hasAcc) out << std::setprecision(4) << p.accuracy; else out << "nan";
            out << ',';
            if (p.hasPhase) {
                out << std::setprecision(8) << p.openingLoss << ','
                    << p.middlegameLoss << ',' << p.endgameLoss;
            } else {
                out << "nan,nan,nan";
            }
            out << ',';
            if (p.hasNps) out << std::setprecision(1) << p.nps; else out << "nan";
            out << '\n';
        }
    }
#ifdef _WIN32
    MoveFileExW(
        fs::path(tmpPath).wstring().c_str(),
        fs::path(path).wstring().c_str(),
        MOVEFILE_REPLACE_EXISTING);
#else
    std::rename(tmpPath.c_str(), path.c_str());
#endif
}

// Append a single point to the persistent CSV (fast incremental write).
void appendGraphPoint(const std::string& dataDir, const TrainPoint& p) {
    std::string path = graphDataPath(dataDir);
    bool exists = fs::exists(path);
    std::ofstream out(path, std::ios::app);
    if (!exists) {
        out << "# gen,epoch,train,val,lr,acc,openLoss,midLoss,endLoss,nps\n";
    }
    out << std::fixed;
    out << p.gen << ',' << p.step << ','
        << std::setprecision(8) << p.train << ',';
    if (p.hasVal) out << std::setprecision(8) << p.val; else out << "nan";
    out << ',';
    if (p.hasLR) out << std::scientific << std::setprecision(6) << p.lr; else out << "nan";
    out << ',';
    out << std::fixed;
    if (p.hasAcc) out << std::setprecision(4) << p.accuracy; else out << "nan";
    out << ',';
    if (p.hasPhase) {
        out << std::setprecision(8) << p.openingLoss << ','
            << p.middlegameLoss << ',' << p.endgameLoss;
    } else {
        out << "nan,nan,nan";
    }
    out << ',';
    if (p.hasNps) out << std::setprecision(1) << p.nps; else out << "nan";
    out << '\n';
    out.flush();
}

// Scan g_st.pts for the generation with the lowest final validation loss.
// Groups points by gen, picks the last val_loss per gen (= best epoch saved),
// then returns the gen whose final val_loss is smallest.
// Merges persistent gen_stats.csv (full history) with live pts (in-progress gen).
int findBestGen() {
    return findBestGenFor(g_cfg.dataDir);
}
// Overload accepting explicit dataDir (used by button handler with current UI config).
int findBestGenFor(const std::string& dataDir) {
    // 1. Load persistent file data — covers ALL completed gens across all runs
    std::map<int, double> bestVal;
    {
        std::string path = genStatsPath(dataDir);
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            auto c = line.find(',');
            if (c == std::string::npos) continue;
            try {
                int g = std::stoi(line.substr(0, c));
                if (g <= 0) continue;
                double v = std::stod(line.substr(c + 1));
                auto it = bestVal.find(g);
                if (it == bestVal.end() || v < it->second)
                    bestVal[g] = v;
            } catch (...) {}
        }
    }
    // 2. Merge live pts — may include in-progress gen not yet saved to file
    {
        std::lock_guard<std::mutex> lk(g_st.mtx);
        for (auto& p : g_st.pts) {
            if (!p.hasVal) continue;
            auto it = bestVal.find(p.gen);
            if (it == bestVal.end() || p.val < it->second)
                bestVal[p.gen] = p.val;
        }
    }
    if (bestVal.empty()) return 0;
    int bestG = 0; double bestV = 1e9;
    for (auto& [g, v] : bestVal) {
        if (v < bestV) { bestV = v; bestG = g; }
    }
    return bestG;
}

// -- Preset management --
static std::string presetFilePath() {
    return exeDir() + "\\custom_presets.cfg";
}
static std::string defaultPresetFilePath() {
    return exeDir() + "\\default_presets.cfg";
}
static std::string calibrationFilePath() {
    return exeDir() + "\\calibration.cfg";
}

// -- Serialize a single preset to pipe-delimited string --
static std::string SerializePreset(const Preset& p) {
    std::ostringstream o;
    o << p.name << "|"
      << p.generations << "|" << p.gamesPerGen << "|" << p.epochsPerGen << "|"
      << p.batchSize << "|" << p.workers << "|" << p.depth << "|"
      << p.gradAccum << "|" << p.warmupSteps << "|"
      << dbl2s(p.lr,8) << "|" << dbl2s(p.weightDecay,8) << "|"
      << dbl2s(p.dropout,4) << "|" << dbl2s(p.labelSmooth,4) << "|"
      << dbl2s(p.drawWeight,4) << "|" << dbl2s(p.mateBoost,4) << "|"
      << dbl2s(p.splRatio,4) << "|" << 0 << "|"  // startGen placeholder for compat
      << (p.eloValidate?1:0) << "|" << p.eloGames << "|" << p.swaGames << "|" << (p.overfitDetect?1:0) << "|"
      << p.maxPositions << "|" << p.earlyStop << "|"
      << (p.cosineLr?1:0) << "|" << p.cosineT0 << "|"
      << (p.swa?1:0) << "|" << p.swaStart << "|"
      << dbl2s(p.drawPct,2) << "|" << dbl2s(p.frcMix,3) << "|" << p.replayWindow << "|" << dbl2s(p.replayDecay,2) << "|" << p.resignCp << "|" << p.contemptCp << "|" << p.maxPlies << "|" << p.drawCp
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
    auto next = [&]() -> std::string {
        std::string t;
        if (std::getline(ss, t, '|')) return t;
        return "";
    };
    p.name = next(); if (p.name.empty()) return p;
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
    p.drawWeight   = pDbl(next(), 0.5);
    p.mateBoost    = pDbl(next(), 3.0);
    p.splRatio     = pDbl(next(), 0.4);
    next();  // startGen placeholder (no longer in preset, kept for compat)
    p.eloValidate  = pInt(next(), 0) != 0;
    p.eloGames     = pInt(next(), 100);
    p.swaGames     = pInt(next(), 50);
    p.overfitDetect= pInt(next(), 1) != 0;
    p.maxPositions = pInt(next(), 300000);
    p.earlyStop    = pInt(next(), 10);
    p.cosineLr     = pInt(next(), 1) != 0;
    p.cosineT0     = pInt(next(), 0);
    p.swa          = pInt(next(), 1) != 0;
    p.swaStart     = pInt(next(), 3);
    p.drawPct      = pDbl(next(), 10.0);
    p.frcMix       = pDbl(next(), 0.0);
    p.replayWindow = pInt(next(), 3);
    p.replayDecay  = pDbl(next(), 0.7);
    p.resignCp   = pInt(next(), 500);
    p.contemptCp = pInt(next(), 25);
    p.maxPlies   = pInt(next(), 250);
    p.drawCp     = pInt(next(), 8);
    // v2 fields — appended for backward compatibility (old files simply get defaults)
    p.mixedDepthLow   = pInt(next(), 4);
    p.mixedDepthRatio = pDbl(next(), 0.0);  // FIX 6.33: no float cast
    p.wdlAlpha        = pDbl(next(), 0.5);
    p.wdlDrawElo      = pDbl(next(), 100.0);
    // v3 fields — depth shuffle (backward compat: old files get defaults)
    p.depthShuffle     = pInt(next(), 0) != 0;
    p.depthShuffleBias = pDbl(next(), 2.0);
    // v4 fields — self-play diversity, recording, adjudication (backward compat: old files get defaults)
    p.openingTemp      = pDbl(next(), 1.5);
    p.openingPlies     = pInt(next(), 4);
    p.softmaxPlies     = pInt(next(), 8);
    p.softmaxTemp      = pDbl(next(), 0.5);
    p.rootNoiseEps     = pDbl(next(), 0.0);
    p.recordMinPly     = pInt(next(), 10);
    p.recordMaxEval    = pInt(next(), 2500);
    p.resignCount      = pInt(next(), 3);
    p.drawCount        = pInt(next(), 6);
    p.drawMinPly       = pInt(next(), 40);
    p.drawAdjMoves     = pInt(next(), 12);
    p.drawAdjThreshold = pInt(next(), 4);
    p.drawAdjMinMove   = pInt(next(), 50);
    return p;
}

// SaveDefaultPresets: always writes a fresh snapshot of the current in-memory
// builtin presets.  This file is for human reference only — it is regenerated
// from HardcodedDefaults() every startup, so editing it has no effect.
void SaveDefaultPresets() {
    std::string path = defaultPresetFilePath();
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "# Auto-generated on startup — do not edit (changes will be overwritten)\n";
    f << "# To persist hardware calibration use Calibrate Hardware in the GUI\n";
    f << "# Format: name|...|drawCp|mixedDepthLow|mixedDepthRatio|wdlAlpha|wdlDrawElo|depthShuffle|depthShuffleBias|openingTemp|openingPlies|softmaxPlies|softmaxTemp|rootNoiseEps|recordMinPly|recordMaxEval|resignCount|drawCount|drawMinPly|drawAdjMoves|drawAdjThreshold|drawAdjMinMove\n";
    for (auto& p : g_allPresets) {
        if (p.isBuiltin) f << SerializePreset(p) << "\n";
    }
}

// SaveCalibration: persists ONLY the two hardware-dependent fields so they
// survive restarts without blocking future default-value changes.
void SaveCalibration() {
    std::string path = calibrationFilePath();
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "# Hardware calibration — auto-managed, do not edit manually\n";
    f << "# Format: presetName|gamesPerGen|maxPositions\n";
    for (auto& p : g_allPresets) {
        if (p.isBuiltin)
            f << p.name << "|" << p.gamesPerGen << "|" << p.maxPositions << "\n";
    }
}

// LoadCalibration: overlays stored hardware values on top of HardcodedDefaults.
void LoadCalibration() {
    std::string path = calibrationFilePath();
    std::ifstream f(path);
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string name, sGames, sMaxPos;
        if (!std::getline(ss, name, '|')) continue;
        if (!std::getline(ss, sGames, '|')) continue;
        if (!std::getline(ss, sMaxPos, '|')) continue;
        int games  = pInt(sGames,  0);
        int maxPos = pInt(sMaxPos, 0);
        if (games <= 0 && maxPos <= 0) continue;
        for (auto& p : g_allPresets) {
            if (p.isBuiltin && p.name == name) {
                if (games  > 0) p.gamesPerGen   = games;
                if (maxPos > 0) p.maxPositions   = maxPos;
                break;
            }
        }
    }
}

void HardcodedDefaults() {
    // ---- Speedy Test ----
    {
        Preset p;
        p.name = "Speedy Test"; p.isBuiltin = true;
        p.generations = 3; p.gamesPerGen = 20; p.epochsPerGen = 3;
        p.batchSize = 1024; p.lr = 0.001; p.weightDecay = 1e-5;
        p.dropout = 0.0; p.labelSmooth = 0.0; p.gradAccum = 2;
        p.warmupSteps = 0; p.drawWeight = 1.0; p.mateBoost = 3.0;
        p.splRatio = 0.4; p.workers = 12; p.depth = 5;
        p.eloValidate = false; p.eloGames = 10; p.swaGames = 5; p.overfitDetect = false;
        p.maxPositions = 10000; p.earlyStop = 5;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = false; p.swaStart = 3;
        p.drawPct = 10.0; p.frcMix = 0.0; p.replayWindow = 0; p.replayDecay = 0.7;
        p.wdlAlpha = 0.5; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- Mating Training ----
    {
        Preset p;
        p.name = "Mating Training"; p.isBuiltin = true;
        p.generations = 10; p.gamesPerGen = 3500; p.epochsPerGen = 12;
        p.batchSize = 2048; p.lr = 0.0008; p.weightDecay = 1e-4;
        p.dropout = 0.15; p.labelSmooth = 0.03; p.gradAccum = 4;
        p.warmupSteps = 55; p.drawWeight = 0.5; p.mateBoost = 10.0;
        p.splRatio = 0.5; p.workers = 12; p.depth = 8;
        p.eloValidate = false; p.eloGames = 60; p.swaGames = 30; p.overfitDetect = true;
        p.maxPositions = 350000; p.earlyStop = 10;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 4;
        p.drawPct = 10.0; p.frcMix = 0.0; p.replayWindow = 2; p.replayDecay = 0.7;
        p.wdlAlpha = 0.3; p.wdlDrawElo = 50.0;
        g_allPresets.push_back(p);
    }
    // ---- Opening Training ----
    {
        Preset p;
        p.name = "Opening Training"; p.isBuiltin = true;
        p.generations = 10; p.gamesPerGen = 4000; p.epochsPerGen = 10;
        p.batchSize = 2048; p.lr = 0.001; p.weightDecay = 1e-4;
        p.dropout = 0.12; p.labelSmooth = 0.04; p.gradAccum = 4;
        p.warmupSteps = 40; p.drawWeight = 1.5; p.mateBoost = 2.0;
        p.splRatio = 0.6; p.workers = 12; p.depth = 6;
        p.eloValidate = false; p.eloGames = 60; p.swaGames = 30; p.overfitDetect = true;
        p.maxPositions = 300000; p.earlyStop = 10;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 3;
        p.drawPct = 8.0; p.frcMix = 0.15; p.replayWindow = 3; p.replayDecay = 0.7;
        p.wdlAlpha = 0.6; p.wdlDrawElo = 130.0;
        g_allPresets.push_back(p);
    }
    // ---- Quick Test (~3 min) ----
    {
        Preset p;
        p.name = "Quick Test"; p.isBuiltin = true;
        p.generations = 3; p.gamesPerGen = 500; p.epochsPerGen = 3;
        p.batchSize = 1024; p.lr = 0.001; p.weightDecay = 1e-5;
        p.dropout = 0.0; p.labelSmooth = 0.0; p.gradAccum = 2;
        p.warmupSteps = 3; p.drawWeight = 1.0; p.mateBoost = 3.0;
        p.splRatio = 0.4; p.workers = 12; p.depth = 5;
        p.eloValidate = false; p.eloGames = 20; p.swaGames = 10; p.overfitDetect = false;
        p.maxPositions = 100000; p.earlyStop = 5;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = false; p.swaStart = 3;
        p.drawPct = 10.0; p.frcMix = 0.0; p.replayWindow = 1; p.replayDecay = 0.7;
        p.wdlAlpha = 0.5; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- 1 Hour ----
    {
        Preset p;
        p.name = "1 Hour"; p.isBuiltin = true;
        p.generations = 5; p.gamesPerGen = 750; p.epochsPerGen = 12;
        p.batchSize = 2048; p.lr = 0.0005; p.weightDecay = 1e-4;
        p.dropout = 0.04; p.labelSmooth = 0.02; p.gradAccum = 4;
        p.warmupSteps = 20; p.drawWeight = 1.0; p.mateBoost = 3.0;
        p.splRatio = 0.45; p.workers = 12; p.depth = 6;
        p.eloValidate = false; p.eloGames = 30; p.swaGames = 20; p.overfitDetect = true;
        p.maxPositions = 100000; p.earlyStop = 8;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 3;
        p.drawPct = 10.0; p.frcMix = 0.0; p.replayWindow = 3; p.replayDecay = 0.7;
        p.wdlAlpha = 0.5; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- 2 Hours ----
    {
        Preset p;
        p.name = "2 Hours"; p.isBuiltin = true;
        p.generations = 6; p.gamesPerGen = 900; p.epochsPerGen = 12;
        p.batchSize = 2048; p.lr = 0.0005; p.weightDecay = 1e-4;
        p.dropout = 0.04; p.labelSmooth = 0.02; p.gradAccum = 4;
        p.warmupSteps = 35; p.drawWeight = 1.0; p.mateBoost = 3.0;
        p.splRatio = 0.42; p.workers = 12; p.depth = 7;
        p.eloValidate = false; p.eloGames = 40; p.swaGames = 20; p.overfitDetect = true;
        p.maxPositions = 150000; p.earlyStop = 8;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 3;
        p.drawPct = 10.0; p.frcMix = 0.0; p.replayWindow = 3; p.replayDecay = 0.7;
        p.wdlAlpha = 0.5; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- 3 Hours ----
    {
        Preset p;
        p.name = "3 Hours"; p.isBuiltin = true;
        p.generations = 8; p.gamesPerGen = 1000; p.epochsPerGen = 12;
        p.batchSize = 2048; p.lr = 0.0005; p.weightDecay = 1e-4;
        p.dropout = 0.05; p.labelSmooth = 0.02; p.gradAccum = 4;
        p.warmupSteps = 50; p.drawWeight = 1.0; p.mateBoost = 3.0;
        p.splRatio = 0.4; p.workers = 12; p.depth = 7;
        p.eloValidate = true; p.eloGames = 50; p.swaGames = 30; p.overfitDetect = true;
        p.maxPositions = 160000; p.earlyStop = 8;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 3;
        p.drawPct = 12.0; p.frcMix = 0.10; p.replayWindow = 3; p.replayDecay = 0.7;
        p.wdlAlpha = 0.5; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- 5 Hours ----
    {
        Preset p;
        p.name = "5 Hours"; p.isBuiltin = true;
        p.generations = 10; p.gamesPerGen = 1200; p.epochsPerGen = 12;
        p.batchSize = 2048; p.lr = 0.0005; p.weightDecay = 1e-4;
        p.dropout = 0.05; p.labelSmooth = 0.02; p.gradAccum = 4;
        p.warmupSteps = 60; p.drawWeight = 1.0; p.mateBoost = 3.5;
        p.splRatio = 0.38; p.workers = 12; p.depth = 8;
        p.eloValidate = true; p.eloGames = 60; p.swaGames = 30; p.overfitDetect = true;
        p.maxPositions = 220000; p.earlyStop = 8;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 4;
        p.drawPct = 12.0; p.frcMix = 0.10; p.replayWindow = 3; p.replayDecay = 0.7;
        p.wdlAlpha = 0.5; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- 10 Hours ----
    {
        Preset p;
        p.name = "10 Hours"; p.isBuiltin = true;
        p.generations = 12; p.gamesPerGen = 1800; p.epochsPerGen = 11;
        p.batchSize = 2048; p.lr = 0.0005; p.weightDecay = 2e-4;
        p.dropout = 0.06; p.labelSmooth = 0.02; p.gradAccum = 4;
        p.warmupSteps = 90; p.drawWeight = 1.2; p.mateBoost = 4.0;
        p.splRatio = 0.37; p.workers = 12; p.depth = 8;
        p.eloValidate = true; p.eloGames = 80; p.swaGames = 40; p.overfitDetect = true;
        p.maxPositions = 350000; p.earlyStop = 8;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 4;
        p.drawPct = 15.0; p.frcMix = 0.15; p.replayWindow = 3; p.replayDecay = 0.7;
        p.wdlAlpha = 0.5; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- 15 Hours ----
    {
        Preset p;
        p.name = "15 Hours"; p.isBuiltin = true;
        p.generations = 15; p.gamesPerGen = 2200; p.epochsPerGen = 11;
        p.batchSize = 2048; p.lr = 0.0005; p.weightDecay = 2e-4;
        p.dropout = 0.06; p.labelSmooth = 0.02; p.gradAccum = 4;
        p.warmupSteps = 120; p.drawWeight = 1.2; p.mateBoost = 4.0;
        p.splRatio = 0.35; p.workers = 12; p.depth = 9;
        p.eloValidate = true; p.eloGames = 100; p.swaGames = 50; p.overfitDetect = true;
        p.maxPositions = 450000; p.earlyStop = 8;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 4;
        p.drawPct = 18.0; p.frcMix = 0.15; p.replayWindow = 3; p.replayDecay = 0.7;
        p.wdlAlpha = 0.5; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- Early Gen Training ----
    // Bootstrap phase: shallow search, high LR, heavy regularization.
    // Depth 4 games are fast, providing broad positional variety.
    // Tuned for Ryzen 7 7730U (8c/16t laptop, CPU-only, 16GB DDR4).
    {
        Preset p;
        p.name = "Early Gen Training"; p.isBuiltin = true;
        p.generations = 12; p.gamesPerGen = 3000; p.epochsPerGen = 8;
        p.batchSize = 2048; p.lr = 0.001; p.weightDecay = 1e-4;
        p.dropout = 0.10; p.labelSmooth = 0.05; p.gradAccum = 4;
        p.warmupSteps = 15; p.drawWeight = 0.8; p.mateBoost = 3.0;
        p.splRatio = 0.50; p.workers = 12; p.depth = 5;
        p.eloValidate = false; p.eloGames = 40; p.swaGames = 20; p.overfitDetect = true;
        p.maxPositions = 150000; p.earlyStop = 6;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = false; p.swaStart = 3;
        p.drawPct = 10.0; p.frcMix = 0.0; p.replayWindow = 2; p.replayDecay = 0.7;
        p.wdlAlpha = 0.4; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- Mid Gen Training ----
    // Transition phase: moderate depth, medium LR, balanced regularization.
    // Bridges Early Gen's broad exploration with Late Gen's deep refinement.
    // Depth 7 games give meaningful tactical content without Late Gen's cost.
    // Tuned for Ryzen 7 7730U (8c/16t laptop, CPU-only, 16GB DDR4).
    {
        Preset p;
        p.name = "Mid Gen Training"; p.isBuiltin = true;
        p.generations = 14; p.gamesPerGen = 2000; p.epochsPerGen = 10;
        p.batchSize = 2048; p.lr = 0.0006; p.weightDecay = 1e-4;
        p.dropout = 0.05; p.labelSmooth = 0.03; p.gradAccum = 4;
        p.warmupSteps = 20; p.drawWeight = 1.0; p.mateBoost = 4.0;
        p.splRatio = 0.42; p.workers = 12; p.depth = 7;
        p.eloValidate = true; p.eloGames = 60; p.swaGames = 30; p.overfitDetect = true;
        p.maxPositions = 250000; p.earlyStop = 7;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 5;
        p.drawPct = 15.0; p.frcMix = 0.10; p.replayWindow = 3; p.replayDecay = 0.75;
        p.wdlAlpha = 0.5; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
    // ---- Late Gen Training ----
    // Refinement phase: deep search, low LR, minimal regularization.
    // Depth 7 games produce high-quality positions for fine-tuning.
    // SWA smooths final weights.
    // Tuned for Ryzen 7 7730U (8c/16t laptop, CPU-only, 16GB DDR4).
    {
        Preset p;
        p.name = "Late Gen Training"; p.isBuiltin = true;
        p.generations = 15; p.gamesPerGen = 4000; p.epochsPerGen = 12;
        p.batchSize = 2048; p.lr = 0.0003; p.weightDecay = 5e-5;
        p.dropout = 0.02; p.labelSmooth = 0.01; p.gradAccum = 4;
        p.warmupSteps = 25; p.drawWeight = 1.3; p.mateBoost = 5.0;
        p.splRatio = 0.35; p.workers = 12; p.depth = 9;
        p.eloValidate = true; p.eloGames = 100; p.swaGames = 50; p.overfitDetect = true;
        p.maxPositions = 400000; p.earlyStop = 8;
        p.cosineLr = true; p.cosineT0 = 0; p.swa = true; p.swaStart = 4;
        p.drawPct = 20.0; p.frcMix = 0.20; p.replayWindow = 4; p.replayDecay = 0.8;
        p.wdlAlpha = 0.6; p.wdlDrawElo = 100.0;
        g_allPresets.push_back(p);
    }
}

void InitBuiltinPresets() {
    g_allPresets.clear();
    HardcodedDefaults();    // always start from code — no version bumping needed
    LoadCalibration();      // overlay hardware-specific gamesPerGen/maxPositions
    SaveDefaultPresets();   // write reference snapshot (human-readable, not loaded back)
}
void LoadCustomPresets() {
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

void SaveCustomPresets() {
    std::string path = presetFilePath();
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "# Custom presets\n";
    for (auto& p : g_allPresets) {
        if (p.isBuiltin) continue;
        f << SerializePreset(p) << "\n";
    }
}

void PopulatePresetCombo() {
    if (!g_ui.hPreset) return;
    SendMessageW(g_ui.hPreset, CB_RESETCONTENT, 0, 0);
    for (auto& p : g_allPresets)
        SendMessageW(g_ui.hPreset, CB_ADDSTRING, 0, (LPARAM)W(p.name).c_str());
    if (g_currentPresetIdx >= 0 && g_currentPresetIdx < (int)g_allPresets.size())
        SendMessageW(g_ui.hPreset, CB_SETCURSEL, g_currentPresetIdx, 0);
}

// FIX 10: Validate config ranges before pipeline launch.
// Returns empty string if valid, or a human-readable error message.
std::string ValidateConfig(const Config& cfg) {
    // ── Self-play parameters ──
    if (cfg.generations < 1 || cfg.generations > 1000)
        return "Generations must be 1-1000 (got " + std::to_string(cfg.generations) + ")";
    if (cfg.gamesPerGen < 10 || cfg.gamesPerGen > 100000)
        return "Games per gen must be 10-100,000 (got " + std::to_string(cfg.gamesPerGen) + ")";
    if (cfg.depth < 1 || cfg.depth > 20)
        return "Search depth must be 1-20 (got " + std::to_string(cfg.depth) + ")";
    if (cfg.workers < 1 || cfg.workers > 256)
        return "Workers must be 1-256 (got " + std::to_string(cfg.workers) + ")";

    // ── Training parameters ──
    if (cfg.epochsPerGen < 1 || cfg.epochsPerGen > 500)
        return "Epochs must be 1-500 (got " + std::to_string(cfg.epochsPerGen) + ")";
    if (cfg.batchSize < 1 || cfg.batchSize > 65536)
        return "Batch size must be 1-65,536 (got " + std::to_string(cfg.batchSize) + ")";
    if (cfg.lr <= 0.0 || cfg.lr > 1.0)
        return "Learning rate must be >0 and <=1.0 (got " + dbl2s(cfg.lr) + ")";
    if (cfg.weightDecay < 0.0 || cfg.weightDecay > 1.0)
        return "Weight decay must be 0-1.0 (got " + dbl2s(cfg.weightDecay) + ")";
    if (cfg.dropout < 0.0 || cfg.dropout > 0.9)
        return "Dropout must be 0-0.9 (got " + dbl2s(cfg.dropout) + ")";
    if (cfg.labelSmooth < 0.0 || cfg.labelSmooth > 1.0)
        return "Label smoothing must be 0-1.0 (got " + dbl2s(cfg.labelSmooth) + ")";
    if (cfg.gradAccum < 1 || cfg.gradAccum > 128)
        return "Gradient accumulation must be 1-128 (got " + std::to_string(cfg.gradAccum) + ")";
    if (cfg.warmupSteps < 0 || cfg.warmupSteps > 50000)
        return "Warmup steps must be 0-50,000 (got " + std::to_string(cfg.warmupSteps) + ")";

    // ── Data parameters ──
    if (cfg.maxPositions < 1000 || cfg.maxPositions > 10000000)
        return "Max positions must be 1,000-10,000,000 (got " + std::to_string(cfg.maxPositions) + ")";
    if (cfg.splRatio < 0.0 || cfg.splRatio > 1.0)
        return "Self-play ratio must be 0-1.0 (got " + dbl2s(cfg.splRatio) + ")";
    if (cfg.drawWeight < 0.0 || cfg.drawWeight > 10.0)
        return "Draw weight must be 0-10.0 (got " + dbl2s(cfg.drawWeight) + ")";

    // ── Game parameters ──
    if (cfg.resignCp < 0 || cfg.resignCp > 10000)
        return "Resign threshold must be 0-10,000 cp (got " + std::to_string(cfg.resignCp) + ")";
    if (cfg.maxPlies < 10 || cfg.maxPlies > 2000)
        return "Max plies must be 10-2,000 (got " + std::to_string(cfg.maxPlies) + ")";

    // ── Mixed depth consistency ──
    if (cfg.mixedDepthRatio > 0.0 && cfg.mixedDepthLow >= cfg.depth)
        return "Mixed depth low (" + std::to_string(cfg.mixedDepthLow) +
               ") must be less than main depth (" + std::to_string(cfg.depth) + ")";
    if (cfg.depthShuffle && cfg.mixedDepthRatio > 0.0 && cfg.depth - cfg.mixedDepthLow < 2)
        return "Depth shuffle requires at least 2 depth levels (depth=" + std::to_string(cfg.depth) +
               ", low=" + std::to_string(cfg.mixedDepthLow) + ")";
    if (cfg.depthShuffle && cfg.depthShuffleBias < 0.1)
        return "Depth shuffle bias must be >= 0.1 (got " + dbl2s(cfg.depthShuffleBias) + ")";
    if (cfg.depthShuffle && cfg.mixedDepthRatio <= 0.0)
        return "Depth shuffle requires Mixed Depth % > 0 (currently 0% — depth shuffle would have no effect)";

    // ── Self-play diversity settings ──
    if (cfg.openingTemp < 0.0 || cfg.openingTemp > 10.0)
        return "Opening temp must be 0-10.0 (got " + dbl2s(cfg.openingTemp) + ")";
    if (cfg.openingPlies < 0 || cfg.openingPlies > 30)
        return "Opening plies must be 0-30 (got " + std::to_string(cfg.openingPlies) + ")";
    if (cfg.softmaxPlies < 0 || cfg.softmaxPlies > 30)
        return "Softmax plies must be 0-30 (got " + std::to_string(cfg.softmaxPlies) + ")";
    if (cfg.softmaxTemp < 0.0 || cfg.softmaxTemp > 10.0)
        return "Softmax temp must be 0-10.0 (got " + dbl2s(cfg.softmaxTemp) + ")";
    if (cfg.rootNoiseEps < 0.0 || cfg.rootNoiseEps > 1.0)
        return "Root noise must be 0-1.0 (got " + dbl2s(cfg.rootNoiseEps) + ")";
    if (cfg.recordMinPly < 0 || cfg.recordMinPly > 100)
        return "Record min ply must be 0-100 (got " + std::to_string(cfg.recordMinPly) + ")";
    if (cfg.recordMaxEval < 100 || cfg.recordMaxEval > 30000)
        return "Record max eval must be 100-30000 (got " + std::to_string(cfg.recordMaxEval) + ")";

    // ── Adjudication fine-tuning ──
    if (cfg.resignCount < 1 || cfg.resignCount > 20)
        return "Resign count must be 1-20 (got " + std::to_string(cfg.resignCount) + ")";
    if (cfg.drawCount < 1 || cfg.drawCount > 30)
        return "Draw count must be 1-30 (got " + std::to_string(cfg.drawCount) + ")";
    if (cfg.drawMinPly < 0 || cfg.drawMinPly > 200)
        return "Draw min ply must be 0-200 (got " + std::to_string(cfg.drawMinPly) + ")";
    if (cfg.drawAdjMoves < 1 || cfg.drawAdjMoves > 50)
        return "Draw adj moves must be 1-50 (got " + std::to_string(cfg.drawAdjMoves) + ")";
    if (cfg.drawAdjThreshold < 1 || cfg.drawAdjThreshold > 50)
        return "Draw adj threshold must be 1-50 cp (got " + std::to_string(cfg.drawAdjThreshold) + ")";
    if (cfg.drawAdjMinMove < 10 || cfg.drawAdjMinMove > 200)
        return "Draw adj min move must be 10-200 (got " + std::to_string(cfg.drawAdjMinMove) + ")";

    // ── ELO validation ──
    if (cfg.eloValidate && cfg.eloGames < 2)
        return "ELO validation requires at least 2 games (got " + std::to_string(cfg.eloGames) + ")";

    return "";  // all good
}

// ── Read config from UI ───────────────────────────────────────────
Config ReadConfig() {
    Config c;
    if (!g_ui.hCfgPane) return c;  // FIX 6.32: handles not yet created
    auto e = [&](int id) -> std::string {
        auto it = g_ui.edits.find(id);
        if (it == g_ui.edits.end()) return "";
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
    c.mixedDepthRatio = pDbl(e(ID_EDIT_MIXDEPTH_PCT), 0.0) / 100.0;  // FIX 6.33: no float cast
    c.mixedDepthLow   = pInt(e(ID_EDIT_MIXDEPTH_LOW), 4);
    c.depthShuffle    = Button_GetCheck(g_ui.hChkDepthShuffle) == BST_CHECKED;
    c.depthShuffleBias = pDbl(e(ID_EDIT_DEPTH_SHUFFLE_BIAS), 2.0);
    c.startGen     = pInt(e(ID_EDIT_STARTGEN), 0);
    c.maxPositions  = pInt(e(ID_EDIT_MAXPOS), 300000);
    c.earlyStop     = pInt(e(ID_EDIT_EARLYSTOP), 10);
    c.cosineLr      = Button_GetCheck(g_ui.hChkCosineLR) == BST_CHECKED;
    c.cosineT0      = pInt(e(ID_EDIT_COSINET0), 50);
    c.swa           = Button_GetCheck(g_ui.hChkSWA) == BST_CHECKED;
    c.swaStart      = pInt(e(ID_EDIT_SWASTART), 3);
    c.drawPct       = pDbl(e(ID_EDIT_DRAWPCT), 10.0);
    c.frcMix        = pDbl(e(ID_EDIT_FRCMIX), 0.0) / 100.0;  // GUI shows %, config uses fraction
    c.replayWindow  = pInt(e(ID_EDIT_REPLAYWIN), 3);
    c.replayDecay   = pDbl(e(ID_EDIT_REPLAYDECAY), 0.7);
    c.wdlAlpha      = pDbl(e(ID_EDIT_WDLALPHA), 0.5);
    c.wdlDrawElo    = pDbl(e(ID_EDIT_WDLDRAWELO), 100.0);
    c.eloGames      = pInt(e(ID_EDIT_ELOGAMES), 100);
    c.swaGames      = pInt(e(ID_EDIT_SWAGAMES), 50);
    c.eloValidate   = Button_GetCheck(g_ui.hChkElo)   == BST_CHECKED;
    c.overfitDetect = Button_GetCheck(g_ui.hChkOvfit) == BST_CHECKED;
    c.resignCp     = pInt(e(ID_EDIT_RESIGNCP),  500);
    c.contemptCp   = pInt(e(ID_EDIT_CONTEMPT),   25);
    c.maxPlies     = pInt(e(ID_EDIT_MAXPLIES),  250);
    c.drawCp       = pInt(e(ID_EDIT_DRAWCP),      8);
    c.openingTemp    = pDbl(e(ID_EDIT_OPENING_TEMP),  1.5);
    c.openingPlies   = pInt(e(ID_EDIT_OPENING_PLIES), 4);
    c.softmaxPlies   = pInt(e(ID_EDIT_SOFTMAX_PLIES), 8);
    c.softmaxTemp    = pDbl(e(ID_EDIT_SOFTMAX_TEMP),  0.5);
    c.rootNoiseEps   = pDbl(e(ID_EDIT_ROOT_NOISE),    0.0);
    c.recordMinPly   = pInt(e(ID_EDIT_RECORD_MIN_PLY),  10);
    c.recordMaxEval  = pInt(e(ID_EDIT_RECORD_MAX_EVAL), 2500);
    c.resignCount    = pInt(e(ID_EDIT_RESIGN_COUNT),    3);
    c.drawCount      = pInt(e(ID_EDIT_DRAW_COUNT),      6);
    c.drawMinPly     = pInt(e(ID_EDIT_DRAW_MIN_PLY),    40);
    c.drawAdjMoves   = pInt(e(ID_EDIT_DRAW_ADJ_MOVES),  12);
    c.drawAdjThreshold = pInt(e(ID_EDIT_DRAW_ADJ_THRESH), 4);
    c.drawAdjMinMove = pInt(e(ID_EDIT_DRAW_ADJ_MIN_MOVE), 50);
    return c;
}

// ── Apply preset from g_allPresets ────────────────────────────────
void ApplyPreset(int idx) {
    if (idx < 0 || idx >= (int)g_allPresets.size()) return;
    g_currentPresetIdx = idx;
    Preset& p = g_allPresets[idx];
    auto se = [&](int id, const std::string& v) {
        auto it = g_ui.edits.find(id);
        if (it != g_ui.edits.end()) setEdit(it->second, v);
    };
    se(ID_EDIT_GENS,      std::to_string(p.generations));
    se(ID_EDIT_GAMES,     std::to_string(p.gamesPerGen));
    se(ID_EDIT_EPOCHS,    std::to_string(p.epochsPerGen));
    se(ID_EDIT_BATCHSZ,   std::to_string(p.batchSize));
    se(ID_EDIT_WORKERS,   std::to_string(p.workers));
    se(ID_EDIT_DEPTH,     std::to_string(p.depth));
    se(ID_EDIT_MIXDEPTH_PCT, dbl2s(p.mixedDepthRatio * 100.0, 0));
    se(ID_EDIT_MIXDEPTH_LOW, std::to_string(p.mixedDepthLow));
    Button_SetCheck(g_ui.hChkDepthShuffle, p.depthShuffle ? BST_CHECKED : BST_UNCHECKED);
    se(ID_EDIT_DEPTH_SHUFFLE_BIAS, dbl2s(p.depthShuffleBias, 1));
    se(ID_EDIT_GRADACCUM, std::to_string(p.gradAccum));
    se(ID_EDIT_WARMUP,    std::to_string(p.warmupSteps));
    se(ID_EDIT_LR,        dbl2s(p.lr,6));
    se(ID_EDIT_WD,        dbl2s(p.weightDecay,8));
    se(ID_EDIT_DROPOUT,   dbl2s(p.dropout,4));
    se(ID_EDIT_LSMOOTH,   dbl2s(p.labelSmooth,4));
    se(ID_EDIT_DRAWWT,    dbl2s(p.drawWeight,4));
    se(ID_EDIT_MATEBOOST, dbl2s(p.mateBoost,4));
    se(ID_EDIT_SPLRATIO,  dbl2s(p.splRatio,4));
    se(ID_EDIT_MAXPOS,    std::to_string(p.maxPositions));
    se(ID_EDIT_EARLYSTOP, std::to_string(p.earlyStop));
    se(ID_EDIT_COSINET0,  std::to_string(p.cosineT0));
    se(ID_EDIT_SWASTART,  std::to_string(p.swaStart));
    se(ID_EDIT_DRAWPCT,   dbl2s(p.drawPct,1));
    se(ID_EDIT_FRCMIX,    dbl2s(p.frcMix * 100.0, 1));
    se(ID_EDIT_REPLAYWIN,    std::to_string(p.replayWindow));
    se(ID_EDIT_REPLAYDECAY,  dbl2s(p.replayDecay, 2));
    se(ID_EDIT_WDLALPHA,    dbl2s(p.wdlAlpha, 2));
    se(ID_EDIT_WDLDRAWELO,  dbl2s(p.wdlDrawElo, 1));
    Button_SetCheck(g_ui.hChkCosineLR, p.cosineLr      ? BST_CHECKED : BST_UNCHECKED);
    Button_SetCheck(g_ui.hChkSWA,      p.swa           ? BST_CHECKED : BST_UNCHECKED);
    se(ID_EDIT_ELOGAMES,  std::to_string(p.eloGames));
    se(ID_EDIT_SWAGAMES,  std::to_string(p.swaGames));
    Button_SetCheck(g_ui.hChkElo,      p.eloValidate   ? BST_CHECKED : BST_UNCHECKED);
    Button_SetCheck(g_ui.hChkOvfit,    p.overfitDetect  ? BST_CHECKED : BST_UNCHECKED);
    se(ID_EDIT_RESIGNCP,  std::to_string(p.resignCp));
    se(ID_EDIT_CONTEMPT,  std::to_string(p.contemptCp));
    se(ID_EDIT_MAXPLIES,  std::to_string(p.maxPlies));
    se(ID_EDIT_DRAWCP,    std::to_string(p.drawCp));
    se(ID_EDIT_OPENING_TEMP,   dbl2s(p.openingTemp, 2));
    se(ID_EDIT_OPENING_PLIES,  std::to_string(p.openingPlies));
    se(ID_EDIT_SOFTMAX_PLIES,  std::to_string(p.softmaxPlies));
    se(ID_EDIT_SOFTMAX_TEMP,   dbl2s(p.softmaxTemp, 2));
    se(ID_EDIT_ROOT_NOISE,     dbl2s(p.rootNoiseEps, 3));
    se(ID_EDIT_RECORD_MIN_PLY, std::to_string(p.recordMinPly));
    se(ID_EDIT_RECORD_MAX_EVAL,std::to_string(p.recordMaxEval));
    se(ID_EDIT_RESIGN_COUNT,   std::to_string(p.resignCount));
    se(ID_EDIT_DRAW_COUNT,     std::to_string(p.drawCount));
    se(ID_EDIT_DRAW_MIN_PLY,   std::to_string(p.drawMinPly));
    se(ID_EDIT_DRAW_ADJ_MOVES, std::to_string(p.drawAdjMoves));
    se(ID_EDIT_DRAW_ADJ_THRESH,std::to_string(p.drawAdjThreshold));
    se(ID_EDIT_DRAW_ADJ_MIN_MOVE, std::to_string(p.drawAdjMinMove));
    if (g_ui.hBtnDel) EnableWindow(g_ui.hBtnDel, !p.isBuiltin);
}

// ── Save As dialog helper ─────────────────────────────────────────

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
            GetDlgItemTextW(hDlg, 200, d2->name, 128); // INFO [6.23]: 128-char limit silently truncates long preset names — acceptable UX trade-off
            d2->ok = true;
            EndDialog(hDlg, IDOK); return TRUE;
        }
        if (LOWORD(wp2)==IDCANCEL) { EndDialog(hDlg, IDCANCEL); return TRUE; }
    }
    if (msg==WM_CLOSE) { EndDialog(hDlg, IDCANCEL); return TRUE; }
    return FALSE;
}

// ── Save As dialog ───────────────────────────────────────────────
void SavePresetAs() {
    // Build a dialog template in memory
    SavePresetDlgData dd = {};
    // FIX 6.23: wcscpy_s is safe here — literal fits in dd.name[128].
    // GetDlgItemTextW at line 3972 also respects the 128 limit.
    wcscpy_s(dd.name, L"My Custom Preset"); dd.ok = false;


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

    INT_PTR result = DialogBoxIndirectParamW(g_hInst, (DLGTEMPLATE*)buf, g_ui.hWnd,
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
    np.mixedDepthLow = c.mixedDepthLow; np.mixedDepthRatio = c.mixedDepthRatio;
    np.depthShuffle = c.depthShuffle; np.depthShuffleBias = c.depthShuffleBias;
    np.maxPositions = c.maxPositions; np.earlyStop = c.earlyStop;
    np.cosineLr = c.cosineLr; np.cosineT0 = c.cosineT0;
    np.swa = c.swa; np.swaStart = c.swaStart;
    np.drawPct = c.drawPct; np.frcMix = c.frcMix;
    np.replayWindow = c.replayWindow; np.replayDecay = c.replayDecay;
    np.wdlAlpha = c.wdlAlpha; np.wdlDrawElo = c.wdlDrawElo;
    np.eloValidate = c.eloValidate; np.eloGames = c.eloGames; np.swaGames = c.swaGames; np.overfitDetect = c.overfitDetect;
    np.resignCp = c.resignCp; np.contemptCp = c.contemptCp;
    np.maxPlies = c.maxPlies; np.drawCp = c.drawCp;
    np.openingTemp = c.openingTemp; np.openingPlies = c.openingPlies;
    np.softmaxPlies = c.softmaxPlies; np.softmaxTemp = c.softmaxTemp;
    np.rootNoiseEps = c.rootNoiseEps;
    np.recordMinPly = c.recordMinPly; np.recordMaxEval = c.recordMaxEval;
    np.resignCount = c.resignCount; np.drawCount = c.drawCount;
    np.drawMinPly = c.drawMinPly; np.drawAdjMoves = c.drawAdjMoves;
    np.drawAdjThreshold = c.drawAdjThreshold; np.drawAdjMinMove = c.drawAdjMinMove;

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
    if (g_ui.hBtnDel) EnableWindow(g_ui.hBtnDel, TRUE);
}

void DeleteCurrentPreset() {
    if (g_currentPresetIdx < 0 || g_currentPresetIdx >= (int)g_allPresets.size()) return;
    if (g_allPresets[g_currentPresetIdx].isBuiltin) return;
    std::string name = g_allPresets[g_currentPresetIdx].name;
    int res = MessageBoxW(g_ui.hWnd,
        (L"Delete preset \"" + W(name) + L"\"?").c_str(),
        L"Confirm Delete", MB_YESNO | MB_ICONQUESTION);
    if (res != IDYES) return;
    g_allPresets.erase(g_allPresets.begin() + g_currentPresetIdx);
    SaveCustomPresets();
    g_currentPresetIdx = std::min(1, (int)g_allPresets.size() - 1);
    PopulatePresetCombo();
    ApplyPreset(g_currentPresetIdx);
}

