// TR_Pipeline.cpp  --  Training pipeline: SelfPlay, Training, EloVal, SWA
#include "TR_Types.h"
#include "TR_Globals.h"
#include "TR_Fwd.h"
// TR_VCEnv.h no longer needed — g++ (MinGW) is used instead of MSVC

// ── AppState::_writeToFileLog  (called from pushLog in TR_Types.h) ──
// Auto-classifies log level from message content and forwards to file logger.
void AppState::_writeToFileLog(const std::string& s) {
    if (!g_fileLog.isOpen()) return;
    if (s.empty()) return;
    // Skip \r progress updates (noisy)
    if (s[0] == '\r') return;

    // Read current phase and gen from AppState (already under lock from pushLog caller)
    std::string curPhase = this->phase;
    int curGenNum = this->curGen;

    // Auto-classify level
    LogLevel lvl = LogLevel::INFO;
    if (s.find("[ERR]") != std::string::npos || s.find("[ERROR]") != std::string::npos ||
        s.find("Error") != std::string::npos || s.find("failed") != std::string::npos ||
        s.find("Failed") != std::string::npos)
        lvl = LogLevel::ERR;
    else if (s.find("[WARN]") != std::string::npos || s.find("WARNING") != std::string::npos)
        lvl = LogLevel::WARN;
    else if (s.find("[CMD]") != std::string::npos)
        lvl = LogLevel::CMD;
    else if (s.find("[SKIP]") != std::string::npos)
        lvl = LogLevel::EVENT;

    g_fileLog.log(lvl, curPhase, curGenNum, s);
}

// ── Loss parser ───────────────────────────────────────────────────
static bool ParseLoss(const std::string& line, TrainPoint& pt) {
    // AUDIT FIX H3: Add word-boundary check to prevent substring false matches
    auto findNum = [&](const std::string& key) -> double {
        size_t i = 0;
        while ((i = line.find(key, i)) != std::string::npos) {
            if (i > 0 && std::isalnum((unsigned char)line[i - 1])) { i++; continue; }
            i += key.size();
            while (i < line.size() && (line[i] == ' ' || line[i] == ':')) i++;
            try { size_t n; return std::stod(line.substr(i), &n); } catch (...) {}
            return -1.0;
        }
        return -1.0;
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

// ── Phase loss parser ──────────────────────────────────────────────
static bool ParsePhaseLoss(const std::string& line, TrainPoint& pt) {
    if (line.find("Phase loss") == std::string::npos) return false;
    // AUDIT FIX H3: Add word-boundary check to prevent substring false matches
    auto findNum = [&](const std::string& key) -> double {
        size_t i = 0;
        while ((i = line.find(key, i)) != std::string::npos) {
            if (i > 0 && std::isalnum((unsigned char)line[i - 1])) { i++; continue; }
            i += key.size();
            while (i < line.size() && (line[i] == ' ' || line[i] == ':')) i++;
            try { size_t n; return std::stod(line.substr(i), &n); } catch (...) {}
            return -1.0;
        }
        return -1.0;
    };
    double op = findNum("Opening");
    double mg = findNum("Middlegame");
    double eg = findNum("Endgame");
    if (op < 0 && mg < 0 && eg < 0) return false;
    pt.openingLoss    = (op >= 0) ? op : 0.0;
    pt.middlegameLoss = (mg >= 0) ? mg : 0.0;
    pt.endgameLoss    = (eg >= 0) ? eg : 0.0;
    pt.hasPhase = true;
    return true;
}

// ── Pipeline ──────────────────────────────────────────────────────
// Play an MP3 from the assets folder (non-blocking, fire-and-forget).
static void PlayMp3(const std::string& alias, const std::string& dataDir, const std::string& filename) {
    if (g_muteSounds) return;
    fs::path soundPath = fs::path(exeDir()) / dataDir / filename;
    if (!fs::exists(soundPath)) return;
    std::string close = "close " + alias;
    std::string open  = "open \"" + soundPath.string() + "\" type mpegvideo alias " + alias;
    std::string play  = "play " + alias;
    mciSendStringA(close.c_str(), NULL, 0, NULL);
    mciSendStringA(open.c_str(), NULL, 0, NULL);
    mciSendStringA(play.c_str(), NULL, 0, NULL);
}

static bool SelfPlay(const Config& cfg, int gen) {
    std::string d = exeDir();
    fs::path assetsDir = fs::path(d)/cfg.dataDir;
    fs::create_directories(assetsDir);
    fs::path outFile = assetsDir/("selfplay_gen"+std::to_string(gen)+".bin");
    std::string weightsArg;
    if (gen > 1) {
        fs::path prev = assetsDir/("nnue_weights_gen"+std::to_string(gen-1)+".bin");
        if (fs::exists(prev)) weightsArg = " --weights \""+prev.string()+"\"";
    }
    // Auto-detect openings book in assets directory
    std::string openingsArg;
    {
        fs::path openingsPath = assetsDir / "openings.txt";
        if (fs::exists(openingsPath))
            openingsArg = " --openings \"" + openingsPath.string() + "\"";
    }
    std::wstring cmd = W(
        "\"" + (fs::path(d)/cfg.exeName).string() + "\""
        " --generate --games " + std::to_string(cfg.gamesPerGen) +
        " --depth " + std::to_string(cfg.depth) +
        (cfg.mixedDepthRatio > 0.0 ? " --mixed-depth-ratio " + dbl2s(cfg.mixedDepthRatio, 4) + " --mixed-depth-low " + std::to_string(cfg.mixedDepthLow) + (cfg.depthShuffle ? " --depth-shuffle --depth-shuffle-bias " + dbl2s(cfg.depthShuffleBias, 2) : "") : "") +
        " --workers " + std::to_string(cfg.workers) +
        " --output \"" + outFile.string() + "\"" +
        weightsArg + openingsArg +
        (cfg.frcMix > 0 ? " --frc-mix " + dbl2s(cfg.frcMix, 2) : "") +
        // Scale game time limit by depth: each extra depth level ~3x more nodes
        // depth<=5: 120s  depth=6: 240s  depth=7: 480s  depth>=8: 720s
        " --game-time-limit " + std::to_string(
            cfg.depth <= 5 ? 120 :
            cfg.depth == 6 ? 240 :
            cfg.depth == 7 ? 480 : 720) +
        " --resign-cp " + std::to_string(cfg.resignCp) +
        " --contempt " + std::to_string(cfg.contemptCp) +
        " --maxplies " + std::to_string(cfg.maxPlies) +
        " --draw-cp " + std::to_string(cfg.drawCp) +
        " --resign-count " + std::to_string(cfg.resignCount) +
        " --draw-count " + std::to_string(cfg.drawCount) +
        " --draw-min-ply " + std::to_string(cfg.drawMinPly) +
        " --draw-adj-moves " + std::to_string(cfg.drawAdjMoves) +
        " --draw-adj-threshold " + std::to_string(cfg.drawAdjThreshold) +
        " --draw-adj-min-move " + std::to_string(cfg.drawAdjMinMove) +
        " --opening-plies " + std::to_string(cfg.openingPlies) +
        " --opening-temp " + dbl2s(cfg.openingTemp, 2) +
        " --softmax-plies " + std::to_string(cfg.softmaxPlies) +
        " --softmax-temp " + dbl2s(cfg.softmaxTemp, 2) +
        (cfg.rootNoiseEps > 0.0 ? " --root-noise " + dbl2s(cfg.rootNoiseEps, 3) : "") +
        " --record-min-ply " + std::to_string(cfg.recordMinPly) +
        " --record-max-eval " + std::to_string(cfg.recordMaxEval)
    );
    g_st.setStatus("Gen "+std::to_string(gen)+": self-play ("+std::to_string(cfg.gamesPerGen)+" games)");
    g_st.setPhase("selfplay");
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.phaseStart = std::chrono::steady_clock::now(); }  // FIX 6.1
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.pausedPhaseSec = 0; }
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curEpoch=0; g_st.totalEpochs=cfg.gamesPerGen; g_st.selfPlayEtaSec=0; }
    g_fileLog.logPhaseStart("selfplay", gen, "games=" + std::to_string(cfg.gamesPerGen) + " depth=" + std::to_string(cfg.depth) + " workers=" + std::to_string(cfg.workers));
    auto selfPlayStart = std::chrono::steady_clock::now();
    bool selfPlayOk = RunProc(cmd, d, [&](const std::string& ln){
        g_st.pushLog(ln);
        // SelfPlayGen outputs: "\r[SelfPlay] done/total  pos=..."
        // Parse the leading integer after "[SelfPlay] " as games completed.
        auto p = ln.find("[SelfPlay] ");
        if (p != std::string::npos) {
            try { size_t n; int g2=std::stoi(ln.substr(p+11), &n);
                std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curEpoch=g2; }
            catch(...) {}
        }
        // Parse ETA from "[SelfPlay] ... ETA HH:MM:SS"
        if (p != std::string::npos) {
            auto ep = ln.find("ETA ");
            if (ep != std::string::npos) {
                std::string etaStr = ln.substr(ep + 4);
                // Parse HH:MM:SS format
                int hh = 0, mm = 0, ss = 0;
                if (std::sscanf(etaStr.c_str(), "%d:%d:%d", &hh, &mm, &ss) >= 2) {
                    int totalSec = hh * 3600 + mm * 60 + ss;
                    auto now = std::chrono::steady_clock::now();
                    std::lock_guard<std::mutex> lk(g_st.mtx);
                    g_st.selfPlayEtaSec   = totalSec;
                    g_st.selfPlayEtaStamp = now;
                }
            }
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
        // Parse NPS from STATS/PROGRESS lines: "... 360 nps" or "1.2K nps" or "1.5M nps"
        {
            auto npsPos = ln.find(" nps");
            if (npsPos != std::string::npos && npsPos > 0) {
                // Walk back to find the start of the number
                size_t numEnd = npsPos;
                size_t numStart = numEnd;
                while (numStart > 0 && (std::isdigit(ln[numStart-1]) || ln[numStart-1] == '.' || ln[numStart-1] == ' '))
                    --numStart;
                // Skip leading space
                while (numStart < numEnd && ln[numStart] == ' ') ++numStart;
                std::string numStr = ln.substr(numStart, numEnd - numStart);
                // Handle K/M suffix that may appear just before " nps"
                double npsVal = 0.0;
                try {
                    if (!numStr.empty() && (numStr.back() == 'K' || numStr.back() == 'k')) {
                        npsVal = std::stod(numStr.substr(0, numStr.size()-1)) * 1000.0;
                    } else if (!numStr.empty() && (numStr.back() == 'M' || numStr.back() == 'm')) {
                        npsVal = std::stod(numStr.substr(0, numStr.size()-1)) * 1000000.0;
                    } else {
                        npsVal = std::stod(numStr);
                    }
                    if (npsVal > 0.0) {
                        std::lock_guard<std::mutex> lk(g_st.mtx);
                        g_st.curNps = npsVal;
                    }
                } catch(...) {}
            }
        }
    }, g_st.stopFlag);
    double selfPlayElapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - selfPlayStart).count());
    g_fileLog.logPhaseEnd("selfplay", gen, selfPlayOk, selfPlayElapsed);
    return selfPlayOk;
}

static bool Training(const Config& cfg, int gen) {
    std::string d = exeDir();
    fs::path assetsDir = fs::path(d)/cfg.dataDir;
    fs::create_directories(assetsDir);
    // Data source priority: Stockfish-labeled > shards > legacy material-count
    fs::path sfData       = assetsDir/"training_data_sf.bin";
    fs::path legacyData   = assetsDir/"training_data.bin";
    fs::path shardsDir    = assetsDir/"training_data_shards";
    fs::path baseData;

    if (fs::exists(sfData)) {
        baseData = sfData;
        g_st.pushLog("  Using Stockfish-labeled data: " + sfData.string());
    } else if (fs::exists(shardsDir / "shards.cfg")) {
        baseData = shardsDir;
        g_st.pushLog("  Using sharded training data: " + shardsDir.string());
    } else if (fs::exists(legacyData)) {
        baseData = legacyData;
        g_st.pushLog("  WARNING: Using legacy material-count data (training_data_sf.bin not found)");
    } else {
        g_st.pushLog("  ERROR: No training data found in " + assetsDir.string());
        return false;
    }
    // Build self-play extra-data args: current gen + replay buffer from past generations
    // Weights are normalized so ALL self-play combined equals exactly splRatio.
    // Without normalization, replay window causes total self-play to be
    // splRatio*(1 + decay + decay^2 + ...) which starves the base dataset.
    // Example: splRatio=0.4, decay=0.7, window=2 -> old: 87.6% self-play, new: 40% self-play.
    std::string selfplayArgs;
    {
        fs::path currentSp = assetsDir/("selfplay_gen"+std::to_string(gen)+".bin");

        // Step 1: sum decay weights for existing files to compute normalization factor
        double normFactor = 0.0;
        if (fs::exists(currentSp)) normFactor += 1.0;
        if (cfg.replayWindow > 0) {
            double w = 1.0;
            for (int pastGen = gen - 1; pastGen >= 1 && pastGen >= gen - cfg.replayWindow; --pastGen) {
                w *= cfg.replayDecay;
                fs::path pastSp = assetsDir/("selfplay_gen"+std::to_string(pastGen)+".bin");
                if (fs::exists(pastSp)) normFactor += w;
            }
        }
        if (normFactor <= 0.0) normFactor = 1.0;  // no self-play files: avoid divide-by-zero

        // Step 2: assign normalized weights so total self-play == splRatio
        if (fs::exists(currentSp)) {
            double w = cfg.splRatio / normFactor;
            selfplayArgs += " --extra-data \"" + currentSp.string() + "\" " + dbl2s(w, 4);
            g_st.pushLog("  Self-play current gen (weight " + dbl2s(w, 4) + ", base gets " + dbl2s(1.0 - cfg.splRatio, 4) + ")");
        }
        if (cfg.replayWindow > 0) {
            double decay = 1.0;
            for (int pastGen = gen - 1; pastGen >= 1 && pastGen >= gen - cfg.replayWindow; --pastGen) {
                decay *= cfg.replayDecay;
                fs::path pastSp = assetsDir/("selfplay_gen"+std::to_string(pastGen)+".bin");
                if (fs::exists(pastSp)) {
                    double w = cfg.splRatio * decay / normFactor;
                    selfplayArgs += " --extra-data \"" + pastSp.string() + "\" " + dbl2s(w, 4);
                    g_st.pushLog("  Replay buffer: gen " + std::to_string(pastGen) + " (weight " + dbl2s(w, 4) + ")");
                }
            }
        }
    }
    // FIX 6.22: Guard against exceeding CreateProcessW's 32767-char command line limit
    if (selfplayArgs.size() > 30000) {
        g_st.pushLog("[WARN] selfplay command line is " + std::to_string(selfplayArgs.size()) +
                     " chars — approaching Windows 32767 limit. Consider reducing replay window.");
    }
    fs::path prevWeights  = assetsDir/("nnue_weights_gen"+std::to_string(gen-1)+".bin");
    fs::path outputWeights= assetsDir/"nnue_weights.bin";
    fs::path genWeights   = assetsDir/("nnue_weights_gen"+std::to_string(gen)+".bin");
    std::string args =
        " --data \""        + baseData.string() + "\""
        + selfplayArgs +
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
        " --wdl-alpha "     + dbl2s(cfg.wdlAlpha) +
        " --wdl-draw-elo "  + dbl2s(cfg.wdlDrawElo) +
        " --enhanced --plot";
    if (cfg.cosineLr) {
        args += " --cosine-lr";
        if (cfg.cosineT0 > 0) {
            args += " --cosine-t0 " + std::to_string(cfg.cosineT0);
        } else {
            args += " --no-cosine-restarts";
        }
    } else {
        args += " --no-cosine-lr";
    }
    if (cfg.swa) {
        args += " --swa --swa-start " + std::to_string(cfg.swaStart);
    }
    if (cfg.labelSmooth > 0.0)
        args += " --label-smoothing " + dbl2s(cfg.labelSmooth);
    // Auto-detect draws dataset and include as extra training data
    fs::path drawsData = assetsDir/"training_data_draws.bin";
    if (fs::exists(drawsData) && cfg.drawPct > 0.0) {
        double drawRatio = cfg.drawPct / 100.0;
        if (drawRatio > 0.95) drawRatio = 0.95;
        args += " --extra-data \"" + drawsData.string() + "\" " + dbl2s(drawRatio, 4);
    }
    if (fs::exists(prevWeights))
        args += " --load-weights \"" + prevWeights.string() + "\"";
    args += " --output \"" + outputWeights.string() + "\"";
    std::wstring cmd = W("py -3.10 -u \"" + (fs::path(d)/cfg.pyScript).string() + "\"" + args);
    g_st.setStatus("Gen "+std::to_string(gen)+": training ("+std::to_string(cfg.epochsPerGen)+" epochs)");
    g_st.setPhase("training");
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.phaseStart = std::chrono::steady_clock::now(); }  // FIX 6.1
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.pausedPhaseSec = 0; }
    { std::lock_guard<std::mutex> lk(g_st.mtx);
      g_st.curEpoch=0; g_st.totalEpochs=cfg.epochsPerGen;
      g_st.batchEtaSec=0; g_st.epochEtaSec=0; g_st.nextEpochSec=0; }
    g_fileLog.logPhaseStart("training", gen, "epochs=" + std::to_string(cfg.epochsPerGen) +
        " batch=" + std::to_string(cfg.batchSize) + " lr=" + dbl2s(cfg.lr, 8));
    auto trainingStart = std::chrono::steady_clock::now();
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

    // No compiler wrapper needed — g++ (MinGW-w64) is on the system PATH,
    // and PyTorch Inductor finds it automatically.

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
            appendGraphPoint(cfg.dataDir, pt);
            // Log structured metric to file
            g_fileLog.logMetric(gen, pt.step, pt.train, pt.val, pt.lr, pt.accuracy,
                                pt.openingLoss, pt.middlegameLoss, pt.endgameLoss,
                                pt.hasVal, pt.hasAcc, pt.hasPhase);
            std::lock_guard<std::mutex> lk(g_st.mtx);
            if (pt.step > 0) g_st.curEpoch = pt.step;
        } else {
            // Try parsing phase loss line -> attach to last point
            TrainPoint phasePt;
            if (ParsePhaseLoss(ln, phasePt)) {
                std::lock_guard<std::mutex> lk(g_st.mtx);
                if (!g_st.pts.empty()) {
                    g_st.pts.back().openingLoss    = phasePt.openingLoss;
                    g_st.pts.back().middlegameLoss = phasePt.middlegameLoss;
                    g_st.pts.back().endgameLoss    = phasePt.endgameLoss;
                    g_st.pts.back().hasPhase       = true;
                    g_graph.dirty.store(true, std::memory_order_relaxed);  // FIX 7
                }
            }
        }
    }, g_st.stopFlag);
    {
        double trainingElapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - trainingStart).count());
        g_fileLog.logPhaseEnd("training", gen, ok, trainingElapsed);
    }
    if (ok) {
        std::error_code ec;
        fs::copy_file(outputWeights, genWeights, fs::copy_options::overwrite_existing, ec);
        if (ec) g_st.pushLog("[WARN] Failed to copy weights to " + genWeights.string() + ": " + ec.message());

        // Archive SWA weights if they exist
        fs::path swaWeights = assetsDir/"nnue_weights_swa.bin";
        fs::path swaGen     = assetsDir/("nnue_weights_swa_gen"+std::to_string(gen)+".bin");
        if (cfg.swa && fs::exists(swaWeights)) {
            std::error_code ec2;
            fs::copy_file(swaWeights, swaGen, fs::copy_options::overwrite_existing, ec2);
            if (ec2) g_st.pushLog("[WARN] Failed to archive SWA weights: " + ec2.message());
            else     g_st.pushLog("  SWA weights archived to " + swaGen.filename().string());
        }
    }
    return ok;
}

// ── Built-in ELO validation match (writes+runs a Python match script) ──
static void InternalEloMatch(const Config& cfg, int gen,
                             const std::string& newWtPath,
                             const std::string& prevWtPath,
                             int numGames, int depth)
{
    std::string d = exeDir();
    fs::path assetsDir = fs::path(d)/cfg.dataDir;
    // ARCHITECTURE FIX: Deduplicated elo_match.py and swa_match.py into uci_match.py
    fs::path matchScript = assetsDir / "uci_match.py";

    // FIX 6.30: Don't clobber user-customized scripts
    if (!fs::exists(matchScript)) {
    // Write a unified Python UCI match manager script (used by both ELO and SWA matches)
    {
        std::ofstream f(matchScript.string());
        if (!f.is_open()) {
            g_st.pushLog("[ELO] ERROR: Could not create match script at " + matchScript.string());
            return;
        }
        f << R"PY(#!/usr/bin/env python3
"""Unified UCI match manager for ELO/SWA validation with draw detection and opening variety."""
import subprocess, sys, math, random, threading

try:
    import chess
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--default-timeout=60", "python-chess"])  # Prevent hanging
    import chess

def _popen_kwargs():
    kw = {}
    if sys.platform == 'win32':
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        kw['startupinfo'] = si
        kw['creationflags'] = 0x08000000  # CREATE_NO_WINDOW
    return kw

def uci_engine(path, weights):
    p = subprocess.Popen([path, "--uci"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1,
                         **_popen_kwargs())
    p.stdin.write("uci\n"); p.stdin.flush()
    for line in p.stdout:
        if line.strip() == "uciok": break
    if weights:
        p.stdin.write(f"setoption name WeightsFile value {weights}\n")
    p.stdin.write("isready\n"); p.stdin.flush()
    for line in p.stdout:
        if line.strip() == "readyok": break
    return p

def random_opening(rng_plies=8):
    """Generate a random opening by playing rng_plies random moves from startpos."""
    board = chess.Board()
    for _ in range(rng_plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(rng_plies and random.choice(legal))
    return board

def read_bestmove(eng, timeout=30):
    """Read bestmove from engine with a timeout. Returns move string or None."""
    result = [None]
    def _read():
        for line in eng.stdout:
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                result[0] = parts[1] if len(parts) > 1 else None
                return
    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0]

def play_game(ew, eb, depth, board):
    """Play a game from the given board position. Returns 1.0=white wins, 0.0=black wins, 0.5=draw."""
    for e in [ew, eb]:
        e.stdin.write("ucinewgame\nisready\n"); e.stdin.flush()
        for line in e.stdout:
            if line.strip() == "readyok": break
    moves = [m.uci() for m in board.move_stack]
    while not board.is_game_over(claim_draw=True):
        if board.fullmove_number > 300:
            return 0.5
        eng = ew if board.turn == chess.WHITE else eb
        pos = "position startpos"
        if moves: pos += " moves " + " ".join(moves)
        eng.stdin.write(pos + "\n")
        eng.stdin.write(f"go depth {depth}\n"); eng.stdin.flush()
        bm = read_bestmove(eng, timeout=30)
        if not bm or bm == "(none)":
            return 0.0 if board.turn == chess.WHITE else 1.0
        try:
            move = chess.Move.from_uci(bm)
            if move not in board.legal_moves:
                return 0.0 if board.turn == chess.WHITE else 1.0
            board.push(move)
            moves.append(bm)
        except Exception:
            return 0.0 if board.turn == chess.WHITE else 1.0
    result = board.result(claim_draw=True)
    if result == "1-0": return 1.0
    if result == "0-1": return 0.0
    return 0.5

def main():
    # Usage: uci_match.py <engine> <new_wt> <prev_wt> <num_games> <depth> <mode>
    # mode: "elo" or "swa"
    engine_path, new_wt, prev_wt = sys.argv[1], sys.argv[2], sys.argv[3]
    num_games, depth = int(sys.argv[4]), int(sys.argv[5])
    mode = sys.argv[6] if len(sys.argv) > 6 else "elo"
    tag = "[ELO]" if mode == "elo" else "[SWA]"
    min_openings = 25 if mode == "elo" else 15
    wins = draws = losses = 0
    openings = [random_opening(random.randint(4, 8)) for _ in range(max(num_games // 2, min_openings))]
    for g in range(num_games):
        new_is_white = (g % 2 == 0)
        w_wt = new_wt if new_is_white else prev_wt
        b_wt = prev_wt if new_is_white else new_wt
        board = openings[g % len(openings)].copy()
        ew = uci_engine(engine_path, w_wt)
        eb = uci_engine(engine_path, b_wt)
        try:
            r = play_game(ew, eb, depth, board)
            ns = r if new_is_white else 1.0 - r
            if ns > 0.7: wins += 1
            elif ns < 0.3: losses += 1
            else: draws += 1
        except Exception as ex:
            print(f"{tag} Game {g+1} error: {ex}", flush=True)
            draws += 1
        finally:
            for e in [ew, eb]:
                try: e.stdin.write("quit\n"); e.stdin.flush(); e.wait(timeout=5)
                except: e.kill()
        if (g+1) % 10 == 0:
            print(f"{tag} {g+1}/{num_games}: +{wins} ={draws} -{losses}", flush=True)
    total = wins + draws + losses
    if total > 0:
        wr = (wins + 0.5 * draws) / total
        elo = -999 if wr <= 0.001 else (999 if wr >= 0.999 else int(-400 * math.log10(1/wr - 1)))
        if mode == "elo":
            ci = int(400 * math.sqrt(wr * (1-wr) / total) / (math.log(10) * wr * (1-wr) + 1e-9))
            print(f"{tag} Final: +{wins} ={draws} -{losses} ELO: {elo} +/- {ci}", flush=True)
        else:
            print(f"{tag} Final: +{wins} ={draws} -{losses} SWA_ELO: {elo}", flush=True)
    else:
        print(f"{tag} No games completed", flush=True)

if __name__ == "__main__":
    main()
)PY";
    }
    } // FIX 6.30: end clobber guard

    std::string enginePath = (fs::path(d)/cfg.exeName).string();
    std::wstring cmd = W(
        "py -3 \"" + matchScript.string() + "\""
        " \"" + enginePath + "\""
        " \"" + newWtPath + "\""
        " \"" + prevWtPath + "\""
        " " + std::to_string(numGames)
        + " " + std::to_string(depth)
        + " elo"
    );

    g_st.pushLog("[ELO] Running built-in match: Gen" + std::to_string(gen) +
                 " vs Gen" + std::to_string(gen-1) + " (" +
                 std::to_string(numGames) + " games)");
    RunProc(cmd, d, [&](const std::string& ln) {
        g_st.pushLog(ln);
        // Parse final line: "[ELO] Final: +W =D -L ELO: N +/- CI"
        if (ln.find("Final:") != std::string::npos && ln.find("ELO:") != std::string::npos) {
            int w=0,dr=0,l=0,elo=0;
            auto pPlus = ln.find('+'); auto pEq = ln.find('='); auto pMinus = ln.find('-', pEq > 0 ? pEq : 0);
            try { if (pPlus!=std::string::npos) w=std::stoi(ln.substr(pPlus+1)); } catch(...){}
            try { if (pEq!=std::string::npos) dr=std::stoi(ln.substr(pEq+1)); } catch(...){}
            try { if (pMinus!=std::string::npos) l=std::stoi(ln.substr(pMinus+1)); } catch(...){}
            auto ep = ln.find("ELO:");
            if (ep != std::string::npos) {
                size_t s = ep + 4;
                while (s < ln.size() && (ln[s]==' '||ln[s]=='+')) s++;
                try { elo = std::stoi(ln.substr(s)); } catch(...){}
            }
            { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.lastElo = elo; }
            g_fileLog.logElo(gen, elo, w, dr, l, "elo");
        } else {
            auto ep = ln.find("ELO:");
            if (ep != std::string::npos) {
                size_t s = ep + 4;
                while (s < ln.size() && (ln[s]==' '||ln[s]=='+')) s++;
                try { int e = std::stoi(ln.substr(s));
                    std::lock_guard<std::mutex> lk(g_st.mtx); g_st.lastElo = e; }
                catch(...) {}
            }
        }
    }, g_st.stopFlag);
}

static void EloVal(const Config& cfg, int gen) {
    if (!cfg.eloValidate || gen <= cfg.startGen + 1) return;
    std::string d = exeDir();
    fs::path assetsDir = fs::path(d)/cfg.dataDir;
    fs::path newWt  = assetsDir/("nnue_weights_gen"+std::to_string(gen)+".bin");
    fs::path prevWt = assetsDir/("nnue_weights_gen"+std::to_string(gen-1)+".bin");
    if (!fs::exists(newWt)||!fs::exists(prevWt)) return;

    g_st.setStatus("Gen "+std::to_string(gen)+": ELO validation");
    g_st.setPhase("elo");
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.phaseStart = std::chrono::steady_clock::now(); }  // FIX 6.1
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.pausedPhaseSec = 0; }
    g_fileLog.logPhaseStart("elo", gen, "games=" + std::to_string(cfg.eloGames));
    auto eloStart = std::chrono::steady_clock::now();

    // Try cutechess-cli first (preferred)
    fs::path cutechess = fs::path(d)/"cutechess"/"cutechess-cli.exe";
    if (fs::exists(cutechess)) {
        fs::path pgnOut = assetsDir/("validation_gen"+std::to_string(gen)+".pgn");
        std::string enginePath2 = (fs::path(d)/cfg.exeName).string();
        std::wstring cmd = W(
            "\"" + cutechess.string() + "\""
            " -engine name=Gen" + std::to_string(gen) + " cmd=\"" + enginePath2 + "\" arg=--uci"
            " option.WeightsFile=\"" + newWt.string() + "\""
            " -engine name=Gen" + std::to_string(gen-1) + " cmd=\"" + enginePath2 + "\" arg=--uci"
            " option.WeightsFile=\"" + prevWt.string() + "\""
            " -each proto=uci tc=1+0.1"
            " -rounds " + std::to_string(cfg.eloGames)
            + " -pgnout \"" + pgnOut.string() + "\""
            " -recover"
        );
        RunProc(cmd, d, [&](const std::string& ln){
            g_st.pushLog(ln);
            // cutechess-cli outputs: "Elo difference: 45.2 +/- 32.1, LOS: ..."
            auto p = ln.find("Elo difference:");
            if (p != std::string::npos) {
                size_t numStart = p + 15; // strlen("Elo difference:")
                while (numStart < ln.size() && (ln[numStart]==' '||ln[numStart]=='+')) numStart++;
                try {
                    double ed = std::stod(ln.substr(numStart));
                    std::lock_guard<std::mutex> lk(g_st.mtx);
                    g_st.lastElo = static_cast<int>(ed);
                } catch(...) {}
            }
            // Also check for "ELO:" from built-in format
            auto ep2 = ln.find("ELO:");
            if (ep2 != std::string::npos) {
                size_t s = ep2 + 4;
                while (s < ln.size() && (ln[s]==' '||ln[s]=='+')) s++;
                try { int e = std::stoi(ln.substr(s));
                    std::lock_guard<std::mutex> lk(g_st.mtx); g_st.lastElo = e; }
                catch(...) {}
            }
        }, g_st.stopFlag);
    } else {
        // Fallback: built-in Python match system (no cutechess-cli needed)
        g_st.pushLog("[ELO] cutechess-cli.exe not found — using built-in match system");
        InternalEloMatch(cfg, gen, newWt.string(), prevWt.string(), cfg.eloGames, cfg.depth);
    }
    {
        double eloElapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - eloStart).count());
        g_fileLog.logPhaseEnd("elo", gen, true, eloElapsed);
    }
}

// ── SWA Best-of-Two Selection ────────────────────────────────────
// Plays a quick match between best-val and SWA weights.
// If SWA wins, promotes SWA as the gen weights.
static void SwaBestOfTwo(const Config& cfg, int gen) {
    if (!cfg.swa) return;
    std::string d = exeDir();
    fs::path assetsDir = fs::path(d)/cfg.dataDir;
    fs::path genWt  = assetsDir/("nnue_weights_gen"+std::to_string(gen)+".bin");
    fs::path swaWt  = assetsDir/("nnue_weights_swa_gen"+std::to_string(gen)+".bin");
    if (!fs::exists(genWt) || !fs::exists(swaWt)) return;

    g_st.setStatus("Gen "+std::to_string(gen)+": SWA vs Best-Val match");
    g_st.setPhase("swa_match");
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.phaseStart = std::chrono::steady_clock::now(); }  // FIX 6.1
    { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.pausedPhaseSec = 0; }
    g_fileLog.logPhaseStart("swa_match", gen, "games=" + std::to_string(cfg.swaGames));
    auto swaStart = std::chrono::steady_clock::now();

    g_st.pushLog("[SWA] Running best-of-two: Best-Val vs SWA (" + std::to_string(cfg.swaGames) + " games)");

    // Reuse the built-in match infrastructure.
    // We capture the ELO result from the match output.
    int swaElo = 0;

    // ARCHITECTURE FIX: Reuse unified uci_match.py (written by InternalEloMatch) with --mode swa
    // Positive ELO means SWA is stronger.
    fs::path matchScript = assetsDir / "uci_match.py";
    // If uci_match.py doesn't exist yet (e.g. SWA called before ELO), write it now
    // FIX 6.30: Don't clobber user-customized scripts
    if (!fs::exists(matchScript)) {
        g_st.pushLog("[SWA] WARNING: uci_match.py not found — SWA match requires ELO match to run first or manual script placement");
        return;
    }

    std::string enginePath = (fs::path(d)/cfg.exeName).string();
    std::wstring cmd = W(
        "py -3 \"" + matchScript.string() + "\""
        " \"" + enginePath + "\""
        " \"" + swaWt.string() + "\""
        " \"" + genWt.string() + "\""
        " " + std::to_string(cfg.swaGames)
        + " " + std::to_string(cfg.depth)
        + " swa"
    );

    RunProc(cmd, d, [&](const std::string& ln) {
        g_st.pushLog(ln);
        auto ep = ln.find("SWA_ELO:");
        if (ep != std::string::npos) {
            size_t s = ep + 8;
            while (s < ln.size() && (ln[s]==' '||ln[s]=='+')) s++;
            try { swaElo = std::stoi(ln.substr(s)); } catch(...) {}
        }
    }, g_st.stopFlag);

    {
        double swaElapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - swaStart).count());
        g_fileLog.logPhaseEnd("swa_match", gen, true, swaElapsed);
        g_fileLog.logElo(gen, swaElo, 0, 0, 0, "swa");  // wins/draws/losses not tracked here
    }

    if (swaElo > 0) {
        // SWA won — promote SWA weights as the gen weights
        g_st.pushLog("[SWA] SWA wins by +" + std::to_string(swaElo) +
                     " ELO — promoting SWA weights for gen " + std::to_string(gen));
        std::error_code ec;
        fs::copy_file(swaWt, genWt, fs::copy_options::overwrite_existing, ec);
        if (ec) g_st.pushLog("[SWA] ERROR: Failed to promote SWA weights: " + ec.message());
    } else if (swaElo == 0) {
        g_st.pushLog("[SWA] Draw — keeping best-val weights for gen " + std::to_string(gen));
    } else {
        g_st.pushLog("[SWA] Best-val wins by " + std::to_string(-swaElo) +
                     " ELO — keeping best-val weights for gen " + std::to_string(gen));
    }
}

void PipelineThread(Config cfg) {
    g_st.stopFlag.store(false);
    g_proc.pauseFlag.store(false);
    g_proc.skipPhaseFlag.store(false);
    int firstGen = cfg.startGen + 1;
    int lastGen  = cfg.startGen + cfg.generations;
    { std::lock_guard<std::mutex> lk(g_st.mtx);
      g_st.running=true; g_st.curGen=0; g_st.totalGens=cfg.generations;
      g_st.pipelineStart = std::chrono::steady_clock::now();
      g_st.phaseStart = g_st.pipelineStart;  // FIX 6.1 (already under g_st.mtx)
      g_st.pausedPipelineSec = 0;
      g_st.pausedPhaseSec = 0;
      g_st.completedGens = 0;
      g_st.emaGenSec = 0.0;
      g_st.prevGenCompletedSec = 0;
      // Load persistent graph data, then truncate from firstGen onwards
      g_st.pts = loadGraphData(cfg.dataDir);
      g_st.pts.erase(
          std::remove_if(g_st.pts.begin(), g_st.pts.end(),
              [firstGen](const TrainPoint& p) { return p.gen >= firstGen; }),
          g_st.pts.end());
      g_graph.dirty.store(true, std::memory_order_relaxed); }
    // Rewrite the truncated graph data to disk (outside lock — file I/O)
    { std::lock_guard<std::mutex> lk(g_st.mtx);
      saveGraphData(cfg.dataDir, g_st.pts); }
    // FIX 6.4: Validate paths before building any command lines to prevent injection
    try {
        ValidatePathsForInjection(cfg.dataDir, cfg.exeName);
    } catch (const std::runtime_error& e) {
        g_st.pushLog(std::string("[ERR] ") + e.what() + " — aborting pipeline");
        { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.running = false; }
        PostMessageW(g_ui.hWnd, WM_USER+1, 0, 0);
        return;
    }
    g_st.pushLog("=== Pipeline start: "+std::to_string(cfg.generations)+" generations (gen "+std::to_string(firstGen)+" to "+std::to_string(lastGen)+") ===");

    // ── File logger: open log file and dump config ──
    {
        std::string logDir = exeDir() + "\\" + cfg.dataDir;
        std::string logPath = g_fileLog.open(logDir);
        if (!logPath.empty())
            g_st.pushLog("[LOG] Writing structured log to: " + logPath);

        g_fileLog.logPipelineStart(firstGen, lastGen, cfg.generations);

        // Dump full config as key=value pairs
        std::ostringstream cfgDump;
        cfgDump << "generations=" << cfg.generations
                << " startGen=" << cfg.startGen
                << " gamesPerGen=" << cfg.gamesPerGen
                << " epochsPerGen=" << cfg.epochsPerGen
                << " batchSize=" << cfg.batchSize
                << " lr=" << cfg.lr
                << " weightDecay=" << cfg.weightDecay
                << " dropout=" << cfg.dropout
                << " labelSmooth=" << cfg.labelSmooth
                << " gradAccum=" << cfg.gradAccum
                << " warmupSteps=" << cfg.warmupSteps
                << " drawWeight=" << cfg.drawWeight
                << " mateBoost=" << cfg.mateBoost
                << " splRatio=" << cfg.splRatio
                << " frcMix=" << cfg.frcMix
                << " replayWindow=" << cfg.replayWindow
                << " replayDecay=" << cfg.replayDecay
                << " workers=" << cfg.workers
                << " depth=" << cfg.depth
                << " mixedDepthLow=" << cfg.mixedDepthLow
                << " mixedDepthRatio=" << cfg.mixedDepthRatio
                << " depthShuffle=" << (cfg.depthShuffle ? "true" : "false")
                << " depthShuffleBias=" << cfg.depthShuffleBias
                << " maxPositions=" << cfg.maxPositions
                << " earlyStop=" << cfg.earlyStop
                << " cosineLr=" << (cfg.cosineLr ? "true" : "false")
                << " cosineT0=" << cfg.cosineT0
                << " swa=" << (cfg.swa ? "true" : "false")
                << " swaStart=" << cfg.swaStart
                << " drawPct=" << cfg.drawPct
                << " wdlAlpha=" << cfg.wdlAlpha
                << " wdlDrawElo=" << cfg.wdlDrawElo
                << " eloValidate=" << (cfg.eloValidate ? "true" : "false")
                << " eloGames=" << cfg.eloGames
                << " swaGames=" << cfg.swaGames
                << " overfitDetect=" << (cfg.overfitDetect ? "true" : "false")
                << " resignCp=" << cfg.resignCp
                << " contemptCp=" << cfg.contemptCp
                << " maxPlies=" << cfg.maxPlies
                << " drawCp=" << cfg.drawCp
                << " exeName=" << cfg.exeName
                << " pyScript=" << cfg.pyScript
                << " dataDir=" << cfg.dataDir
                << " openingTemp=" << cfg.openingTemp
                << " openingPlies=" << cfg.openingPlies
                << " softmaxPlies=" << cfg.softmaxPlies
                << " softmaxTemp=" << cfg.softmaxTemp
                << " rootNoiseEps=" << cfg.rootNoiseEps
                << " recordMinPly=" << cfg.recordMinPly
                << " recordMaxEval=" << cfg.recordMaxEval
                << " resignCount=" << cfg.resignCount
                << " drawCount=" << cfg.drawCount
                << " drawMinPly=" << cfg.drawMinPly
                << " drawAdjMoves=" << cfg.drawAdjMoves
                << " drawAdjThreshold=" << cfg.drawAdjThreshold
                << " drawAdjMinMove=" << cfg.drawAdjMinMove;
        g_fileLog.logConfig(cfgDump.str());
    }

    int consecutiveOverfit = 0;
    double prevBestVal = -1.0;
    for (int gen=firstGen; gen<=lastGen && !g_st.stopFlag.load(); gen++) {
        { std::lock_guard<std::mutex> lk(g_st.mtx); g_st.curGen=gen-firstGen; }
        g_st.pushLog("--- Generation "+std::to_string(gen)+" ---");
        if (!SelfPlay(cfg, gen))  { g_st.pushLog("[ERR] Self-play failed gen "+std::to_string(gen)); break; }
        PlayMp3("snd_sp", cfg.dataDir, "ding_selfplay_complete.mp3");
        if (g_proc.skipPhaseFlag.load()) { g_st.pushLog("[SKIP] Self-play skipped — using partial data"); g_proc.skipPhaseFlag.store(false); }
        if (g_st.stopFlag.load()) break;
        if (!Training(cfg, gen))  { g_st.pushLog("[ERR] Training failed gen "+std::to_string(gen)); break; }
        if (g_proc.skipPhaseFlag.load()) { g_st.pushLog("[SKIP] Training skipped — using best checkpoint"); g_proc.skipPhaseFlag.store(false); }
        if (g_st.stopFlag.load()) break;
        // SWA best-of-two: if SWA weights beat best-val, promote them
        SwaBestOfTwo(cfg, gen);
        if (g_proc.skipPhaseFlag.load()) { g_proc.skipPhaseFlag.store(false); }
        if (g_st.stopFlag.load()) break;
        // Persist this gen's best val_loss so "Load Best" works after restart
        double bestValForGen = 1e9;
        {   std::lock_guard<std::mutex> lk(g_st.mtx);
            for (auto& p : g_st.pts) {
                if (p.gen == gen && p.hasVal && p.val < bestValForGen)
                    bestValForGen = p.val;
            }
            // Stamp the final NPS from self-play onto all points for this gen
            if (g_st.curNps > 0.0) {
                for (auto& p : g_st.pts) {
                    if (p.gen == gen) { p.nps = g_st.curNps; p.hasNps = true; }
                }
                g_st.curNps = 0.0;  // reset for next gen
            }
            if (bestValForGen < 1e9) saveGenStat(gen, bestValForGen, cfg.dataDir);
            saveGraphData(cfg.dataDir, g_st.pts);
        }
        // ── Overfitting detection ──
        if (cfg.overfitDetect && bestValForGen < 1e9) {
            if (prevBestVal > 0.0) {
                if (bestValForGen > prevBestVal * 1.005) {
                    consecutiveOverfit++;
                    g_st.pushLog("[WARN] Val loss increased: " + dbl2s(prevBestVal, 6) +
                                 " -> " + dbl2s(bestValForGen, 6) +
                                 " (" + std::to_string(consecutiveOverfit) + " consecutive)");
                    if (consecutiveOverfit >= 5) {
                        g_st.pushLog("[WARN] Severe overfitting — stopping pipeline early (5 consecutive val-loss increases)");
                        g_st.stopFlag.store(true);
                    }
                } else {
                    consecutiveOverfit = 0;
                }
            }
            prevBestVal = bestValForGen;
        }
        // Export graph snapshot as PNG into training_progress folder
        SaveGraphPng(cfg.dataDir);
        EloVal(cfg, gen);
        if (g_proc.skipPhaseFlag.load()) { g_st.pushLog("[SKIP] ELO validation skipped"); g_proc.skipPhaseFlag.store(false); }
        long long genSecRaw = 0;  // BUG-2 fix: declared outside lock scope for logging below
        { std::lock_guard<std::mutex> lk(g_st.mtx);
          std::ostringstream ss; ss<<std::fixed<<std::setprecision(5);
          ss<<"Gen "<<gen<<" done | train="<<g_st.lastTrain<<" val="<<g_st.lastVal<<" ELO="<<g_st.lastElo;
          g_st.log.push_back(ss.str());
          g_st.lastGenCompletedSec = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now() - g_st.pipelineStart).count() - g_st.pausedPipelineSec;
          if (g_st.lastGenCompletedSec < 0) g_st.lastGenCompletedSec = 0;
          // Compute this gen's duration and update EMA  (BUG-2 fix: capture before overwrite)
          {
              genSecRaw = g_st.lastGenCompletedSec - g_st.prevGenCompletedSec;
              if (genSecRaw < 0) genSecRaw = 0;
              if (g_st.completedGens == 0) {
                  // First gen: seed EMA with actual duration
                  g_st.emaGenSec = (double)genSecRaw;
              } else {
                  // EMA with alpha=0.4 — adapts quickly to pace changes
                  // After gen 1, heavily discount it if gen 2 is much faster (bootstrap effect)
                  constexpr double alpha = 0.4;
                  g_st.emaGenSec = alpha * (double)genSecRaw + (1.0 - alpha) * g_st.emaGenSec;
              }
              g_st.prevGenCompletedSec = g_st.lastGenCompletedSec;
          }
          g_st.completedGens++; }
        // Log structured generation summary  (BUG-2 fix: reuse genSecRaw instead of recomputing)
        {
            std::lock_guard<std::mutex> lk(g_st.mtx);
            double genSec = static_cast<double>(genSecRaw);
            if (genSec < 0) genSec = 0;
            g_fileLog.logGenSummary(gen, g_st.lastTrain, g_st.lastVal, g_st.lastElo, genSec);
        }
        PlayMp3("snd_gen", cfg.dataDir, "ding_gen_complete.mp3");
    }
    {
        std::lock_guard<std::mutex> lk(g_st.mtx);
        g_st.pipelineTotalSec = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - g_st.pipelineStart).count() - g_st.pausedPipelineSec;
        if (g_st.pipelineTotalSec < 0) g_st.pipelineTotalSec = 0;
        g_st.running = false;
    }
    g_st.setStatus(g_st.stopFlag.load() ? "Stopped." : "Pipeline complete!");
    g_st.pushLog("=== Pipeline "+std::string(g_st.stopFlag.load()?"stopped":"complete")+" ===");
    // Final graph PNG snapshot
    SaveGraphPng(cfg.dataDir);

    // ── File logger: log pipeline end and close ──
    {
        long long totalSec = 0;
        int completed = 0;
        { std::lock_guard<std::mutex> lk(g_st.mtx); totalSec = g_st.pipelineTotalSec; completed = g_st.completedGens; }
        g_fileLog.logPipelineEnd(g_st.stopFlag.load(), (double)totalSec, completed);
        g_fileLog.close();
    }

    if (!g_st.stopFlag.load())
        PlayMp3("snd_pipe", cfg.dataDir, "ding_pipeline_finished.mp3");
    PostMessage(g_ui.hWnd, WM_USER+1, 0, 0);
}

