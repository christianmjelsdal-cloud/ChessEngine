#include "UCI.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <filesystem>  // FIX 13.8: validate output directory

// ============================================================
//  UCI Protocol Implementation
//  Reference: https://backscattering.de/chess/uci/
// ============================================================

UCI::UCI() : nnue_(std::make_unique<NNUE::Network>()) {
    Engine::initZobrist();
    // Load NNUE weights if available
    if (nnue_->loadWeights(assetPath("assets/nnue_weights.bin"))) {
        nnue_->releaseFloatWeights();  // free ~160 MB — only quantized weights needed for search
        engine_.setNNUE(nnue_.get());
        nnueLoaded_ = true;
    }
}

UCI::~UCI() {
    if (searchThread_.joinable()) {
        engine_.stop();
        searchThread_.join();
    }
    Syzygy::free();
}

// ------------------------------------------------------------
//  Main loop
// ------------------------------------------------------------
void UCI::loop() {
    std::string line;
    while (!quit_ && std::getline(std::cin, line)) {
        // FIX 3.15: O(n) whitespace trim (was O(n²) front-erase loop)
        {
            auto first = std::find_if(line.begin(), line.end(),
                [](unsigned char c) { return !std::isspace(c); });
            line.erase(line.begin(), first);
        }
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
            line.pop_back();
        if (line.empty()) continue;

        // Extract the first token
        std::string token;
        {
            std::istringstream iss(line);
            iss >> token;
        }

        if      (token == "uci")        cmdUCI();
        else if (token == "isready")    cmdIsReady();
        else if (token == "ucinewgame") cmdNewGame();
        else if (token == "position")   cmdPosition(line);
        else if (token == "go")         cmdGo(line);
        else if (token == "stop")       cmdStop();
        else if (token == "ponderhit") cmdPonderHit();
        else if (token == "quit")       cmdQuit();
        else if (token == "generate")   cmdGenerate(line);
        else if (token == "setoption") {
            // FIX 3.3: Stop search before modifying engine state
            if (searchThread_.joinable()) {
                engine_.stop();
                searchThread_.join();
            }
            // Parse: setoption name <name> [value <value>]
            std::istringstream ss(line);
            std::string tmp, name, value;
            ss >> tmp; // "setoption"
            ss >> tmp; // "name"
            std::getline(ss, name, '\0');
            auto vpos = name.find(" value ");
            if (vpos != std::string::npos) {
                value = name.substr(vpos + 7);
                name  = name.substr(0, vpos);
            }
            // trim leading space
            while (!name.empty() && name[0] == ' ') name.erase(0, 1);
            // Handle known options
            if (name == "Hash") {
                // FIX M-4: Validate before parsing to avoid crash on empty/invalid value
                try {
                    if (value.empty()) continue;
                    size_t mb = static_cast<size_t>(std::clamp(std::stoi(value), 1, 4096));  // FIX 13.14: enforce max
                    // TIER-2 FIX #7: Packed TTEntry is 12 bytes (was ~24)
                    size_t entries = (mb * 1024ULL * 1024ULL) / Engine::TT_ENTRY_BYTES;
                    engine_.resizeTT(entries);
                    if (nnueLoaded_ && nnue_) engine_.setNNUE(nnue_.get());
                } catch (const std::exception&) {
                    // Silently ignore invalid Hash value (per UCI spec)
                }
            }
            else if (name == "Threads") {
                try {
                    if (value.empty()) continue;
                    int n = std::max(1, std::stoi(value));
                    engine_.setThreadCount(n);
                } catch (const std::exception&) {
                    // Silently ignore invalid Threads value
                }
            }
            else if (name == "MultiPV") {
                try {
                    if (value.empty()) continue;
                    int n = std::max(1, std::stoi(value));
                    engine_.setMultiPV(n);
                } catch (const std::exception&) {
                    // Silently ignore invalid MultiPV value
                }
            }
            else if (name == "Contempt") {
                try {
                    if (value.empty()) continue;
                    int cp = std::stoi(value);
                    engine_.setContempt(cp);
                } catch (const std::exception&) {
                    // Silently ignore invalid Contempt value
                }
            }
            else if (name == "Ponder") {
                ponderEnabled_ = (value == "true");
            }
            else if (name == "SyzygyPath") {
                if (Syzygy::init(value)) {
                    std::cerr << "info string Syzygy tablebases loaded: "
                              << Syzygy::maxPieces() << "-piece" << std::endl;
                } else {
                    std::cerr << "info string Syzygy tablebases not found at: "
                              << value << std::endl;
                }
            }
            // Unknown options are silently ignored (per UCI spec)
        }
        // Silently ignore unknown commands (per UCI spec)
    }
}

// ------------------------------------------------------------
//  uci
// ------------------------------------------------------------
void UCI::cmdUCI() {
    // AUDIT FIX 3: Added version string for debugging and tournament ID
    std::cout << "id name ChessEngine NNUE v1.0" << std::endl;
    std::cout << "id author Christian Mjelsdal" << std::endl;
    std::cout << "option name Hash type spin default 96 min 1 max 4096" << std::endl;
    std::cout << "option name Threads type spin default 1 min 1 max 256" << std::endl;
    std::cout << "option name MultiPV type spin default 1 min 1 max 500" << std::endl;
    std::cout << "option name Contempt type spin default 0 min -100 max 100" << std::endl;
    std::cout << "option name Ponder type check default true" << std::endl;
    std::cout << "option name SyzygyPath type string default " << std::endl;
    std::cout << "uciok" << std::endl;
}

// ------------------------------------------------------------
//  isready
// ------------------------------------------------------------
void UCI::cmdIsReady() {
    // Auto-detect Syzygy tablebases if not already loaded
    if (Syzygy::maxPieces() == 0) {
        std::string tbPath = assetPath("Syzygy345");
        if (std::filesystem::is_directory(tbPath)) {
            if (Syzygy::init(tbPath)) {
                std::cerr << "info string Syzygy tablebases auto-loaded ("
                          << Syzygy::maxPieces() << "-piece) from: "
                          << tbPath << std::endl;
            }
        }
    }
    std::cout << "readyok" << std::endl;
}

// ------------------------------------------------------------
//  ucinewgame
// ------------------------------------------------------------
void UCI::cmdNewGame() {
    // Wait for any running search to finish
    if (searchThread_.joinable()) {
        engine_.stop();
        searchThread_.join();
    }
    board_ = Board();
    board_.setStartingPosition();
    positionHistory_.clear();
    engine_.clearSearchState();  // prevent stale TT/killers/history leaking into next game
}

// ------------------------------------------------------------
//  position startpos [moves ...]
//  position fen <fen> [moves ...]
// ------------------------------------------------------------
void UCI::cmdPosition(const std::string& line) {
    // Stop any running search before modifying board state
    if (searchThread_.joinable()) {
        engine_.stop();
        searchThread_.join();
    }

    std::istringstream iss(line);
    std::string token;
    iss >> token; // "position"
    iss >> token; // "startpos" or "fen"

    positionHistory_.clear();

    if (token == "startpos") {
        board_ = Board();
        board_.setStartingPosition();
        // UCI spec requires "moves" keyword between position and move list.
        // Some GUIs omit it; we handle both cases below (see FIX 13.4).
        iss >> token;
    }
    else if (token == "fen") {
        // Read the 6 FEN fields
        std::string fen;
        int fenFields = 0;
        for (int i = 0; i < 6 && iss >> token; ++i) {
            if (i > 0) fen += " ";
            fen += token;
            fenFields++;
        }
        if (fenFields < 6) {
            // FIX 13.3: Reject incomplete FEN (must have exactly 6 fields)
            std::cerr << "info string Error: FEN must have 6 fields, got " << fenFields << std::endl;
            return;
        }
        // AUDIT FIX C-1: Use Board's FEN parser (which also updates king squares,
        // hash, nonPawnMaterial, phase) instead of the duplicate UCI::parseFEN
        board_.setFromFEN(fen);
        // Consume optional "moves" token
        iss >> token;
    }

    // FIX N-2: Build position history during UCI move parsing for threefold repetition
    // Always include the initial position hash before any moves are applied
    positionHistory_.push_back(Engine::computeHash(board_));

    // FIX 13.4: Apply moves — accept both "moves e2e4 ..." and "e2e4 ..." (without keyword)
    if (token == "moves") {
        while (iss >> token) {
            Move m = parseMove(token);
            board_.applyMove(m);
            board_.recomputeBitboards();
            positionHistory_.push_back(Engine::computeHash(board_));
        }
    } else if (!token.empty() && token.size() >= 4 && token.size() <= 5) {
        // Possibly a move without the "moves" keyword — try to parse it
        Move m = parseMove(token);
        if (m.from.isValid()) {
            board_.applyMove(m);
            board_.recomputeBitboards();
            positionHistory_.push_back(Engine::computeHash(board_));
            while (iss >> token) {
                m = parseMove(token);
                board_.applyMove(m);
                board_.recomputeBitboards();
                positionHistory_.push_back(Engine::computeHash(board_));
            }
        }
    }
}

// ------------------------------------------------------------
//  go [depth N] [movetime N] [wtime N] [btime N] [winc N] [binc N]
// ------------------------------------------------------------
void UCI::cmdGo(const std::string& line) {
    // Wait for any previous search to finish
    if (searchThread_.joinable()) {
        engine_.stop();
        searchThread_.join();
    }

    std::istringstream iss(line);
    std::string token;
    iss >> token; // "go"

    int depth = 64;
    int moveTime = 0;
    int wtime = 0, btime = 0, winc = 0, binc = 0;
    int movestogo = 0;
    bool wtimeSet = false, btimeSet = false;  // FIX 3.5: track which time params were set
    bool infinite = false;
    bool depthSpecified = false;
    bool ponder = false;

    while (iss >> token) {
        if      (token == "depth")     { if (!(iss >> depth)) { depth = 64; } else { depthSpecified = true; } }
        else if (token == "movetime")  { if (!(iss >> moveTime)) moveTime = 0; }
        else if (token == "wtime")     { if (!(iss >> wtime)) wtime = 0; wtimeSet = true; }
        else if (token == "btime")     { if (!(iss >> btime)) btime = 0; btimeSet = true; }
        else if (token == "winc")      { if (!(iss >> winc)) winc = 0; }
        else if (token == "binc")      { if (!(iss >> binc)) binc = 0; }
        else if (token == "movestogo") { if (!(iss >> movestogo)) movestogo = 0; }
        else if (token == "infinite")  infinite = true;
        else if (token == "ponder")    ponder = true;
    }

    // Calculate time for this move
    int computedSoftMs = 999999999, computedHardMs = 999999999;  // FIX 13.1: default to infinite (was 5s)
    if (moveTime > 0) {
        computedSoftMs = computedHardMs = moveTime;
    }
    else if (wtimeSet || btimeSet) {  // FIX 3.5: only enter time allocation if time params were explicitly set
        // FIX 3.5: Use only the side-to-move's time if it was explicitly set
        int myTime = (board_.turn == Color::White) ? (wtimeSet ? wtime : 0) : (btimeSet ? btime : 0);
        int myInc  = (board_.turn == Color::White) ? winc  : binc;
        int allocatedMs;
        if (movestogo > 0) {
            // Use movestogo for more accurate time allocation
            allocatedMs = myTime / (movestogo + 1) + myInc;
        } else {
            // Simple time management: use 1/20 of remaining time + increment
            allocatedMs = myTime / 20 + myInc / 2;
        }
        // FIX 13.10: When time is very low, use a smaller minimum to avoid time loss
        int minMs = std::max(1, std::min(100, myTime / 2));
        allocatedMs = std::max(minMs, std::min(allocatedMs, myTime * 4 / 5));
        // AUDIT FIX S-3: Use soft/hard split so the engine can extend
        // search when the best move is unstable.
        // Soft = allocated time (target), Hard = 2.5× (absolute max, capped at 80% clock)
        computedSoftMs = allocatedMs;
        computedHardMs = std::min(allocatedMs * 5 / 2, myTime * 4 / 5);
        computedHardMs = std::max(computedHardMs, computedSoftMs);  // hard must be >= soft
    }
    else if (infinite || depthSpecified) {
        computedSoftMs = computedHardMs = 999999999; // effectively infinite
    }

    // TIER-2 FIX #8: Ponder mode — save real time limits and set infinite
    if (ponder) {
        ponderSoftMs_ = computedSoftMs;
        ponderHardMs_ = computedHardMs;
        isPondering_ = true;
        engine_.startPonder();
        engine_.setTimeLimit(999999999);  // infinite during ponder
    } else {
        isPondering_ = false;
        engine_.setTimeLimits(computedSoftMs, computedHardMs);
    }

    // Set position history for repetition detection
    engine_.setPositionHistory(positionHistory_);

    // Capture side-to-move before launching search thread to avoid data race
    Color sideToMove = board_.turn;

    // Wire up per-depth info callback for UCI output
    // NOTE: multiPV is read live via engine_.getMultiPV() rather than captured by value,
    // so if changed via setoption during search the info callback reflects the new value.
    // Under normal UCI protocol, setoption should not be sent during search.
    engine_.onInfoCallback = [this, sideToMove](int depth, int eval, uint64_t nodes,
                                    uint64_t nps, int64_t elapsed,
                                    const std::string& pv, int pvIndex) {
        // UCI protocol: score cp must be from side-to-move perspective.
        // Engine stores liveEval_ as white-perspective, so convert here.
        int sideToMoveEval = (sideToMove == Color::White) ? eval : -eval;
        {
            std::lock_guard<std::mutex> lk(coutMutex_);
            std::cout << "info depth " << depth;
            { int mpv = engine_.getMultiPV(); if (mpv > 1) std::cout << " multipv " << pvIndex; }  // FIX 13.2: read current multiPV
            std::cout << " score cp " << sideToMoveEval
                      << " nodes " << nodes
                      << " nps " << nps
                      << " time " << elapsed;
            { uint64_t tbh = engine_.getTBHits(); if (tbh > 0) std::cout << " tbhits " << tbh; }
            if (!pv.empty()) std::cout << " pv " << pv;
            std::cout << std::endl;
        }
    };

    // AUDIT FIX 13: THREAD SAFETY PROTOCOL
    // The search thread accesses engine_ and a copy of board_.
    // cmdGo() and cmdPosition() both call engine_.stop() + searchThread_.join()
    // before mutating state, ensuring no concurrent access. This is safe under
    // normal UCI protocol usage. Rapid-fire commands from a buggy GUI could
    // theoretically race, but the join() guard prevents it in practice.
    // Run search in a background thread so "stop" works
    searchThread_ = std::thread([this, depth, sideToMove]() {
        Board searchBoard = board_; // copy
        Move best = engine_.getBestMove(searchBoard, depth);

        // FIX 3.1: Capture all engine results atomically immediately after
        // getBestMove() returns, before the main thread can process a new
        // command that modifies engine state.
        int d = engine_.getLastDepth();
        int whiteEval = engine_.getLiveEval();
        std::vector<Move> pv = engine_.getPV();  // COPY, not reference
        Move ponderMove = ponderEnabled_ ? engine_.getPonderMove() : Move{};

        // Convert to side-to-move perspective for UCI protocol
        int eval = (sideToMove == Color::White) ? whiteEval : -whiteEval;

        std::string pvStr;
        for (const auto& m : pv) {
            if (!pvStr.empty()) pvStr += " ";
            pvStr += moveToUCI(m);
        }

        {
            // FIX 3.13: This post-search info line intentionally duplicates the last
            // per-depth callback output. It is kept because the callback fires during
            // search and may be missed by some GUIs if "bestmove" arrives too quickly.
            // This ensures the final depth/score/PV is always reported just before bestmove.
            std::lock_guard<std::mutex> lk(coutMutex_);
            std::cout << "info depth " << d;
            // FIX 3.14: removed hardcoded "multipv 1" — final info line reports best PV only
            std::cout << " score cp " << eval;
            if (!pvStr.empty())
                std::cout << " pv " << pvStr;
            std::cout << std::endl;

            // TIER-2 FIX #8: Include ponder move if available
            std::cout << "bestmove " << moveToUCI(best);
            if (ponderEnabled_ && ponderMove.from.isValid() && ponderMove.to.isValid()) {
                std::cout << " ponder " << moveToUCI(ponderMove);
            }
            std::cout << std::endl;
        }
    });
}

// ------------------------------------------------------------
//  ponderhit — TIER-2 FIX #8
// ------------------------------------------------------------
void UCI::cmdPonderHit() {
    if (isPondering_) {
        isPondering_ = false;
        engine_.ponderHit(ponderSoftMs_, ponderHardMs_);
    }
}

// ------------------------------------------------------------
//  stop
// ------------------------------------------------------------
void UCI::cmdStop() {
    isPondering_ = false;  // TIER-2 FIX #8: clear ponder state on stop
    engine_.stop();
    if (searchThread_.joinable())
        searchThread_.join();
}

// ------------------------------------------------------------
//  quit
// ------------------------------------------------------------
void UCI::cmdQuit() {
    quit_ = true;
    engine_.stop();
    if (searchThread_.joinable())
        searchThread_.join();
}

// ------------------------------------------------------------
//  generate games N [movetime Ms] [workers W] [output path] [maxplies N]
//
//  Runs native C++ self-play and writes a binary training data file.
//  Blocks until all games are complete (no background thread).
//  Time-based search: each move is given `movetime` milliseconds (default 200ms).
//  Example: generate games 1000 movetime 200 workers 12 output assets/selfplay_v1.bin
// ------------------------------------------------------------
void UCI::cmdGenerate(const std::string& line) {
    // Wait for any running search to finish first
    if (searchThread_.joinable()) {
        engine_.stop();
        searchThread_.join();
    }

    SelfPlayGen::Config cfg;

    std::istringstream iss(line);
    std::string token;
    iss >> token;  // consume "generate"

    try {
        while (iss >> token) {
            if      (token == "games"        && iss >> token) cfg.games           = std::stoi(token);
            else if (token == "movetime"     && iss >> token) cfg.moveTimeMs      = std::stoi(token);
            else if (token == "workers"      && iss >> token) cfg.workers         = std::stoi(token);
            else if (token == "output"       && iss >> token) cfg.outputPath      = token;
            else if (token == "maxplies"     && iss >> token) cfg.maxPlies        = std::stoi(token);
            else if (token == "depth"        && iss >> token) cfg.searchDepth     = std::stoi(token);
            else if (token == "openings"     && iss >> token) cfg.openingsFile    = token;
            // Phase 1A: Softmax diversity parameters
            else if (token == "openingtemp"  && iss >> token) cfg.openingTemp     = std::stof(token);
            else if (token == "softmaxplies" && iss >> token) cfg.softmaxPlies    = std::stoi(token);
            else if (token == "softmaxtemp"  && iss >> token) cfg.softmaxTemp     = std::stof(token);
            // Phase 1B/1C: Adjudication parameters
            else if (token == "contempt"     && iss >> token) cfg.contemptCp      = std::stoi(token);
            else if (token == "resigncp"     && iss >> token) cfg.resignCp        = std::stoi(token);
            else if (token == "resigncount"  && iss >> token) cfg.resignCount     = std::stoi(token);
            else if (token == "drawcp"       && iss >> token) cfg.drawCp          = std::stoi(token);
            else if (token == "drawcount"    && iss >> token) cfg.drawCount       = std::stoi(token);
            else if (token == "drawminply"   && iss >> token) cfg.drawMinPly      = std::stoi(token);
            else if (token == "openingplies" && iss >> token) cfg.openingPlies    = std::stoi(token);
            // Phase 2B: Root move noise
            else if (token == "rootnoise"    && iss >> token) cfg.rootNoiseEps    = std::stof(token);
            // Phase 3A: Position filtering
            else if (token == "recordminply" && iss >> token) cfg.recordMinPly    = std::stoi(token);
            else if (token == "recordmaxeval"&& iss >> token) cfg.recordMaxEval   = std::stoi(token);
        }
    } catch (const std::exception&) {
        std::cout << "info string Invalid numeric argument" << std::endl;
        return;
    }

    // FIX 13.8: Validate output directory exists before starting generation
    if (!cfg.outputPath.empty()) {
        std::filesystem::path outDir = std::filesystem::path(cfg.outputPath).parent_path();
        if (!outDir.empty() && !std::filesystem::is_directory(outDir)) {
            std::cout << "info string Error: output directory does not exist: "
                      << outDir.string() << std::endl;
            return;
        }
    }

    SelfPlayGen gen(nnue_.get());
    gen.generate(cfg);
}

// AUDIT FIX C-8: Removed dead UCI::parseFEN method — cmdPosition now uses board_.setFromFEN()

// ============================================================
//  Move parsing / formatting
// ============================================================
Square UCI::parseSquare(const std::string& s) {
    if (s.size() < 2) return {-1, -1};
    int col  = s[0] - 'a';
    int rank = s[1] - '1';
    if (col < 0 || col > 7 || rank < 0 || rank > 7) return {-1, -1};
    return {rank, col};
}

std::string UCI::squareToUCI(const Square& sq) {
    std::string s;
    s += static_cast<char>('a' + sq.col);
    s += static_cast<char>('1' + sq.rank);
    return s;
}

Move UCI::parseMove(const std::string& uciMove) const {
    if (uciMove.size() < 4) return Move{};
    Move m;
    m.from = parseSquare(uciMove.substr(0, 2));
    m.to   = parseSquare(uciMove.substr(2, 2));

    // Promotion piece
    if (uciMove.size() == 5) {
        switch (uciMove[4]) {
            case 'q': m.promotion = PieceType::Queen;  break;
            case 'r': m.promotion = PieceType::Rook;   break;
            case 'b': m.promotion = PieceType::Bishop; break;
            case 'n': m.promotion = PieceType::Knight; break;
        }
    }
    else {
        // Auto-detect promotion: pawn reaching last rank
        Piece p = board_.getPiece(m.from);
        if (p.type == PieceType::Pawn) {
            if ((p.color == Color::White && m.to.rank == 7) ||
                (p.color == Color::Black && m.to.rank == 0)) {
                m.promotion = PieceType::Queen; // default to queen
            }
        }
    }

    return m;
}

std::string UCI::moveToUCI(const Move& m) {
    std::string s = squareToUCI(m.from) + squareToUCI(m.to);
    if (m.promotion != PieceType::None) {
        switch (m.promotion) {
            case PieceType::Queen:  s += 'q'; break;
            case PieceType::Rook:   s += 'r'; break;
            case PieceType::Bishop: s += 'b'; break;
            case PieceType::Knight: s += 'n'; break;
            default: break;
        }
    }
    return s;
}


