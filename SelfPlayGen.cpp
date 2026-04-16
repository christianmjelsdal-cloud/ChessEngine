#include "SelfPlayGen.h"
#ifdef _WIN32
#include <windows.h>   // CreateThread, WaitForSingleObject, CloseHandle
#endif
#ifndef _WIN32
#include <unistd.h>  // 9.1: write()/STDERR_FILENO for signal-safe handler
#endif

// 9.3: Binary format requires little-endian (matching train_nnue.py struct.unpack)
// 9.3: Binary format requires little-endian (matching train_nnue.py struct.unpack)
// Use platform check instead of std::endian (C++20) for broader compiler support
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
static_assert(false, "SelfPlayGen binary format requires little-endian architecture");
#elif defined(_M_IX86) || defined(_M_X64) || defined(_M_ARM64) || defined(__x86_64__) || defined(__i386__)
// Known little-endian platforms — OK
#else
#include <bit>
static_assert(std::endian::native == std::endian::little,
    "SelfPlayGen binary format requires little-endian architecture");
#endif
#include "MoveGen.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include <iomanip>
#include <csignal>
#include <cstring>
#include <unordered_map>

// =============================================================================
//  Graceful shutdown on Ctrl+C
// =============================================================================

// FIX M-2: Use volatile sig_atomic_t for async-signal-safe flag
static volatile sig_atomic_t g_stopRequested = 0;

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
static BOOL (WINAPI *g_prevCtrlHandler)(DWORD) = nullptr;
#else
static void (*g_prevSigintHandler)(int) = SIG_DFL;
#endif

#ifdef _WIN32
static BOOL WINAPI ctrlHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
        g_stopRequested = 1;
        std::cerr << "\n[SelfPlay] Ctrl+C received - finishing in-progress games and saving...\n";
        std::cerr.flush();
        return TRUE;  // handled, don't terminate
    }
    return FALSE;
}
#else
static void sigintHandler(int sig) {
    g_stopRequested = 1;
    const char msg[] = "\n[SelfPlay] Ctrl+C received - finishing in-progress games and saving...\n";
    (void)write(STDERR_FILENO, msg, sizeof(msg) - 1);
    (void)sig;  // suppress unused parameter warning
}
#endif

// =============================================================================
//  Feature encoding helpers
//  Must match generate_selfplay.py PIECE_TO_INDEX exactly.
// =============================================================================

static int pieceTypeOffset(PieceType pt, Color c) {
    int base = (c == Color::Black) ? 6 : 0;
    switch (pt) {
        case PieceType::Pawn:   return base + 0;
        case PieceType::Knight: return base + 1;
        case PieceType::Bishop: return base + 2;
        case PieceType::Rook:   return base + 3;
        case PieceType::Queen:  return base + 4;
        case PieceType::King:   return base + 5;
        default:                return -1;
    }
}

std::vector<uint16_t> SelfPlayGen::boardToFeatures(const Board& board) {
    std::vector<uint16_t> features;
    features.reserve(32);
    for (int rank = 0; rank < 8; ++rank) {
        for (int col = 0; col < 8; ++col) {
            const Piece& p = board.squares[rank][col];
            if (p.isNone() || p.isDuck()) continue;
            int pto = pieceTypeOffset(p.type, p.color);
            if (pto < 0) continue;
            features.push_back(static_cast<uint16_t>(pto * 64 + rank * 8 + col));
        }
    }
    // Duck chess: add duck square feature (offset 768)
    if (board.isDuckChess && board.duckSquare.isValid()) {
        uint16_t duckFeature = static_cast<uint16_t>(768 + board.duckSquare.rank * 8 + board.duckSquare.col);
        features.push_back(duckFeature);
    }
    std::sort(features.begin(), features.end());
    return features;
}

// =============================================================================
//  Constructor
// =============================================================================

SelfPlayGen::SelfPlayGen(const NNUE::Network* nnue, const DuckNNUE::Network* duckNnue)
    : nnue_(nnue), duckNnue_(duckNnue) {}

// =============================================================================
//  Chess960 (FRC) starting position generator
//  Generates one of 960 valid starting positions following Chess960 rules:
//    1. Bishops on opposite-colored squares
//    2. King between the two rooks
//    3. Pawns on rank 2/7 as usual
//
//  Uses the standard Chess960 numbering scheme (Scharnagl index 0-959).
// =============================================================================

std::string SelfPlayGen::generateFRC960FEN(int index) {
    // FIX H-4: Removed static RNG fallback — callers must provide a valid index [0, 959].
    // The old `static std::mt19937` was a data race when called from multiple worker threads.
    if (index < 0 || index > 959) {
        // Clamp to a deterministic position rather than invoking thread-unsafe static RNG.
        // Callers (playGame) should always pass a valid index via their thread-local RNG.
        index = 518;  // standard starting position as safe fallback
    }

    char rank[8] = {' ',' ',' ',' ',' ',' ',' ',' '};
    int n = index;

    // 1. Light-square bishop (files 0,2,4,6)
    int b1 = (n % 4) * 2;   n /= 4;
    // 2. Dark-square bishop (files 1,3,5,7)
    int b2 = (n % 4) * 2 + 1; n /= 4;
    if (b1 >= 0 && b1 < 8) rank[b1] = 'B';
    if (b2 >= 0 && b2 < 8) rank[b2] = 'B';

    // 3. Queen in one of the 6 remaining squares
    { int q = n % 6; n /= 6;
      int c = 0;
      for (int i = 0; i < 8; i++) {
          if (rank[i] == ' ') {
              if (c == q) { rank[i] = 'Q'; break; }
              c++;
          }
      }
    }

    // 4. Knights in 2 of the 5 remaining squares (n = 0..9)
    static const int KT[10][2] = {
        {0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}
    };
    { int k1 = KT[n][0], k2 = KT[n][1];
      int c = 0;
      for (int i = 0; i < 8; i++) {
          if (rank[i] == ' ') {
              if (c == k1 || c == k2) rank[i] = 'N';
              c++;
          }
      }
    }

    // 5. Place R, K, R in the 3 remaining empty squares (guarantees K between Rs)
    int empties[3] = {0,0,0}; int ei = 0;
    for (int i = 0; i < 8; i++)
        if (rank[i] == ' ') empties[ei++] = i;
    rank[empties[0]] = 'R';
    rank[empties[1]] = 'K';
    rank[empties[2]] = 'R';

    // Build FEN
    std::string fen;
    for (int i = 0; i < 8; i++) fen += static_cast<char>(rank[i] - 'A' + 'a');
    fen += "/pppppppp/8/8/8/8/PPPPPPPP/";
    for (int i = 0; i < 8; i++) fen += rank[i];

    // FIX H-3: Use Shredder-FEN castling notation for Chess960.
    // empties[0] = queenside rook file, empties[2] = kingside rook file.
    char wkr = static_cast<char>('A' + empties[2]);  // kingside rook (uppercase = white)
    char wqr = static_cast<char>('A' + empties[0]);  // queenside rook
    fen += " w ";
    fen += wkr; fen += wqr;                                          // white castling
    fen += static_cast<char>(wkr + 32); fen += static_cast<char>(wqr + 32);  // black castling
    fen += " - 0 1";
    return fen;
}

// =============================================================================
//  playGame - one self-play game
//
//  Design notes:
//  - getBestMove(Board&) modifies the board during search, so we always pass a
//    copy (searchBoard) and keep the game board intact.
//  - getBestMove resets stop_ internally at the start of each call, so it is
//    safe to call in a tight loop without manual reset.
//  - getLiveEval() returns eval from White's perspective in centipawns.
//    We store centipawns directly to match train_nnue.py's expected scale.
//  - If an opening book is loaded, the game starts from a random FEN instead
//    of the default starting position. Opening plies are still applied on top.
// =============================================================================


// =============================================================================
//  Helper: evaluate all legal moves using incremental NNUE where possible.
//  Uses fusedCopyAndUpdateQ for normal moves (avoids full accumulator refresh),
//  falls back to evaluateQ for promotions and en passant (different feature mapping).
//  Scores are returned from the CURRENT side-to-move's perspective (before move).
//  Uses makeMove/unmakeMove to avoid Board copies.
// =============================================================================
static void evaluateMovesNNUE(
    const NNUE::Network* net, Board& board, const MoveList& legalMoves,
    std::vector<float>& scores)
{
    const Color stm = board.turn;
    scores.resize(static_cast<size_t>(legalMoves.count));

    // One full accumulator refresh for the current position
    NNUE::QAccumulator baseAcc, childAcc;
    net->refreshAccumulatorQ(board, baseAcc);

    for (int i = 0; i < legalMoves.count; i++) {
        const Move& m = legalMoves[i];
        const Piece movedPiece  = board.squares[m.from.rank][m.from.col];
        const Piece capturedPce = board.squares[m.to.rank][m.to.col];

        // Promotion and en passant change the feature mapping in ways that
        // fusedCopyAndUpdateQ doesn't handle (different piece type / capture square).
        // King moves and castling ARE handled internally by fusedCopyAndUpdateQ
        // (it triggers a full refresh when the king moves).
        bool isSpecial = m.promotion != PieceType::None ||
            (movedPiece.type == PieceType::Pawn &&
             m.to.col != m.from.col && capturedPce.isNone()); // en passant

        Board::UndoInfo undo;
        board.makeMove(m, undo);

        int ev;
        if (isSpecial) {
            ev = net->evaluateQ(board);  // full refresh
        } else {
            // 2.9: The `board` parameter is only used for king-move full refresh (post-move board state).
            // For non-king moves, the movedPiece/capturedPce parameters drive the delta computation.
            net->fusedCopyAndUpdateQ(board, baseAcc, childAcc,
                m.from.rank, m.from.col, m.to.rank, m.to.col,
                movedPiece.type, movedPiece.color,
                capturedPce.isNone() ? PieceType::None : capturedPce.type,
                capturedPce.color);
            float phase = NNUE::Network::computePhase(board);
            ev = net->forwardQ(childAcc, board.turn, phase);
        }

        board.unmakeMove(m, undo);

        scores[static_cast<size_t>(i)] = static_cast<float>(-ev);  // ev is from opponent's perspective after makeMove; negate for mover
    }
}

SelfPlayGen::GameOutcome SelfPlayGen::playGame(
    Engine& engine,
    const Config& cfg,
    int gameSeed,
    const std::vector<std::string>& openings,
    std::vector<PositionRecord>& out,
    int searchDepth)
{
    // Clamp mate scores to a learnable range — keeps sigmoid gradient active
    // while still giving a strong "this is winning/losing" signal.
    // Declared here (before use in terminal-check and search-eval blocks).
    static constexpr float MATE_CLAMP = 2500.0f;

    Board board;

    // Set duck chess mode if requested
    if (cfg.isDuckChess) {
        board.isDuckChess = true;
    }

    // 9.6: Hash seed to decorrelate sequential MT19937 seeds (xorshift32 mixing)
    uint32_t hashedSeed = static_cast<uint32_t>(gameSeed) * 2654435761u;  // Knuth multiplicative hash
    hashedSeed ^= hashedSeed << 13; hashedSeed ^= hashedSeed >> 17; hashedSeed ^= hashedSeed << 5;
    std::mt19937 rng(hashedSeed);

    // If we have an opening book, start from a random position
    if (!openings.empty()) {
        std::uniform_int_distribution<size_t> dist(0, openings.size() - 1);
        board.setFromFEN(openings[dist(rng)]);
    } else if (cfg.frcMix > 0.0f) {
        // Phase 4B: Chess960 starting positions for positional diversity
        std::uniform_real_distribution<float> frcDist(0.0f, 1.0f);
        if (frcDist(rng) < cfg.frcMix) {
            std::uniform_int_distribution<int> posDist(0, 959);
            board.setFromFEN(generateFRC960FEN(posDist(rng)));
        } else {
            board.setStartingPosition();
        }
    } else {
        board.setStartingPosition();
    }

    // Initialize Zobrist hash — applyMove now maintains it incrementally
    board.hash = Engine::computeHash(board);

    std::vector<uint64_t> history;
    history.reserve(static_cast<size_t>(cfg.maxPlies));

    std::unordered_map<uint64_t, int> hashCount;

    int resignConsec = 0;
    int resignSign   = 0;  // +1 = white winning, -1 = black winning
    int drawConsec   = 0;
    int drawAdjCounter = 0;  // consecutive plies of near-zero eval for dead-draw adjudication

    // Resign/draw thresholds in centipawns (matching eval scale)
    const float resignThresh = static_cast<float>(cfg.resignCp);
    const float drawThresh   = static_cast<float>(cfg.drawCp);
    const float drawAdjThresh = static_cast<float>(cfg.drawAdjThreshold);

    // Track last search eval for per-game outcome reporting
    float lastEval = 0.0f;

    // FIX: Per-game wall-clock timeout — prevents any single game from stalling the batch
    auto gameStart = std::chrono::steady_clock::now();

    for (int ply = 0; ply < cfg.maxPlies; ++ply) {

        // -- Per-game timeout check -------------------------------------------
        if (cfg.gameTimeLimitSec > 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - gameStart).count();
            if (elapsed >= static_cast<double>(cfg.gameTimeLimitSec))
                return {GameResult::Draw, TermReason::Timeout, ply, lastEval};
        }

        // -- Terminal checks ---------------------------------------------------
        MoveList legalMoves;
        MoveGen::getLegalMoves(board, legalMoves);
        if (legalMoves.empty()) {
            if (MoveGen::isInCheck(board, board.turn)) {
                // Record the terminal checkmate position
                PositionRecord rec;
                rec.features = boardToFeatures(board);
                rec.stm      = (board.turn == Color::White) ? 0 : 1;
                rec.eval     = (board.turn == Color::White) ? -MATE_CLAMP : MATE_CLAMP;
                rec.result   = 0.0f;  // filled in by generate()
                out.push_back(std::move(rec));
                return {(board.turn == Color::White) ? GameResult::BlackWins
                                                        : GameResult::WhiteWins,
                         TermReason::Checkmate, ply,
                         (board.turn == Color::White) ? -MATE_CLAMP : MATE_CLAMP};
            }
            // Record stalemate position
            {
                PositionRecord rec;
                rec.features = boardToFeatures(board);
                rec.stm      = (board.turn == Color::White) ? 0 : 1;
                rec.eval     = 0.0f;
                rec.result   = 0.0f;  // filled in by generate()
                out.push_back(std::move(rec));
            }
            return {GameResult::Draw, TermReason::Stalemate, ply, 0.0f};
        }

        if (board.halfMoveClock >= 100)
            return {GameResult::Draw, TermReason::FiftyMove, ply, lastEval};

        // -- Repetition detection (O(1) via hash map) --------------------------
        uint64_t hash = board.hash;  // maintained incrementally by applyMove
        if (++hashCount[hash] >= 3) return {GameResult::Draw, TermReason::Threefold, ply, lastEval};
        history.push_back(hash);

        // -- Opening: softmax-weighted moves for diversity ----------------------
        // Phase 1A: Use NNUE static eval + softmax instead of uniform random.
        // This creates diverse openings that are still reasonable (not garbage).
        if (ply < cfg.openingPlies) {
            const NNUE::Network* net = engine.getNNUE();
            if (net && cfg.openingTemp > 0.0f && legalMoves.size() > 1) {
                // Evaluate each legal move with incremental NNUE (one refresh + fast updates)
                std::vector<float> scores;
                evaluateMovesNNUE(net, board, legalMoves, scores);
                // Softmax with temperature (scale to centipawns)
                float maxS = *std::max_element(scores.begin(), scores.end());
                float invTemp = 1.0f / (cfg.openingTemp * 100.0f);
                float sum = 0.0f;
                for (auto& s : scores) {
                    s = std::exp((s - maxS) * invTemp);
                    sum += s;
                }
                for (auto& s : scores) s /= sum;
                std::discrete_distribution<int> dist(scores.begin(), scores.end());
                { Board::UndoInfo undo_unused; board.makeMove(legalMoves[dist(rng)], undo_unused); }  // 2.8: undo not needed; move is permanent
            } else {
                // Fallback: uniform random (no NNUE or temp disabled)
                std::uniform_int_distribution<int> dist(
                    0, static_cast<int>(legalMoves.size()) - 1);
                { Board::UndoInfo undo_unused; board.makeMove(legalMoves[dist(rng)], undo_unused); }  // 2.8: undo not needed; move is permanent
            }
            continue;  // don't record opening positions
        }

        // -- Search ------------------------------------------------------------
        engine.setPositionHistory(history);
        Move best = engine.getBestMove(board, searchDepth > 0 ? searchDepth : 64);

        // getLiveEval() = White's perspective eval in centipawns
        // Store centipawns directly - matches train_nnue.py expected scale
        float eval = static_cast<float>(engine.getLiveEval());

        if (eval >  MATE_CLAMP) eval =  MATE_CLAMP;
        if (eval < -MATE_CLAMP) eval = -MATE_CLAMP;
        lastEval = eval;

        // -- Record position (with Phase 3A filtering) --------------------------
        // Skip positions that are too early (book territory) or too extreme (already decided)
        if (ply >= cfg.recordMinPly && std::abs(eval) <= static_cast<float>(cfg.recordMaxEval)) {
            PositionRecord rec;
            rec.features = boardToFeatures(board);
            rec.stm      = (board.turn == Color::White) ? 0 : 1;
            rec.eval     = eval;
            rec.result   = 0.0f;  // filled in by generate() after game ends
            out.push_back(std::move(rec));
        }

        // -- Adjudication: resign ----------------------------------------------
        {
            int sign = (eval >=  resignThresh) ?  1
                     : (eval <= -resignThresh) ? -1 : 0;
            if (sign != 0) {
                if (sign == resignSign) ++resignConsec;
                else { resignConsec = 1; resignSign = sign; }
                if (resignConsec >= cfg.resignCount)
                    return {(resignSign > 0) ? GameResult::WhiteWins
                                              : GameResult::BlackWins,
                             TermReason::Resign, ply, lastEval};
            } else {
                resignConsec = 0;
                resignSign   = 0;
            }
        }

        // -- Adjudication: draw ------------------------------------------------
        if (ply >= cfg.drawMinPly) {
            if (std::abs(eval) <= drawThresh) {
                if (++drawConsec >= cfg.drawCount)
                    return {GameResult::Draw, TermReason::DrawAdj, ply, lastEval};
            } else {
                drawConsec = 0;
            }
        }

        // -- Adjudication: dead-draw (stricter, for truly equal positions) -----
        if (std::abs(eval) < drawAdjThresh) {
            drawAdjCounter++;
        } else {
            drawAdjCounter = 0;
        }
        if (drawAdjCounter >= cfg.drawAdjMoves && ply / 2 >= cfg.drawAdjMinMove) { // 9.5: >= not >, drawAdjMinMove is inclusive
            return {GameResult::Draw, TermReason::DeadDraw, ply, lastEval};
        }

        // -- Post-opening softmax move selection ---------------------------------
        // Phase 1A: During softmaxPlies after opening, pick moves via softmax
        // over NNUE static eval instead of always the engine's best move.
        // The position was already recorded with the search eval (correct for training),
        // but the move played creates diverse game trajectories.
        if (ply < cfg.openingPlies + cfg.softmaxPlies
            && cfg.softmaxTemp > 0.0f && legalMoves.size() > 1) {
            const NNUE::Network* net = engine.getNNUE();
            if (net) {
                std::vector<float> scores;
                evaluateMovesNNUE(net, board, legalMoves, scores);
                float maxS = *std::max_element(scores.begin(), scores.end());
                float invTemp = 1.0f / (cfg.softmaxTemp * 100.0f);
                float sum = 0.0f;
                for (auto& s : scores) {
                    s = std::exp((s - maxS) * invTemp);
                    sum += s;
                }
                for (auto& s : scores) s /= sum;
                std::discrete_distribution<int> dist(scores.begin(), scores.end());
                { Board::UndoInfo undo_unused; board.makeMove(legalMoves[dist(rng)], undo_unused); }  // 2.8: undo not needed; move is permanent
            } else {
                { Board::UndoInfo undo; board.makeMove(best, undo); }
            }
        }
        // -- Phase 2B: Epsilon-greedy root noise ---------------------------------
        // After the softmax phase, with probability rootNoiseEps, replace the
        // engine's best move with a weighted random alternative based on NNUE eval.
        // This creates game diversity in the mid/endgame without corrupting the
        // recorded search eval (which was already stored above).
        else if (cfg.rootNoiseEps > 0.0f && legalMoves.size() > 1) {
            std::uniform_real_distribution<float> coin(0.0f, 1.0f);
            if (coin(rng) < cfg.rootNoiseEps) {
                const NNUE::Network* net = engine.getNNUE();
                if (net) {
                    // Quick static eval of all legal moves using incremental NNUE
                    std::vector<float> scores;
                    evaluateMovesNNUE(net, board, legalMoves, scores);
                    float maxS = *std::max_element(scores.begin(), scores.end());
                    constexpr float NOISE_INV_TEMP = 1.0f / 50.0f;  // temperature=0.5 * 100cp scale
                    float sum = 0.0f;
                    for (auto& s : scores) {
                        s = std::exp((s - maxS) * NOISE_INV_TEMP);
                        sum += s;
                    }
                    for (auto& s : scores) s /= sum;
                    std::discrete_distribution<int> dist(scores.begin(), scores.end());
                    { Board::UndoInfo undo_unused; board.makeMove(legalMoves[dist(rng)], undo_unused); }  // 2.8: undo not needed; move is permanent
                } else {
                    { Board::UndoInfo undo; board.makeMove(best, undo); }
                }
            } else {
                { Board::UndoInfo undo; board.makeMove(best, undo); }  // most of the time, play the engine's best
            }
        } else {
            { Board::UndoInfo undo; board.makeMove(best, undo); }
        }
    }

    return {GameResult::Draw, TermReason::MaxPlies, cfg.maxPlies, lastEval};
}

// =============================================================================
//  generate - orchestrate all games (single or multi-threaded) and write file
// =============================================================================

static const char* termReasonStr(SelfPlayGen::TermReason r) {
    switch (r) {
        case SelfPlayGen::TermReason::Checkmate: return "mate";
        case SelfPlayGen::TermReason::Stalemate: return "stale";
        case SelfPlayGen::TermReason::FiftyMove: return "50move";
        case SelfPlayGen::TermReason::Threefold: return "3fold";
        case SelfPlayGen::TermReason::Resign:    return "resign";
        case SelfPlayGen::TermReason::DrawAdj:   return "drawadj";
        case SelfPlayGen::TermReason::DeadDraw:  return "deaddraw";
        case SelfPlayGen::TermReason::MaxPlies:  return "maxply";
        case SelfPlayGen::TermReason::Timeout:   return "timeout";
        default:                                 return "?";
    }
}

int SelfPlayGen::generate(const Config& cfg) {
    // Install Ctrl+C handler for graceful shutdown
    g_stopRequested = 0;
#ifdef _WIN32
    SetConsoleCtrlHandler(ctrlHandler, TRUE);
#else
    g_prevSigintHandler = std::signal(SIGINT, sigintHandler);
    if (g_prevSigintHandler == SIG_ERR) g_prevSigintHandler = SIG_DFL;
#endif

    // Load opening book if specified
    if (!cfg.openingsFile.empty()) {
        std::ifstream file(cfg.openingsFile);
        if (!file.is_open()) {
            std::cerr << "[SelfPlay] WARNING: Could not open openings file: "
                      << cfg.openingsFile << std::endl;
        } else {
            std::string line;
            while (std::getline(file, line)) {
                // Skip empty lines and comments
                if (line.empty() || line[0] == '#') continue;
                // Trim whitespace
                size_t start = line.find_first_not_of(" \t\r\n");
                size_t end = line.find_last_not_of(" \t\r\n");
                if (start == std::string::npos) continue;
                std::string trimmed = line.substr(start, end - start + 1);

                // Strip optional "N→" prefix (UTF-8 arrow: 0xE2 0x86 0x92)
                // Also handle ASCII "N->" as fallback
                auto arrow = trimmed.find("\xe2\x86\x92");
                if (arrow != std::string::npos) {
                    trimmed = trimmed.substr(arrow + 3); // skip 3-byte UTF-8 arrow
                } else {
                    auto asc = trimmed.find("->");
                    if (asc != std::string::npos)
                        trimmed = trimmed.substr(asc + 2);
                }
                // Trim again after stripping prefix
                start = trimmed.find_first_not_of(" \t");
                if (start != std::string::npos) {
                    end = trimmed.find_last_not_of(" \t\r\n");
                    m_openings.push_back(trimmed.substr(start, end - start + 1));
                }
            }
            std::cout << "[SelfPlay] Loaded " << m_openings.size()
                      << " opening positions from " << cfg.openingsFile << std::endl;
        }
    }

    std::cout << "[SelfPlay] Starting " << cfg.games << " games"
              << "  depth=" << cfg.searchDepth
              << "  movetime=" << cfg.moveTimeMs << "ms"
              << "  workers=" << cfg.workers
              << "  output=" << cfg.outputPath;
    if (!m_openings.empty())
        std::cout << "  openings=" << m_openings.size();
    if (cfg.contemptCp != 0)
        std::cout << "  contempt=" << cfg.contemptCp;
    if (cfg.openingTemp > 0.0f)
        std::cout << "  openingTemp=" << cfg.openingTemp
                  << "  openingPlies=" << cfg.openingPlies;
    if (cfg.softmaxPlies > 0)
        std::cout << "  softmaxPlies=" << cfg.softmaxPlies
                  << "  softmaxTemp=" << cfg.softmaxTemp;
    if (cfg.rootNoiseEps > 0.0f)
        std::cout << "  rootNoise=" << cfg.rootNoiseEps;
    if (cfg.frcMix > 0.0f)
        std::cout << "  frcMix=" << cfg.frcMix;
    if (cfg.recordMinPly > 0)
        std::cout << "  recordMinPly=" << cfg.recordMinPly;
    std::cout << "  recordMaxEval=" << cfg.recordMaxEval;
    std::cout << "  resign=" << cfg.resignCp << "cp/" << cfg.resignCount << "ply"
              << "  draw=" << cfg.drawCp << "cp/" << cfg.drawCount << "ply@" << cfg.drawMinPly;
    if (cfg.drawAdjMoves > 0)
        std::cout << "  drawAdj=" << cfg.drawAdjThreshold << "cp/" << cfg.drawAdjMoves << "ply@" << cfg.drawAdjMinMove;
    if (cfg.mixedDepthRatio > 0.0f) {
        if (cfg.depthShuffle && cfg.searchDepth - cfg.mixedDepthLow >= 2) {
            std::cout << "  depthShuffle=[d" << cfg.mixedDepthLow
                      << "..d" << (cfg.searchDepth - 1) << "] bias=" << cfg.depthShuffleBias
                      << " (" << static_cast<int>(cfg.mixedDepthRatio * 100) << "% shuffled)";
        } else {
            std::cout << "  mixedDepth=" << cfg.mixedDepthLow
                      << "/" << cfg.searchDepth
                      << " (" << static_cast<int>(cfg.mixedDepthRatio * 100) << "%/"
                      << static_cast<int>((1.0f - cfg.mixedDepthRatio) * 100) << "%)";
        }
    }
    std::cout << "\n";
    std::cout.flush();

    // ---- Log full config to file logger for reproducibility -----------------
    {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "[SelfPlay] Config: games=%d depth=%d movetime=%dms workers=%d maxPlies=%d",
            cfg.games, cfg.searchDepth, cfg.moveTimeMs, cfg.workers, cfg.maxPlies);
        if (cfg.logFn) cfg.logFn(buf);

        std::snprintf(buf, sizeof(buf),
            "[SelfPlay]   resign: cp=%d count=%d  |  drawAdj: cp=%d count=%d minPly=%d",
            cfg.resignCp, cfg.resignCount, cfg.drawCp, cfg.drawCount, cfg.drawMinPly);
        if (cfg.logFn) cfg.logFn(buf);

        std::snprintf(buf, sizeof(buf),
            "[SelfPlay]   deadDraw: threshold=%dcp moves=%d minMove=%d",
            cfg.drawAdjThreshold, cfg.drawAdjMoves, cfg.drawAdjMinMove);
        if (cfg.logFn) cfg.logFn(buf);

        std::snprintf(buf, sizeof(buf),
            "[SelfPlay]   contempt=%dcp openingTemp=%.2f openingPlies=%d softmaxPlies=%d softmaxTemp=%.2f",
            cfg.contemptCp, cfg.openingTemp, cfg.openingPlies, cfg.softmaxPlies, cfg.softmaxTemp);
        if (cfg.logFn) cfg.logFn(buf);

        std::snprintf(buf, sizeof(buf),
            "[SelfPlay]   rootNoise=%.3f frcMix=%.2f recordMinPly=%d recordMaxEval=%d",
            cfg.rootNoiseEps, cfg.frcMix, cfg.recordMinPly, cfg.recordMaxEval);
        if (cfg.logFn) cfg.logFn(buf);

        if (cfg.mixedDepthRatio > 0.0f) {
            if (cfg.depthShuffle && cfg.searchDepth - cfg.mixedDepthLow >= 2) {
                std::snprintf(buf, sizeof(buf),
                    "[SelfPlay]   depthShuffle=[d%d..d%d] bias=%.1f ratio=%.0f%% shuffled",
                    cfg.mixedDepthLow, cfg.searchDepth - 1,
                    cfg.depthShuffleBias,
                    cfg.mixedDepthRatio * 100.0f);
            } else {
                std::snprintf(buf, sizeof(buf),
                    "[SelfPlay]   mixedDepth=%d/%d ratio=%.0f%%/%.0f%%",
                    cfg.mixedDepthLow, cfg.searchDepth,
                    cfg.mixedDepthRatio * 100.0f, (1.0f - cfg.mixedDepthRatio) * 100.0f);
            }
            if (cfg.logFn) cfg.logFn(buf);
        }

        if (!cfg.openingsFile.empty()) {
            std::snprintf(buf, sizeof(buf),
                "[SelfPlay]   openings=%zu from %s",
                m_openings.size(), cfg.openingsFile.c_str());
            if (cfg.logFn) cfg.logFn(buf);
        }

        std::snprintf(buf, sizeof(buf),
            "[SelfPlay]   output=%s  gameTimeout=%ds",
            cfg.outputPath.c_str(), cfg.gameTimeLimitSec);
        if (cfg.logFn) cfg.logFn(buf);
    }

    // Pre-allocate per-game storage so threads write to independent indices
    // (no locking required for position writes).
    struct GameData {
        std::vector<PositionRecord> positions;
        GameOutcome outcome;  // full outcome with reason, ply, eval
    };
    std::vector<GameData> allGames(static_cast<size_t>(cfg.games));

    // -- Worker function -------------------------------------------------------
    std::atomic<int> nextGame{0};
    std::atomic<int> doneCount{0};
    std::atomic<int> totalWhiteWins{0};
    std::atomic<int> totalBlackWins{0};
    std::atomic<int> totalDraws{0};
    std::atomic<uint64_t> totalPositions{0};  // 9.9: uint64 to avoid INT_MAX overflow
    std::atomic<uint64_t> totalNodes{0};
    // Depth histogram for shuffle summary (index = depth, max depth 32)
    static constexpr int DEPTH_HIST_SIZE = 32;
    std::atomic<int> depthHist[DEPTH_HIST_SIZE];
    for (int i = 0; i < DEPTH_HIST_SIZE; ++i) depthHist[i].store(0, std::memory_order_relaxed);
    std::mutex       printMtx;
    auto             startTime = std::chrono::steady_clock::now();
    int              actualWorkers = std::max(1, cfg.workers);

    // -- ETA state (protected by printMtx) ------------------------------------
    // Uses EWMA of games/s for a responsive but stable ETA.
    // Previous approach used linear trend extrapolation (dRate/dt projected over
    // remaining time), but that amplified noise from variable game durations into
    // wild ETA swings (83% mean error in simulation vs 9% for EWMA-only).
    double ewmaGps         = 0.0;   // exponentially weighted moving average of games/s
    double lastReportTime  = 0.0;   // elapsed seconds at last progress report
    int    lastReportDone  = 0;     // games completed at last progress report
    int    ewmaSamples     = 0;     // number of EWMA updates so far
    uint64_t lastReportNodes = 0;   // totalNodes at last NPS snapshot
    double ewmaNps         = 0.0;   // EWMA of interval NPS
    double lastNpsSnapTime = 0.0;   // wallElapsed at last NPS snapshot (independent of report timer)

    // NPS sampling: emit NPS_SAMPLE lines at evenly-spaced game intervals
    // so the pipeline can store one NPS value per epoch slot.
    int npsSampleInterval = (cfg.npsSamples > 0 && cfg.games > 0)
                            ? std::max(1, cfg.games / cfg.npsSamples) : 0;
    int npsSampleNext     = npsSampleInterval;  // next game count to emit a sample
    int npsSampleStep     = 0;                  // which sample (1-based, maps to epoch)
    double pauseAdjustSec  = 0.0;   // accumulated dead time from OS thread suspension (pause/resume)
    static constexpr double EWMA_ALPHA = 0.15; // smoothing factor (lower = more stable ETA)

    // -- Countdown display state (protected by printMtx) -------------------------
    // The countdown thread reads these to reprint the status line every second.
    std::atomic<bool> countdownStop{false};
    std::chrono::steady_clock::time_point etaTarget{};   // predicted completion time
    // Thread safety: lastStatusPrefix and lastEwmaGpsForDisplay are protected by printMtx.
    // Do NOT read these without holding the lock.
    char   lastStatusPrefix[256] = {};  // status line before the ETA portion
    double lastEwmaGpsForDisplay = 0.0; // ewmaGps at last progress report
    std::atomic<bool> countdownReady{false}; // FIX M-8: atomic to prevent data race with countdown thread

    // Capture openings by const-ref for threads
    const auto& openings = m_openings;

    auto workerFn = [&]() {
        // Each thread owns its Engine — heap-allocated because Engine has ~282 KB
        // of internal arrays (killers, history, countermoves, pvTable, etc.) which,
        // combined with search recursion (~5.5 KB per ply × 96 max ply), would
        // exceed the default 1 MB Windows thread stack.
        // The NNUE network is shared read-only - safe because Network weights
        // are never modified during inference (evaluate() is functionally const).
        auto eng = std::make_unique<Engine>(Engine::SELFPLAY_TT_SIZE);
        eng->setNNUE(nnue_);
        if (duckNnue_) eng->setDuckNNUE(const_cast<DuckNNUE::Network*>(duckNnue_));
        eng->setContempt(cfg.contemptCp);
        if (cfg.searchDepth > 0 && cfg.moveTimeMs <= 0) {
            // Depth-based: set a very high time limit so it never triggers
            // Scale safety cap by depth: depth 5 → 2.5s, depth 8 → 6.4s, depth 12 → 14.4s
            int safetyMs = std::max(1000, cfg.searchDepth * cfg.searchDepth * 100);
            safetyMs = std::min(safetyMs, 30000);  // never exceed 30s
            eng->setTimeLimit(safetyMs);
        } else {
            eng->setTimeLimit(cfg.moveTimeMs);
        }

        uint64_t prevNodes = 0;  // track last-reported cumulative for delta calculation

        while (true) {
            if (g_stopRequested) break;
            int g = nextGame.fetch_add(1, std::memory_order_relaxed);
            if (g >= cfg.games) break;

            auto& gd = allGames[static_cast<size_t>(g)];
            // Mixed depth strategy: some games at lower depth for throughput
            int gameDepth = cfg.searchDepth;
            if (cfg.mixedDepthRatio > 0.0f && cfg.mixedDepthLow > 0) {
                // Use multiplicative hash to interleave depth-varied games evenly.
                uint32_t h = static_cast<uint32_t>(g);
                h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
                float frac = static_cast<float>(h % 10000) / 10000.0f;
                if (frac < cfg.mixedDepthRatio) {
                    if (cfg.depthShuffle && cfg.searchDepth - cfg.mixedDepthLow >= 2) {
                        // Depth shuffle: sample from geometric distribution over [low, searchDepth)
                        int range = cfg.searchDepth - cfg.mixedDepthLow;
                        // Build CDF with geometric weights: P(d) ∝ bias^(d - low)
                        // Higher depths are more likely when bias > 1.0
                        float bias = cfg.depthShuffleBias;
                        if (bias < 0.1f) bias = 0.1f;  // safety clamp
                        float cumWeights[32];  // max range we'll ever see
                        int maxRange = (range < 32) ? range : 32;
                        float total = 0.0f;
                        for (int i = 0; i < maxRange; ++i) {
                            float w = std::pow(bias, static_cast<float>(i));
                            total += w;
                            cumWeights[i] = total;
                        }
                        // Pick using the hash (second hash to decorrelate from the ratio decision)
                        uint32_t h2 = h ^ 0xDEADBEEF;
                        h2 ^= h2 >> 16; h2 *= 0x45d9f3bu; h2 ^= h2 >> 16;
                        float pick = static_cast<float>(h2 % 10000) / 10000.0f * total;
                        int chosen = 0;
                        for (int i = 0; i < maxRange; ++i) {
                            if (pick <= cumWeights[i]) { chosen = i; break; }
                            chosen = i;
                        }
                        gameDepth = cfg.mixedDepthLow + chosen;
                    } else {
                        // Classic binary: all shuffled games at low depth
                        gameDepth = cfg.mixedDepthLow;
                    }
                }
            }
            if (gameDepth >= 0 && gameDepth < DEPTH_HIST_SIZE)
                depthHist[gameDepth].fetch_add(1, std::memory_order_relaxed);
            gd.outcome = playGame(*eng, cfg, g, openings, gd.positions, gameDepth);

            // Update thread-safe aggregate counters (no reading incomplete games)
            if      (gd.outcome.result == GameResult::WhiteWins) totalWhiteWins.fetch_add(1, std::memory_order_relaxed);
            else if (gd.outcome.result == GameResult::BlackWins) totalBlackWins.fetch_add(1, std::memory_order_relaxed);
            else                                                  totalDraws.fetch_add(1, std::memory_order_relaxed);
            totalPositions.fetch_add(static_cast<uint64_t>(gd.positions.size()), std::memory_order_relaxed);
            // Accumulate search nodes: delta since last game
            uint64_t curNodes = eng->getCumulativeNodes() + eng->getNodes();
            totalNodes.fetch_add(curNodes - prevNodes, std::memory_order_relaxed);
            prevNodes = curNodes;

            int done = doneCount.fetch_add(1, std::memory_order_relaxed) + 1;
            {   // Report progress after every game (no modulo gate)
                std::lock_guard<std::mutex> lk(printMtx);



                int ww = totalWhiteWins.load(std::memory_order_relaxed);
                int bw = totalBlackWins.load(std::memory_order_relaxed);
                int dr = totalDraws.load(std::memory_order_relaxed);
                auto totalPos = totalPositions.load(std::memory_order_relaxed);
                auto now     = std::chrono::steady_clock::now();
                double wallElapsed = std::chrono::duration<double>(now - startTime).count();

                // -- Pause detection --------------------------------------------------
                // When the parent TrainingRunner suspends our threads (pause button),
                // steady_clock keeps ticking but no games complete. On resume, the
                // interval since last report contains a huge dead gap. Detect this by
                // comparing the interval's throughput against the EWMA: if it drops
                // below 10% of the established rate, the excess time was a pause.
                if (ewmaSamples >= 2 && wallElapsed > lastReportTime + 0.001) {
                    double interval = wallElapsed - lastReportTime;
                    int    deltaGames = done - lastReportDone;
                    double intervalGps = deltaGames / interval;
                    if (intervalGps < ewmaGps * 0.10) {
                        // Estimate how long the games actually took at the known rate
                        double expectedTime = (ewmaGps > 0.01) ? deltaGames / ewmaGps : 0.0;
                        double deadTime = interval - expectedTime;
                        if (deadTime > 1.0) {  // only adjust for >1s gaps (noise filter)
                            pauseAdjustSec += deadTime;
                        }
                    }
                }

                double elapsed = wallElapsed - pauseAdjustSec;
                if (elapsed < 0.001) elapsed = 0.001;  // safety clamp
                // rawGps uses true wall time (no pause adjustment) so ETA is honest.
                // Straggler stalls are structural, not user pauses — they must count.
                double rawGps = done / std::max(0.001, wallElapsed);
                int    remain = cfg.games - done;

                // -- EWMA-based ETA (no trend extrapolation) ----------------------
                // Compute instantaneous rate from the interval since last report
                // (using wall time — pause gaps are handled above by skipping EWMA update)
                double reportGps = rawGps;  // fallback for first sample
                if (ewmaSamples > 0 && wallElapsed > lastReportTime + 0.001) {
                    double interval = wallElapsed - lastReportTime;
                    double adjInterval = interval;
                    // Compute interval rate from wall-clock time
                    reportGps = (done - lastReportDone) / std::max(0.001, adjInterval);
                    // During stalls (straggler tail games), let EWMA decay toward the
                    // true rate rather than holding artificially steady.  Floor it at
                    // the cumulative average so it never undershoots reality.
                    if (ewmaSamples >= 2 && reportGps < ewmaGps * 0.10) {
                        reportGps = std::max(rawGps, ewmaGps * 0.50);  // decay by half, floor at avg
                    }
                }

                // Update EWMA of games/s
                if (ewmaSamples == 0) {
                    ewmaGps = reportGps;
                } else {
                    ewmaGps = EWMA_ALPHA * reportGps + (1.0 - EWMA_ALPHA) * ewmaGps;
                }

                // ETA: use the cumulative average rate (done/elapsed) rather than EWMA.
                // EWMA reacts too fast and "forgets" straggler stalls that are structural
                // (every worker batch ends with a slow tail game). The cumulative average
                // naturally accounts for all stalls and gives honest predictions.
                // EWMA is still displayed as "games/s" for real-time throughput feedback.
                uint64_t tn = totalNodes.load(std::memory_order_relaxed);

                // NPS: independent 3-second snapshot window, decoupled from per-game report rate.
                // lastNpsSnapTime tracks its own timer so near-simultaneous game completions
                // (multiple workers finishing at once) don't produce near-zero intSec.
                {
                    double snapInterval = wallElapsed - lastNpsSnapTime;
                    uint64_t nodeDelta = (tn >= lastReportNodes) ? (tn - lastReportNodes) : 0;
                    if (lastNpsSnapTime < 0.001) {
                        // Very first call: seed with cumulative rate
                        ewmaNps = (elapsed > 0.001) ? static_cast<double>(tn) / elapsed : 0.0;
                        lastNpsSnapTime = wallElapsed;
                        lastReportNodes = tn;
                    } else if (snapInterval >= 3.0) {
                        // Only update every 3 seconds — guarantees a meaningful interval
                        if (nodeDelta > 0) {
                            double intervalNps = static_cast<double>(nodeDelta) / snapInterval;
                            // Hard clamp at 50M NPS (far above any realistic duck chess value)
                            intervalNps = std::min(intervalNps, 50000000.0);
                            ewmaNps = (ewmaNps < 1.0) ? intervalNps
                                                      : 0.25 * intervalNps + 0.75 * ewmaNps;
                        }
                        lastNpsSnapTime = wallElapsed;
                        lastReportNodes = tn;
                    }
                    // Between snapshots: keep previous ewmaNps unchanged
                }

                lastReportTime = wallElapsed;  // always track wall time for interval detection
                lastReportDone = done;
                ewmaSamples++;

                // Tail-aware ETA: scale by active workers when parallelism drops
                int activeWorkers = std::min(actualWorkers, remain);
                if (activeWorkers < 1) activeWorkers = 1;
                double tailFactor = static_cast<double>(activeWorkers) / actualWorkers;
                double effectiveGps = rawGps * tailFactor;
                int etaSec = (effectiveGps > 0.01) ? static_cast<int>(remain / effectiveGps) : 0;

                // Build the status prefix (everything before the ETA)
                int winPct  = (done > 0) ? (100 * ww / done) : 0;
                int drawPct = (done > 0) ? (100 * dr / done) : 0;
                int lossPct = (done > 0) ? (100 * bw / done) : 0;
                int donePct = (cfg.games > 0) ? (100 * done / cfg.games) : 0;

                // Use EWMA interval NPS (falls back to cumulative for first sample)
                uint64_t nps = (ewmaNps > 0.0) ? static_cast<uint64_t>(ewmaNps)
                             : (elapsed > 0.001 ? static_cast<uint64_t>(tn / elapsed) : 0);

                // Format NPS with K/M suffix for readability
                char npsBuf[32];
                if (nps >= 1000000)
                    std::snprintf(npsBuf, sizeof(npsBuf), "%.1fM nps", nps / 1000000.0);
                else if (nps >= 1000)
                    std::snprintf(npsBuf, sizeof(npsBuf), "%.1fK nps", nps / 1000.0);
                else
                    std::snprintf(npsBuf, sizeof(npsBuf), "%llu nps", (unsigned long long)nps);

                // Build depth tag: "d4~d8" for shuffle, "d4/d6" for mixed, "d6" for single
                char depthBuf[32];
                if (cfg.mixedDepthRatio > 0.0f && cfg.mixedDepthLow > 0) {
                    if (cfg.depthShuffle && cfg.searchDepth - cfg.mixedDepthLow >= 2)
                        std::snprintf(depthBuf, sizeof(depthBuf), "d%d~d%d", cfg.mixedDepthLow, cfg.searchDepth - 1);
                    else
                        std::snprintf(depthBuf, sizeof(depthBuf), "d%d/d%d", cfg.mixedDepthLow, cfg.searchDepth);
                } else {
                    std::snprintf(depthBuf, sizeof(depthBuf), "d%d", cfg.searchDepth);
                }

                std::snprintf(lastStatusPrefix, sizeof(lastStatusPrefix),
                    "[SelfPlay] %d/%d (%d%%)  pos=%d (%.0f pos/s)  W/D/B=%d/%d/%d  (%d%%/%d%%/%d%%)  %s  %.1f games/s (avg %.1f)  %s",
                    done, cfg.games, donePct, static_cast<int>(totalPos),
                    (wallElapsed > 0.1 ? totalPos / wallElapsed : 0.0),
                    ww, dr, bw, winPct, drawPct, lossPct, depthBuf, ewmaGps, rawGps, npsBuf);

                // Set the countdown target time
                etaTarget = std::chrono::steady_clock::now() + std::chrono::seconds(etaSec);
                lastEwmaGpsForDisplay = ewmaGps;
                countdownReady = true;

                // Print the line immediately too
                int    etaH = etaSec / 3600;
                int    etaM = (etaSec % 3600) / 60;
                int    etaS = etaSec % 60;
                char lineBuf[256];
                int len = std::snprintf(lineBuf, sizeof(lineBuf),
                    "\r%s  ETA %02d:%02d:%02d", lastStatusPrefix, etaH, etaM, etaS);
                // Pad to fixed width to overwrite previous longer lines
                if (len > 0 && len < (int)sizeof(lineBuf) - 1) {
                    int pad = 120 - len;
                    for (int p = 0; p < pad && len < (int)sizeof(lineBuf) - 1; ++p)
                        lineBuf[len++] = ' ';
                    lineBuf[len] = '\0';
                }
                std::fwrite(lineBuf, 1, static_cast<size_t>(len), stdout);
                std::fwrite("\n", 1, 1, stdout);  // newline so pipe readers (TrainingRunner) receive each update
                std::fflush(stdout);

                // Emit NPS_SAMPLE line if we've crossed a sample boundary
                if (npsSampleInterval > 0 && done >= npsSampleNext && ewmaNps > 0.0) {
                    npsSampleStep++;
                    char sampleBuf[128];
                    std::snprintf(sampleBuf, sizeof(sampleBuf),
                        "NPS_SAMPLE step=%d nps=%.1f", npsSampleStep, ewmaNps);
                    std::fwrite(sampleBuf, 1, std::strlen(sampleBuf), stdout);
                    std::fwrite("\n", 1, 1, stdout);
                    std::fflush(stdout);
                    npsSampleNext += npsSampleInterval;
                }
            }
        }
    };
    // -- Countdown thread (ticks every second to update ETA display) ----------
    auto countdownFn = [&]() {
        auto nextTick = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (!countdownStop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_until(nextTick);
            nextTick += std::chrono::seconds(1);
            if (countdownStop.load(std::memory_order_relaxed)) break;

            std::lock_guard<std::mutex> lk(printMtx);
            if (!countdownReady) continue;

            auto now = std::chrono::steady_clock::now();
            int remSec = static_cast<int>(
                std::chrono::duration<double>(etaTarget - now).count());

            // FIX: When ETA goes negative (overtime), display as count-up with + prefix
            bool overtime = (remSec < 0);
            int dispSec = overtime ? -remSec : remSec;

            int h = dispSec / 3600;
            int m = (dispSec % 3600) / 60;
            int s = dispSec % 60;

            char lineBuf[256];
            int len = std::snprintf(lineBuf, sizeof(lineBuf),
                "\r%s  ETA %s%02d:%02d:%02d", lastStatusPrefix,
                overtime ? "+" : "", h, m, s);
            if (len > 0 && len < (int)sizeof(lineBuf) - 1) {
                int pad = 120 - len;
                for (int p = 0; p < pad && len < (int)sizeof(lineBuf) - 1; ++p)
                    lineBuf[len++] = ' ';
                lineBuf[len] = '\0';
            }
            std::fwrite(lineBuf, 1, static_cast<size_t>(len), stdout);
            std::fflush(stdout);
        }
    };

    std::thread countdownThread(countdownFn);

    // -- Launch worker threads -------------------------------------------------
    // Use 8 MB stack per thread — the enlarged Board struct (with bitboards +
    // UndoInfo snapshots) uses ~270 bytes per makeMove call. At depth 9 with
    // ~30 moves per node and getLegalMoves calling makeMove for each, the stack
    // usage per ply is ~8 KB. With 19 plies of recursion × 12 threads, the
    // default 1 MB stack overflows. 8 MB gives comfortable headroom.

    // Print an immediate 0/N status line so TrainingRunner shows progress right away
    // (without this, the UI shows "0/N" with no update until the first game finishes,
    //  which at depth 9 can take several minutes)
    {
        char depthBuf[32];
        if (cfg.mixedDepthRatio > 0.0f && cfg.mixedDepthLow > 0) {
            if (cfg.depthShuffle && cfg.searchDepth - cfg.mixedDepthLow >= 2)
                std::snprintf(depthBuf, sizeof(depthBuf), "d%d~d%d", cfg.mixedDepthLow, cfg.searchDepth - 1);
            else
                std::snprintf(depthBuf, sizeof(depthBuf), "d%d/d%d", cfg.mixedDepthLow, cfg.searchDepth);
        } else {
            std::snprintf(depthBuf, sizeof(depthBuf), "d%d", cfg.searchDepth);
        }
        char startBuf[256];
        std::snprintf(startBuf, sizeof(startBuf),
            "\r[SelfPlay] 0/%d (0%%)  pos=0 (0 pos/s)  W/D/B=0/0/0  %s  warming up...",
            cfg.games, depthBuf);
        std::fwrite(startBuf, 1, std::strlen(startBuf), stdout);
        std::fwrite("\n", 1, 1, stdout);
        std::fflush(stdout);
    }

    const int numWorkers = actualWorkers;
    if (numWorkers == 1) {
        workerFn();
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(numWorkers));
        for (int i = 0; i < numWorkers; ++i) {
#ifdef _WIN32
            // Launch with 8 MB stack to handle deep search + large Board struct
            auto* fn = new std::function<void()>(workerFn);
            HANDLE hThread = CreateThread(
                nullptr,
                8 * 1024 * 1024,  // 8 MB stack
                [](LPVOID arg) -> DWORD {
                    auto* f = static_cast<std::function<void()>*>(arg);
                    (*f)();
                    delete f;
                    return 0;
                },
                fn, 0, nullptr);
            if (hThread) {
                threads.emplace_back([hThread]() {
                    WaitForSingleObject(hThread, INFINITE);
                    CloseHandle(hThread);
                });
            } else {
                delete fn;
                threads.emplace_back(workerFn);  // fallback
            }
#else
            threads.emplace_back(workerFn);
#endif
        }
        for (auto& t : threads) t.join();
    }

    // Stop the countdown thread
    countdownStop.store(true, std::memory_order_relaxed);
    countdownThread.join();

    std::cout << "\n";  // newline after \r progress line
    std::cout.flush();

    // -- Trim to only completed games ------------------------------------------
    int completedGames = doneCount.load(std::memory_order_relaxed);
    bool interrupted = (g_stopRequested != 0);

    // Print depth distribution histogram
    {
        std::string hist = "[SelfPlay] Depth distribution:";
        bool any = false;
        for (int d = 0; d < DEPTH_HIST_SIZE; ++d) {
            int cnt = depthHist[d].load(std::memory_order_relaxed);
            if (cnt > 0) {
                hist += (any ? " " : " ") + std::string("d") + std::to_string(d) + "=" + std::to_string(cnt);
                any = true;
            }
        }
        if (any) {
            std::cout << hist << "\n";
            std::cout.flush();
        }
    }

    if (interrupted) {
        std::cout << "\n[SelfPlay] Interrupted after " << completedGames
                  << "/" << cfg.games << " games. Saving collected data...\n";
        std::cout.flush();
        // Safe: all workers claim sequential indices and complete claimed games before
        // checking the stop flag. Thread join at line above ensures all workers are done.
        allGames.resize(static_cast<size_t>(completedGames));
    }

    // -- Fill result values into all position records --------------------------
    totalPositions.store(0, std::memory_order_relaxed);
    for (auto& gd : allGames) {
        const float rv = (gd.outcome.result == GameResult::Draw)      ? 0.5f
                       : (gd.outcome.result == GameResult::WhiteWins) ? 1.0f : 0.0f;
        for (auto& pos : gd.positions)
            pos.result = rv;
        totalPositions.fetch_add(static_cast<uint64_t>(gd.positions.size()), std::memory_order_relaxed);
    }

    // -- Write binary file -----------------------------------------------------
    FILE* fp = nullptr;
#ifdef _MSC_VER
    fopen_s(&fp, cfg.outputPath.c_str(), "wb");
#else
    fp = std::fopen(cfg.outputPath.c_str(), "wb");
#endif
    if (!fp) {
        std::cerr << "[SelfPlay] ERROR: Cannot open output file: "
                  << cfg.outputPath << "\n";
        return -1;
    }

    // AUDIT FIX PT-3: Write versioned header (magic + version + count)
    // Magic: 'N','N','U','E' | Version: 1 | Count: uint32
    static constexpr uint8_t MAGIC[4] = {'N', 'N', 'U', 'E'};
    static constexpr uint8_t FORMAT_VERSION = 1;
    // 9.2: Check fwrite return values to detect disk-full / I/O errors
    auto checked_fwrite = [&](const void* ptr, size_t size, size_t count) -> bool {
        return std::fwrite(ptr, size, count, fp) == count;
    };
    if (!checked_fwrite(MAGIC, 1, 4) ||
        !checked_fwrite(&FORMAT_VERSION, sizeof(uint8_t), 1)) {
        std::cerr << "[SelfPlay] ERROR: Failed to write file header\n";
        std::fclose(fp);
        return -1;
    }
    const uint64_t totalPos64 = totalPositions.load(std::memory_order_relaxed);
    const auto posCount = static_cast<uint32_t>(std::min(totalPos64, (uint64_t)UINT32_MAX));  // 9.9: safe truncation
    if (!checked_fwrite(&posCount, sizeof(uint32_t), 1)) { std::fclose(fp); return -1; }

    // FIX 9.2: Use checked_fwrite for ALL record writes (was raw fwrite — silent data loss on disk full)
    for (const auto& gd : allGames) {
        for (const auto& pos : gd.positions) {
            const auto nf = static_cast<uint16_t>(pos.features.size());
            if (!checked_fwrite(&nf, sizeof(uint16_t), 1)) { std::fclose(fp); return -1; }
            if (nf > 0)
                if (!checked_fwrite(pos.features.data(), sizeof(uint16_t), nf)) { std::fclose(fp); return -1; }
            if (!checked_fwrite(&pos.stm, sizeof(uint8_t), 1)) { std::fclose(fp); return -1; }
            if (!checked_fwrite(&pos.result, sizeof(float), 1)) { std::fclose(fp); return -1; }
            if (!checked_fwrite(&pos.eval, sizeof(float), 1)) { std::fclose(fp); return -1; }
        }
    }

    std::fclose(fp);

    // Restore default signal handling
#ifdef _WIN32
    SetConsoleCtrlHandler(ctrlHandler, FALSE);
#else
    std::signal(SIGINT, g_prevSigintHandler);
#endif

    if (interrupted) {
        std::cout << "[SelfPlay] Saved " << totalPositions.load(std::memory_order_relaxed)
                  << " positions from " << completedGames << " games to "
                  << cfg.outputPath << " (interrupted)\n";
    } else {
        std::cout << "[SelfPlay] Done. " << totalPositions.load(std::memory_order_relaxed)
                  << " positions written to " << cfg.outputPath << "\n";
    }
    return static_cast<int>(totalPositions.load(std::memory_order_relaxed));
}
