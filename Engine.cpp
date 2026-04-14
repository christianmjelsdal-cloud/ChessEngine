#include "Engine.h"
#include "Zobrist.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <cmath>
#include <sstream>

// =============================================================
//  Dynamic Draw Score
// =============================================================
int Engine::drawScore() const {
    int bias = -rootEval_ / 4;
    return std::max(-50, std::min(50, bias));
}

// =============================================================
//  ZOBRIST HASHING  (static members + Zobrist namespace)
// =============================================================
bool     Engine::zobristReady_ = false;
uint64_t Engine::zPiece_[2][7][64];
uint64_t Engine::zCastle_[16];
uint64_t Engine::zEP_[8];
uint64_t Engine::zSide_;
uint64_t Engine::zDuck_[64];

// Zobrist namespace definitions (used by Board::applyMove for incremental hash)
namespace Zobrist {
    bool     ready       = false;
    uint64_t piece[2][7][64];
    uint64_t castle[16];
    uint64_t ep[8];
    uint64_t side;
    uint64_t duck[64];
}

void Engine::initZobrist() {
    if (zobristReady_) return;
    std::mt19937_64 rng(0xBEEF1234ULL);
    for (int c = 0; c < 2; c++)
        for (int p = 0; p < 7; p++)
            for (int s = 0; s < 64; s++)
                zPiece_[c][p][s] = Zobrist::piece[c][p][s] = rng();
    for (int i = 0; i < 16; i++) zCastle_[i] = Zobrist::castle[i] = rng();
    for (int i = 0; i < 8;  i++) zEP_[i]     = Zobrist::ep[i]     = rng();
    zSide_ = Zobrist::side = rng();
    for (int i = 0; i < 64; i++) zDuck_[i] = Zobrist::duck[i] = rng();
    zobristReady_ = Zobrist::ready = true;
}

uint64_t Engine::computeHash(const Board& board) {
    uint64_t h = 0;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            Piece p = board.squares[r][c];
            if (!p.isNone() && !p.isDuck())
                h ^= Zobrist::piece[(int)p.color][(int)p.type][r * 8 + c];
        }
    int ci = (board.castlingRights[0][0] ? 1 : 0)
           | (board.castlingRights[0][1] ? 2 : 0)
           | (board.castlingRights[1][0] ? 4 : 0)
           | (board.castlingRights[1][1] ? 8 : 0);
    h ^= Zobrist::castle[ci];
    if (board.enPassantTarget.isValid())
        h ^= Zobrist::ep[board.enPassantTarget.col];
    if (board.turn == Color::Black)
        h ^= Zobrist::side;
    if (board.isDuckChess && board.duckSquare.isValid())
        h ^= Zobrist::duck[board.duckSquare.rank * 8 + board.duckSquare.col];
    return h;
}

// =============================================================
//  PeSTO PIECE-SQUARE TABLES
// =============================================================
static const int mg_value[] = { 0, 82, 337, 365, 477, 1025, 0 };
static const int eg_value[] = { 0, 94, 281, 297, 512, 936, 0 };
static const int phase_weight[] = { 0, 0, 1, 1, 2, 4, 0 };
static const int TOTAL_PHASE = 24;

static const int SEE_VAL[] = { 0, 100, 320, 330, 500, 900, 20000 };

static const int FUTILITY_MARGIN[] = { 0, 200, 300, 500 };
static const int LMP_THRESHOLD[] = { 0, 5, 8, 13 };

static const int mg_pawn[8][8] = {
    {  0,   0,   0,   0,   0,   0,   0,   0},
    { 98, 134,  61,  95,  68, 126,  34, -11},
    { -6,   7,  26,  31,  65,  56,  25, -20},
    {-14,  13,   6,  21,  23,  12,  17, -23},
    {-27,  -2,  -5,  12,  17,   6,  10, -25},
    {-26,  -4,  -4, -10,   3,   3,  33, -12},
    {-35,  -1, -20, -23, -15,  24,  38, -22},
    {  0,   0,   0,   0,   0,   0,   0,   0}
};
static const int mg_knight[8][8] = {
    {-167, -89, -34, -49,  61, -97, -15,-107},
    { -73, -41,  72,  36,  23,  62,   7, -17},
    { -47,  60,  37,  65,  84, 129,  73,  44},
    {  -9,  17,  19,  53,  37,  69,  18,  22},
    { -13,   4,  16,  13,  28,  19,  21,  -8},
    { -23,  -9,  12,  10,  19,  17,  25, -16},
    { -29, -53, -12,  -3,  -1,  18, -14, -19},
    {-105, -21, -58, -33, -17, -28, -19, -23}
};
static const int mg_bishop[8][8] = {
    {-29,   4, -82, -37, -25, -42,   7,  -8},
    {-26,  16, -18, -13,  30,  59,  18, -47},
    {-16,  37,  43,  40,  35,  50,  37,  -2},
    { -4,   5,  19,  50,  37,  37,   7,  -2},
    { -6,  13,  13,  26,  34,  12,  10,   4},
    {  0,  15,  15,  15,  14,  27,  18,  10},
    {  4,  15,  16,   0,   7,  21,  33,   1},
    {-33,  -3, -14, -21, -13, -12, -39, -21}
};
static const int mg_rook[8][8] = {
    { 32,  42,  32,  51,  63,   9,  31,  43},
    { 27,  32,  58,  62,  80,  67,  26,  44},
    { -5,  19,  26,  36,  17,  45,  61,  16},
    {-24, -11,   7,  26,  24,  35,  -8, -20},
    {-36, -26, -12,  -1,   9,  -7,   6, -23},
    {-45, -25, -16, -17,   3,   0,  -5, -33},
    {-44, -16, -20,  -9,  -1,  11,  -6, -71},
    {-19, -13,   1,  17,  16,   7, -37, -26}
};
static const int mg_queen[8][8] = {
    {-28,   0,  29,  12,  59,  44,  43,  45},
    {-24, -39,  -5,   1, -16,  57,  28,  54},
    {-13, -17,   7,   8,  29,  56,  47,  57},
    {-27, -27, -16, -16,  -1,  17,  -2,   1},
    { -9, -26,  -9, -10,  -2,  -4,   3,  -3},
    {-14,   2, -11,  -2,  -5,   2,  14,   5},
    {-35,  -8,  11,   2,   8,  15,  -3,   1},
    { -1, -18,  -9,  10, -15, -25, -31, -50}
};
static const int mg_king[8][8] = {
    {-65,  23,  16, -15, -56, -34,   2,  13},
    { 29,  -1, -20,  -7,  -8,  -4, -38, -29},
    { -9,  24,   2, -16, -20,   6,  22, -22},
    {-17, -20, -12, -27, -30, -25, -14, -36},
    {-49,  -1, -27, -39, -46, -44, -33, -51},
    {-14, -14, -22, -46, -44, -30, -15, -27},
    {  1,   7,  -8, -64, -43, -16,   9,   8},
    {-15,  36,  12, -54,   8, -28,  24,  14}
};
static const int eg_pawn[8][8] = {
    {  0,   0,   0,   0,   0,   0,   0,   0},
    {178, 173, 158, 134, 147, 132, 165, 187},
    { 94, 100,  85,  67,  56,  53,  82,  84},
    { 32,  24,  13,   5,  -2,   4,  17,  17},
    { 13,   9,  -3,  -7,  -7,  -8,   3,  -1},
    {  4,   7,  -6,   1,   0,  -5,  -1,  -8},
    { 13,   8,   8, -10,  13,   0,   2,  -7},
    {  0,   0,   0,   0,   0,   0,   0,   0}
};
static const int eg_knight[8][8] = {
    {-58, -38, -13, -28, -31, -27, -63, -99},
    {-25,  -8, -25,  -2,  -9, -25, -24, -52},
    {-24, -20,  10,   9,  -1,  -9, -19, -41},
    {-17,   3,  22,  22,  22,  11,   8, -18},
    {-18,  -6,  16,  25,  16,  17,   4, -18},
    {-23,  -3,  -1,  15,  10,  -3, -20, -22},
    {-42, -20, -10,  -5,  -2, -20, -23, -44},
    {-29, -51, -23, -15, -22, -18, -50, -64}
};
static const int eg_bishop[8][8] = {
    {-14, -21, -11,  -8,  -7,  -9, -17, -24},
    { -8,  -4,   7, -12,  -3, -13,  -4, -14},
    {  2,  -8,   0,  -1,  -2,   6,   0,   4},
    { -3,   9,  12,   9,  14,  10,   3,   2},
    { -6,   3,  13,  19,   7,  10,  -3,  -9},
    {-12,  -3,   8,  10,  13,   3,  -7, -15},
    {-14, -18,  -7,  -1,   4,  -9, -15, -27},
    {-23,  -9, -23,  -5,  -9, -16,  -5, -17}
};
static const int eg_rook[8][8] = {
    { 13,  10,  18,  15,  12,  12,   8,   5},
    { 11,  13,  13,  11,  -3,   3,   8,   3},
    {  7,   7,   7,   5,   4,  -3,  -5,  -3},
    {  4,   3,  13,   1,   2,   1,  -1,   2},
    {  3,   5,   8,   4,  -5,  -6,  -8, -11},
    { -4,   0,  -5,  -1,  -7, -12,  -8, -16},
    { -6,  -6,   0,   2,  -9,  -9, -11,  -3},
    { -9,   2,   3,  -1,  -5, -13,   4, -20}
};
static const int eg_queen[8][8] = {
    { -9,  22,  22,  27,  27,  19,  10,  20},
    {-17,  20,  32,  41,  58,  25,  30,   0},
    {-20,   6,   9,  49,  47,  35,  19,   9},
    {  3,  22,  24,  45,  57,  40,  57,  36},
    {-18,  28,  19,  47,  31,  34,  39,  23},
    {-16, -27,  15,   6,   9,  17,  10,   5},
    {-22, -23, -30, -16, -16, -23, -36, -32},
    {-33, -28, -22, -43,  -5, -32, -20, -41}
};
static const int eg_king[8][8] = {
    {-74, -35, -18, -18, -11,  15,   4, -17},
    {-12,  17,  14,  17,  17,  38,  23,  11},
    { 10,  17,  23,  15,  20,  45,  44,  13},
    { -8,  22,  24,  27,  26,  33,  26,   3},
    {-18,  -4,  21,  24,  27,  23,   9, -11},
    {-19,  -3,  11,  21,  23,  16,   7,  -9},
    {-27, -11,   4,  13,  14,   4,  -5, -17},
    {-53, -34, -21, -11, -28, -14, -24, -43}
};

static const int (*mg_tables[7])[8] = {
    nullptr, mg_pawn, mg_knight, mg_bishop, mg_rook, mg_queen, mg_king
};
static const int (*eg_tables[7])[8] = {
    nullptr, eg_pawn, eg_knight, eg_bishop, eg_rook, eg_queen, eg_king
};

// =============================================================
//  LIVE PV ACCESS (thread-safe)
// =============================================================
std::vector<Move> Engine::getLivePV() {
    std::lock_guard<std::mutex> lock(livePVMutex_);
    return livePV_;
}

// =============================================================
//  HELPER: elapsed milliseconds
// =============================================================
int64_t Engine::elapsedMs() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now - searchStart_).count();
}

// =============================================================
//  HELPER: PV to UCI string
// =============================================================
std::string Engine::pvToUCI(const std::vector<Move>& pv) const {
    // Forward-declared helper — UCI layer provides moveToUCI, but we need
    // a simple inline version for info output from inside Engine.
    std::string result;
    for (const auto& m : pv) {
        if (!result.empty()) result += " ";
        std::string s;
        s += static_cast<char>('a' + m.from.col);
        s += static_cast<char>('1' + m.from.rank);
        s += static_cast<char>('a' + m.to.col);
        s += static_cast<char>('1' + m.to.rank);
        if (m.promotion != PieceType::None) {
            switch (m.promotion) {
                case PieceType::Queen:  s += 'q'; break;
                case PieceType::Rook:   s += 'r'; break;
                case PieceType::Bishop: s += 'b'; break;
                case PieceType::Knight: s += 'n'; break;
                default: break;
            }
        }
        result += s;
    }
    return result;
}

// =============================================================
//  CONSTRUCTORS
// =============================================================
Engine::Engine() : tt_(TT_SIZE), nodes_(0) {
    initZobrist();
    std::memset(killers_, 0, sizeof(killers_));
    std::memset(history_, 0, sizeof(history_));
    std::memset(countermoves_, 0, sizeof(countermoves_));
}

Engine::Engine(size_t ttSize) : tt_(ttSize), nodes_(0) {
    initZobrist();
    std::memset(killers_, 0, sizeof(killers_));
    std::memset(history_, 0, sizeof(history_));
    std::memset(countermoves_, 0, sizeof(countermoves_));
}

void Engine::clearSearchState() {
    std::memset(killers_, 0, sizeof(killers_));
    std::memset(history_, 0, sizeof(history_));
    std::memset(countermoves_, 0, sizeof(countermoves_));
    for (auto& e : tt_) e = TTEntry{};
    ttGen_ = 0;
    nodes_ = 0;
}

// =============================================================
//  TIME MANAGEMENT — uses HARD limit for abort
// =============================================================
bool Engine::shouldStop() {
    if (stop_.load(std::memory_order_relaxed)) return true;

    // Check time every 4096 nodes to avoid expensive clock calls
    if ((nodes_ & 4095) == 0) {
        if (elapsedMs() >= hardLimitMs_) {
            stop_.store(true, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

// =============================================================
//  STATIC EXCHANGE EVALUATION (SEE)
// =============================================================
PieceType Engine::findLVA(const Board& board, Square target, Color side,
                          const bool removed[8][8], Square& outSq) {
    // 1. Pawns
    int pawnSrcRank = (side == Color::White) ? target.rank - 1 : target.rank + 1;
    if (pawnSrcRank >= 0 && pawnSrcRank < 8) {
        for (int dc : {-1, 1}) {
            int pc = target.col + dc;
            if (pc >= 0 && pc < 8 && !removed[pawnSrcRank][pc]) {
                Piece p = board.squares[pawnSrcRank][pc];
                if (p.type == PieceType::Pawn && p.color == side) {
                    outSq = {pawnSrcRank, pc};
                    return PieceType::Pawn;
                }
            }
        }
    }

    // 2. Knights
    static const int kOff[8][2] = {{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
    for (auto& o : kOff) {
        int nr = target.rank + o[0], nc = target.col + o[1];
        if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8 && !removed[nr][nc]) {
            Piece p = board.squares[nr][nc];
            if (p.type == PieceType::Knight && p.color == side) {
                outSq = {nr, nc};
                return PieceType::Knight;
            }
        }
    }

    // 3. Diagonal sliders — bishops first, then queens
    static const int diagDirs[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    Square queenDiagSq = {-1,-1};
    bool foundQueenDiag = false;

    for (auto& d : diagDirs) {
        int nr = target.rank + d[0], nc = target.col + d[1];
        while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            if (!removed[nr][nc]) {
                Piece p = board.squares[nr][nc];
                if (!p.isNone() && !p.isDuck()) {
                    if (p.color == side) {
                        if (p.type == PieceType::Bishop) {
                            outSq = {nr, nc};
                            return PieceType::Bishop;
                        }
                        if (p.type == PieceType::Queen && !foundQueenDiag) {
                            queenDiagSq = {nr, nc};
                            foundQueenDiag = true;
                        }
                    }
                    break;
                }
            }
            nr += d[0]; nc += d[1];
        }
    }

    // 4. Straight sliders — rooks first, then queens
    static const int straightDirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    Square queenStraightSq = {-1,-1};
    bool foundQueenStraight = false;

    for (auto& d : straightDirs) {
        int nr = target.rank + d[0], nc = target.col + d[1];
        while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            if (!removed[nr][nc]) {
                Piece p = board.squares[nr][nc];
                if (!p.isNone() && !p.isDuck()) {
                    if (p.color == side) {
                        if (p.type == PieceType::Rook) {
                            outSq = {nr, nc};
                            return PieceType::Rook;
                        }
                        if (p.type == PieceType::Queen && !foundQueenStraight) {
                            queenStraightSq = {nr, nc};
                            foundQueenStraight = true;
                        }
                    }
                    break;
                }
            }
            nr += d[0]; nc += d[1];
        }
    }

    // 5. Queens (found during slider scan)
    if (foundQueenDiag) {
        outSq = queenDiagSq;
        return PieceType::Queen;
    }
    if (foundQueenStraight) {
        outSq = queenStraightSq;
        return PieceType::Queen;
    }

    // 6. King
    for (int dr = -1; dr <= 1; dr++)
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;
            int nr = target.rank + dr, nc = target.col + dc;
            if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8 && !removed[nr][nc]) {
                Piece p = board.squares[nr][nc];
                if (p.type == PieceType::King && p.color == side) {
                    outSq = {nr, nc};
                    return PieceType::King;
                }
            }
        }

    return PieceType::None;
}

int Engine::see(const Board& board, const Move& m) {
    Piece attacker = board.getPiece(m.from);
    Piece victim   = board.getPiece(m.to);

    if (victim.isNone()) {
        if (attacker.type == PieceType::Pawn && m.from.col != m.to.col)
            return SEE_VAL[(int)PieceType::Pawn];
        return 0;
    }

    bool removed[8][8] = {};
    int gain[32];
    int d = 0;

    gain[0] = SEE_VAL[(int)victim.type];
    PieceType lastType = attacker.type;
    removed[m.from.rank][m.from.col] = true;

    Color side = (attacker.color == Color::White) ? Color::Black : Color::White;

    while (true) {
        d++;
        gain[d] = SEE_VAL[(int)lastType] - gain[d - 1];

        if (std::max(-gain[d - 1], gain[d]) < 0) break;

        Square from;
        PieceType found = findLVA(board, m.to, side, removed, from);
        if (found == PieceType::None) break;

        removed[from.rank][from.col] = true;
        lastType = found;
        side = (side == Color::White) ? Color::Black : Color::White;
    }

    while (--d > 0)
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);

    return gain[0];
}

// =============================================================
//  EVALUATION
// =============================================================
static int countSliderMobility(const Board& board, Square sq, Color color,
                                const int dirs[][2], int numDirs) {
    int count = 0;
    for (int i = 0; i < numDirs; i++) {
        int nr = sq.rank + dirs[i][0], nc = sq.col + dirs[i][1];
        while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            Piece p = board.squares[nr][nc];
            if (p.isNone()) {
                count++;
            } else if (p.isDuck()) {
                break;
            } else {
                if (p.color != color) count++;
                break;
            }
            nr += dirs[i][0]; nc += dirs[i][1];
        }
    }
    return count;
}

static int countKnightMobility(const Board& board, Square sq, Color color) {
    static const int kOff[8][2] = {{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
    int count = 0;
    for (auto& o : kOff) {
        int nr = sq.rank + o[0], nc = sq.col + o[1];
        if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            Piece p = board.squares[nr][nc];
            if (p.isNone() || (p.color != color && !p.isDuck())) count++;
        }
    }
    return count;
}

int Engine::evaluate(const Board& board) {
    // Route to DuckNNUE for duck chess, standard NNUE for regular chess
    if (board.isDuckChess && duckNnue_) {
        return duckNnue_->evaluateQ(board);  // INT16 quantized path
    }
    if (nnue_) {
        return nnue_->evaluateQ(board);
    }

    int mgScore = 0, egScore = 0;
    int phase = 0;

    int whitePawnCnt[8] = {}, blackPawnCnt[8] = {};
    int whiteBishops = 0, blackBishops = 0;
    Square whiteKingSq = {0,0}, blackKingSq = {7,0};
    int mgMobility = 0, egMobility = 0;

    static const int diagDirs[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    static const int straightDirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    static const int allDirs[8][2] = {{1,1},{1,-1},{-1,1},{-1,-1},{1,0},{-1,0},{0,1},{0,-1}};

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            Piece p = board.squares[r][c];
            if (p.isNone()) continue;
            if (p.isDuck()) continue;

            int pt = (int)p.type;
            if (pt < 0 || pt > 6) continue;
            phase += phase_weight[pt];

            int mgPST = (p.color == Color::White) ? mg_tables[pt][7 - r][c]
                                                   : mg_tables[pt][r][c];
            int egPST = (p.color == Color::White) ? eg_tables[pt][7 - r][c]
                                                   : eg_tables[pt][r][c];

            int mgVal = mg_value[pt] + mgPST;
            int egVal = eg_value[pt] + egPST;

            int mob = 0;
            switch (p.type) {
            case PieceType::Knight:
                mob = countKnightMobility(board, {r,c}, p.color);
                if (p.color == Color::White) { mgMobility += mob * 4; egMobility += mob * 3; }
                else                         { mgMobility -= mob * 4; egMobility -= mob * 3; }
                break;
            case PieceType::Bishop:
                mob = countSliderMobility(board, {r,c}, p.color, diagDirs, 4);
                if (p.color == Color::White) { mgMobility += mob * 5; egMobility += mob * 4; }
                else                         { mgMobility -= mob * 5; egMobility -= mob * 4; }
                break;
            case PieceType::Rook:
                mob = countSliderMobility(board, {r,c}, p.color, straightDirs, 4);
                if (p.color == Color::White) { mgMobility += mob * 2; egMobility += mob * 3; }
                else                         { mgMobility -= mob * 2; egMobility -= mob * 3; }
                break;
            case PieceType::Queen:
                mob = countSliderMobility(board, {r,c}, p.color, allDirs, 8);
                if (p.color == Color::White) { mgMobility += mob * 1; egMobility += mob * 1; }
                else                         { mgMobility -= mob * 1; egMobility -= mob * 1; }
                break;
            default: break;
            }

            if (p.color == Color::White) { mgScore += mgVal; egScore += egVal; }
            else                         { mgScore -= mgVal; egScore -= egVal; }

            if (p.type == PieceType::Pawn) {
                if (p.color == Color::White) whitePawnCnt[c]++;
                else blackPawnCnt[c]++;
            }
            if (p.type == PieceType::Bishop) {
                if (p.color == Color::White) whiteBishops++;
                else blackBishops++;
            }
            if (p.type == PieceType::King) {
                if (p.color == Color::White) whiteKingSq = {r, c};
                else blackKingSq = {r, c};
            }
        }
    }

    if (whiteBishops >= 2) { mgScore += 30; egScore += 50; }
    if (blackBishops >= 2) { mgScore -= 30; egScore -= 50; }

    for (int f = 0; f < 8; f++) {
        if (whitePawnCnt[f] > 1) { int e = whitePawnCnt[f]-1; mgScore -= 10*e; egScore -= 20*e; }
        if (blackPawnCnt[f] > 1) { int e = blackPawnCnt[f]-1; mgScore += 10*e; egScore += 20*e; }
    }

    for (int f = 0; f < 8; f++) {
        bool wAdj = (f > 0 && whitePawnCnt[f-1]) || (f < 7 && whitePawnCnt[f+1]);
        bool bAdj = (f > 0 && blackPawnCnt[f-1]) || (f < 7 && blackPawnCnt[f+1]);
        if (whitePawnCnt[f] && !wAdj) { mgScore -= 15; egScore -= 20; }
        if (blackPawnCnt[f] && !bAdj) { mgScore += 15; egScore += 20; }
    }

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            Piece p = board.squares[r][c];
            if (p.isNone() || p.isDuck()) continue;
            if (p.type != PieceType::Pawn) continue;

            if (p.color == Color::White) {
                bool passed = true;
                for (int rr = r + 1; rr < 8 && passed; rr++)
                    for (int dc = -1; dc <= 1 && passed; dc++) {
                        int cc = c + dc;
                        if (cc < 0 || cc > 7) continue;
                        if (board.squares[rr][cc].type == PieceType::Pawn &&
                            board.squares[rr][cc].color == Color::Black)
                            passed = false;
                    }
                if (passed) {
                    int bonus = 10 + r * r * 2;
                    mgScore += bonus / 2; egScore += bonus;
                }
            } else {
                bool passed = true;
                for (int rr = r - 1; rr >= 0 && passed; rr--)
                    for (int dc = -1; dc <= 1 && passed; dc++) {
                        int cc = c + dc;
                        if (cc < 0 || cc > 7) continue;
                        if (board.squares[rr][cc].type == PieceType::Pawn &&
                            board.squares[rr][cc].color == Color::White)
                            passed = false;
                    }
                if (passed) {
                    int bonus = 10 + (7 - r) * (7 - r) * 2;
                    mgScore -= bonus / 2; egScore -= bonus;
                }
            }
        }
    }

    auto pawnShield = [&](Square kSq, Color col, const int pawnCnt[8]) -> int {
        int shield = 0;
        int dir = (col == Color::White) ? 1 : -1;
        for (int dc = -1; dc <= 1; dc++) {
            int fc = kSq.col + dc;
            if (fc < 0 || fc > 7) continue;
            int r1 = kSq.rank + dir;
            int r2 = kSq.rank + dir * 2;
            if (r1 >= 0 && r1 < 8) {
                Piece p = board.squares[r1][fc];
                if (p.type == PieceType::Pawn && p.color == col)
                    shield += 30;
            }
            if (r2 >= 0 && r2 < 8) {
                Piece p = board.squares[r2][fc];
                if (p.type == PieceType::Pawn && p.color == col)
                    shield += 15;
            }
            if (pawnCnt[fc] == 0)
                shield -= 20;
        }
        return shield;
    };

    int whiteKingSafety = pawnShield(whiteKingSq, Color::White, whitePawnCnt);
    int blackKingSafety = pawnShield(blackKingSq, Color::Black, blackPawnCnt);
    mgScore += whiteKingSafety - blackKingSafety;

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            Piece p = board.squares[r][c];
            if (p.isNone() || p.isDuck()) continue;
            if (p.type != PieceType::Rook) continue;
            bool friendlyPawn = (p.color == Color::White) ? whitePawnCnt[c] > 0
                                                          : blackPawnCnt[c] > 0;
            bool enemyPawn    = (p.color == Color::White) ? blackPawnCnt[c] > 0
                                                          : whitePawnCnt[c] > 0;
            int sign = (p.color == Color::White) ? 1 : -1;
            if (!friendlyPawn && !enemyPawn) {
                mgScore += sign * 25; egScore += sign * 15;
            } else if (!friendlyPawn) {
                mgScore += sign * 12; egScore += sign * 8;
            }
        }
    }

    mgScore += mgMobility;
    egScore += egMobility;

    int ph = std::min(phase, TOTAL_PHASE);
    int score = (mgScore * ph + egScore * (TOTAL_PHASE - ph)) / TOTAL_PHASE;

    return (board.turn == Color::White) ? score : -score;
}

// =============================================================
//  MOVE ORDERING
// =============================================================
bool Engine::isCapture(const Board& board, const Move& m) {
    if (!board.getPiece(m.to).isNone()) return true;
    Piece p = board.getPiece(m.from);
    if (p.type == PieceType::Pawn && m.from.col != m.to.col && board.getPiece(m.to).isNone())
        return true;
    return false;
}

int Engine::mvvLva(const Board& board, const Move& m) {
    Piece victim = board.getPiece(m.to);
    Piece attacker = board.getPiece(m.from);
    int victimVal  = victim.isNone() ? 100 : mg_value[(int)victim.type];
    int attackerVal = mg_value[(int)attacker.type];
    return victimVal * 10 - attackerVal;
}

int Engine::scoreMove(const Move& m, const Board& board,
                      int ply, const Move& hashMove) const {
    if (m == hashMove)
        return 10000000;

    if (isCapture(board, m)) {
        int seeVal = see(board, m);
        if (seeVal >= 0)
            return 5000000 + seeVal;
        else
            return -1000000 + seeVal;
    }

    if (m.promotion != PieceType::None)
        return 4000000 + mg_value[(int)m.promotion];

    if (ply < MAX_PLY) {
        if (m == killers_[ply][0]) return 3000000;
        if (m == killers_[ply][1]) return 2900000;
    }

    if (previousMove_.from.isValid()) {
        int ci = (board.turn == Color::White) ? 1 : 0;
        int fSq = previousMove_.from.rank * 8 + previousMove_.from.col;
        int tSq = previousMove_.to.rank * 8 + previousMove_.to.col;
        if (m == countermoves_[ci][fSq][tSq])
            return 2800000;
    }

    int ci = (board.turn == Color::White) ? 0 : 1;
    int fromSq = m.from.rank * 8 + m.from.col;
    int toSq   = m.to.rank * 8 + m.to.col;
    return history_[ci][fromSq][toSq];
}

void Engine::orderMoves(MoveList& moves, const Board& board,
                        int ply, const Move& hashMove) const {
    std::vector<std::pair<int, int>> scored(moves.size());
    for (int i = 0; i < (int)moves.size(); i++)
        scored[i] = { scoreMove(moves[i], board, ply, hashMove), i };

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    MoveList sorted;
    for (int i = 0; i < (int)moves.size(); i++) sorted.add(moves[scored[i].second]);
    moves = std::move(sorted);
}

// =============================================================
//  QUIESCENCE SEARCH
// =============================================================
int Engine::qsearch(Board& board, int alpha, int beta, int ply) {
    nodes_++;
    if (shouldStop()) return 0;

    if (ply >= MAX_PLY + 32) return evaluate(board);

    int standPat = evaluate(board);
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;

    MoveList legalMoves; MoveGen::getLegalMoves(board, legalMoves);
    MoveList captures;
    for (auto& m : legalMoves) {
        if (isCapture(board, m) || m.promotion != PieceType::None)
            captures.push_back(m);
    }

    Move noMove{};
    orderMoves(captures, board, ply, noMove);

    for (auto& m : captures) {
        if (shouldStop()) return 0;

        if (see(board, m) < -50) continue;

        Piece victim = board.getPiece(m.to);
        int delta = victim.isNone() ? SEE_VAL[(int)PieceType::Pawn]
                                    : SEE_VAL[(int)victim.type];
        if (m.promotion != PieceType::None)
            delta += SEE_VAL[(int)m.promotion] - SEE_VAL[(int)PieceType::Pawn];
        if (standPat + delta + 200 < alpha) continue;

        Board::UndoInfo undo;
        board.makeMove(m, undo);
        int score = -qsearch(board, -beta, -alpha, ply + 1);
        board.unmakeMove(m, undo);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

// =============================================================
//  ALPHA-BETA with Null Move, LMR, PVS, Futility, RFP, LMP, IID
// =============================================================
int Engine::search(Board& board, int depth, int alpha, int beta,
                   int ply, bool doNull, bool isPV) {
    if (shouldStop()) return 0;
    nodes_++;

    pvLength_[ply] = ply;

    if (ply >= MAX_PLY) return evaluate(board);

    if (depth <= 0) return qsearch(board, alpha, beta, ply);

    uint64_t hash = computeHash(board);

    if (ply > 0) {
        for (const auto& h : gameHistory_) {
            if (h == hash) return drawScore();
        }
        for (int i = 0; i < ply; i++) {
            if (searchStack_[i] == hash) return drawScore();
        }
    }
    searchStack_[ply] = hash;

    if (board.halfMoveClock >= 100) return drawScore();

    size_t ttIdx = hash % activeTT().size();
    TTEntry& tte = activeTT()[ttIdx];
    Move hashMove{};

    if (tte.key == hash) {
        hashMove = tte.best;
        if (tte.depth >= depth && !isPV) {
            if (tte.flag == 0) return tte.score;
            if (tte.flag == 1 && tte.score >= beta)  return tte.score;
            if (tte.flag == 2 && tte.score <= alpha) return tte.score;
        }
    }

    bool inCheck = MoveGen::isInCheck(board, board.turn);

    if (inCheck && ply < MAX_PLY - 2) depth++;

    int staticEval = evaluate(board);

    if (!isPV && !inCheck && depth <= 3 && std::abs(beta) < MATE_SCORE - 100) {
        int rfpMargin = 120 * depth;
        if (staticEval - rfpMargin >= beta)
            return staticEval - rfpMargin;
    }

    if (doNull && !isPV && !inCheck && depth >= 3) {
        bool hasNonPawn = false;
        for (int r = 0; r < 8 && !hasNonPawn; r++)
            for (int c = 0; c < 8 && !hasNonPawn; c++) {
                Piece p = board.squares[r][c];
                if (p.color == board.turn && p.type != PieceType::Pawn &&
                    p.type != PieceType::King && !p.isNone() && !p.isDuck())
                    hasNonPawn = true;
            }

        if (hasNonPawn) {
            int R = 3 + depth / 6;
            Board nullBoard = board;
            nullBoard.turn = (board.turn == Color::White) ? Color::Black : Color::White;
            nullBoard.enPassantTarget = {-1, -1};

            int nullScore = -search(nullBoard, depth - 1 - R, -beta, -beta + 1,
                                    ply + 1, false, false);
            if (nullScore >= beta)
                return beta;
        }
    }

    MoveList moves; MoveGen::getLegalMoves(board, moves);

    if (moves.empty()) {
        if (inCheck) return -(MATE_SCORE - ply);
        return drawScore();
    }

    if (isPV && !(hashMove.from.isValid() && hashMove.to.isValid()) && depth >= 4) {
        search(board, depth - 2, alpha, beta, ply, false, true);
        if (activeTT()[ttIdx].key == hash)
            hashMove = activeTT()[ttIdx].best;
    }

    orderMoves(moves, board, ply, hashMove);

    bool canFutility = !isPV && !inCheck && depth <= 3 &&
                       std::abs(alpha) < MATE_SCORE - 100 &&
                       staticEval + FUTILITY_MARGIN[depth] <= alpha;

    Move bestMove = moves[0];
    uint8_t ttFlag = 2;
    int bestScore = -INF;
    int quietsSearched = 0;

    Move quietsTriedArr[64];
    int quietsTriedCnt = 0;

    for (int i = 0; i < (int)moves.size(); i++) {
        if (shouldStop()) return 0;

        const Move& m = moves[i];
        bool isCap    = isCapture(board, m);
        bool isPromo  = (m.promotion != PieceType::None);
        bool isQuiet  = !isCap && !isPromo;

        if (isQuiet) quietsSearched++;

        if (canFutility && isQuiet && i > 0 && bestScore > -MATE_SCORE + 100) {
            bool isKiller = ply < MAX_PLY &&
                (m == killers_[ply][0] || m == killers_[ply][1]);
            if (!isKiller) continue;
        }

        if (!isPV && !inCheck && depth <= 3 && isQuiet &&
            quietsSearched > LMP_THRESHOLD[depth] &&
            bestScore > -MATE_SCORE + 100) {
            continue;
        }

        if (!isPV && depth <= 2 && isCap && see(board, m) < -100 * depth) {
            continue;
        }

        Board::UndoInfo undo;
        board.makeMove(m, undo);

        int score;

        if (i == 0) {
            score = -search(board, depth - 1, -beta, -alpha, ply + 1, true, isPV);
        } else {
            int reduction = 0;
            if (depth >= 3 && i >= 3 && isQuiet && !inCheck) {
                reduction = 1 + (int)(std::log(depth) * std::log(i + 1) / 2.5);

                if (ply < MAX_PLY &&
                    (m == killers_[ply][0] || m == killers_[ply][1]))
                    reduction = std::max(0, reduction - 1);

                int ci2 = (board.turn == Color::White) ? 0 : 1;
                int fSq = m.from.rank * 8 + m.from.col;
                int tSq = m.to.rank * 8 + m.to.col;
                if (history_[ci2][fSq][tSq] > 5000)
                    reduction = std::max(0, reduction - 1);

                reduction = std::min(reduction, depth - 2);
                reduction = std::max(reduction, 0);
            }

            score = -search(board, depth - 1 - reduction, -alpha - 1, -alpha,
                            ply + 1, true, false);

            if (reduction > 0 && score > alpha)
                score = -search(board, depth - 1, -alpha - 1, -alpha,
                                ply + 1, true, false);

            if (score > alpha && score < beta)
                score = -search(board, depth - 1, -beta, -alpha,
                                ply + 1, true, true);
        }

        board.unmakeMove(m, undo);

        if (score > bestScore) {
            bestScore = score;
            bestMove  = m;

            if (score > alpha) {
                alpha  = score;
                ttFlag = 0;

                pvTable_[ply][ply] = m;
                for (int j = ply + 1; j < pvLength_[ply + 1]; j++)
                    pvTable_[ply][j] = pvTable_[ply + 1][j];
                pvLength_[ply] = pvLength_[ply + 1];

                if (alpha >= beta) {
                    ttFlag = 1;

                    if (isQuiet && ply < MAX_PLY) {
                        killers_[ply][1] = killers_[ply][0];
                        killers_[ply][0] = m;

                        int ci3 = (board.turn == Color::White) ? 0 : 1;
                        int fSq = m.from.rank * 8 + m.from.col;
                        int tSq = m.to.rank * 8 + m.to.col;

                        int bonus = depth * depth;
                        history_[ci3][fSq][tSq] += bonus;
                        if (history_[ci3][fSq][tSq] > 1000000)
                            history_[ci3][fSq][tSq] = 1000000;

                        for (int q = 0; q < quietsTriedCnt; q++) {
                            int qf = quietsTriedArr[q].from.rank * 8 + quietsTriedArr[q].from.col;
                            int qt = quietsTriedArr[q].to.rank * 8 + quietsTriedArr[q].to.col;
                            history_[ci3][qf][qt] -= bonus;
                            if (history_[ci3][qf][qt] < -1000000)
                                history_[ci3][qf][qt] = -1000000;
                        }

                        if (previousMove_.from.isValid()) {
                            int prevC = (board.turn == Color::White) ? 1 : 0;
                            int prevF = previousMove_.from.rank * 8 + previousMove_.from.col;
                            int prevT = previousMove_.to.rank * 8 + previousMove_.to.col;
                            countermoves_[prevC][prevF][prevT] = m;
                        }
                    }
                    break;
                }
            }
        }

        if (isQuiet && quietsTriedCnt < 64)
            quietsTriedArr[quietsTriedCnt++] = m;
    }

    if (tte.key != hash || tte.gen != ttGen_ || depth >= tte.depth) {
        tte.key   = hash;
        tte.score = bestScore;
        tte.depth = (int16_t)depth;
        tte.flag  = ttFlag;
        tte.best  = bestMove;
        tte.gen   = ttGen_;
    }

    return bestScore;
}

#ifdef DUCK_CHESS
// =============================================================
//  DUCK CHESS — Duck placement heuristics
// =============================================================
int Engine::scoreDuckPlacement(const Board& board, Square duckSq, Color myColor) const {
    int score = 0;
    Color opponent = (myColor == Color::White) ? Color::Black : Color::White;

    Square oppKingSq = {-1, -1};
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            if (board.squares[r][c].type == PieceType::King && board.squares[r][c].color == opponent)
                oppKingSq = {r, c};

    if (oppKingSq.isValid()) {
        int dr = abs(duckSq.rank - oppKingSq.rank);
        int dc = abs(duckSq.col - oppKingSq.col);
        int dist = dr + dc;

        if (dr <= 1 && dc <= 1) score += 100;
        else if (dist <= 3) score += 50;
        else if (dist <= 5) score += 20;
    }

    int centerDist = abs(duckSq.rank - 3) + abs(duckSq.col - 3);
    score += (7 - centerDist) * 5;

    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            Piece p = board.squares[r][c];
            if (p.isNone() || p.isDuck() || p.color != opponent) continue;

            if (p.type == PieceType::Rook || p.type == PieceType::Queen) {
                if (duckSq.rank == r || duckSq.col == c) score += 15;
            }
            if (p.type == PieceType::Bishop || p.type == PieceType::Queen) {
                if (abs(duckSq.rank - r) == abs(duckSq.col - c) && duckSq.rank != r)
                    score += 15;
            }
        }

    return score;
}

void Engine::orderDuckPlacements(SquareList& placements, const Board& board, Color myColor) const {
    std::vector<std::pair<int, int>> scored(placements.size());
    for (int i = 0; i < (int)placements.size(); i++)
        scored[i] = { scoreDuckPlacement(board, placements[i], myColor), i };

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    SquareList sorted;
    for (int i = 0; i < (int)placements.size(); i++)
        sorted.add(placements[scored[i].second]);
    placements = sorted;
}

#endif // DUCK_CHESS


#ifdef DUCK_CHESS
// =============================================================
//  DUCK CHESS SEARCH
// =============================================================
int Engine::searchDuck(Board& board, int depth, int alpha, int beta, int ply) {
    if (shouldStop()) return 0;
    nodes_++;

    if (ply >= MAX_PLY) return evaluate(board);
    if (depth <= 0) return evaluate(board);

    Color us = board.turn;
    Color them = (us == Color::White) ? Color::Black : Color::White;
    if (MoveGen::isKingCaptured(board, us))
        return -(MATE_SCORE - ply);

    MoveList chessMoves; MoveGen::getDuckChessMoves(board, chessMoves);
    if (chessMoves.empty())
        return -(MATE_SCORE - ply);

    Move noMove{};
    orderMoves(chessMoves, board, ply, noMove);

    int bestScore = -INF;

    for (int i = 0; i < (int)chessMoves.size(); i++) {
        if (shouldStop()) return 0;

        Board temp = board;
        temp.applyMove(chessMoves[i]);

        if (MoveGen::isKingCaptured(temp, them))
            return MATE_SCORE - ply;

        SquareList duckSquares; MoveGen::getDuckPlacements(temp, duckSquares);
        orderDuckPlacements(duckSquares, temp, us);

        int maxDucks = (depth <= 1) ? 12 : (depth <= 2) ? 18 : (int)duckSquares.size();
        maxDucks = std::min(maxDucks, (int)duckSquares.size());

        for (int d = 0; d < maxDucks; d++) {
            if (shouldStop()) return 0;

            Board duckBoard = temp;
            duckBoard.placeDuck(duckSquares[d]);

            int score = -searchDuck(duckBoard, depth - 1, -beta, -alpha, ply + 1);

            if (score > bestScore) {
                bestScore = score;
                if (score > alpha) {
                    alpha = score;
                    if (alpha >= beta) goto nextChessMove;
                }
            }
        }
        nextChessMove:;
        if (alpha >= beta) break;
    }

    return bestScore;
}

// =============================================================
//  getBestMove — Iterative Deepening with Time Management
//  NOW: uses soft/hard limits, outputs info at each depth,
//       extends time when best move is unstable

#endif // DUCK_CHESS

// =============================================================
Move Engine::getBestMove(Board& board, int maxDepth) {
    stop_.store(false, std::memory_order_relaxed);
    nodes_ = 0;
    searchStart_ = std::chrono::steady_clock::now();
    ttGen_++;

    std::memset(killers_, 0, sizeof(killers_));
    for (int c = 0; c < 2; c++)
        for (int f = 0; f < 64; f++)
            for (int t = 0; t < 64; t++)
                history_[c][f][t] /= 2;

    // =========================================================
    //  LAZY SMP — spawn helper threads sharing our TT
    // =========================================================
    std::vector<std::unique_ptr<Engine>> helpers;
    std::vector<std::thread> helperThreads;
    if (numThreads_ > 1) {
        int numHelpers = numThreads_ - 1;
        helpers.reserve(numHelpers);
        helperThreads.reserve(numHelpers);
        for (int i = 0; i < numHelpers; ++i) {
            auto h = std::make_unique<Engine>(1);  // tiny own TT (unused)
            h->setNNUE(nnue_);
            h->setDuckNNUE(duckNnue_);
            h->setSharedTT(&activeTT());           // share main thread's TT
            h->setTimeLimits(softLimitMs_, hardLimitMs_);
            h->setPositionHistory(gameHistory_);
            h->stop_.store(false, std::memory_order_relaxed);
            h->searchStart_ = searchStart_;
            h->ttGen_ = ttGen_;
            // Helpers search at slightly different depths to avoid duplication
            int helperDepth = maxDepth + (i % 2 == 0 ? 1 : -1);
            helperDepth = std::max(1, helperDepth);
            Engine* hPtr = h.get();
            Board boardCopy = board;
            helperThreads.emplace_back([hPtr, boardCopy, helperDepth]() mutable {
                hPtr->getBestMove(boardCopy, helperDepth);
            });
            helpers.push_back(std::move(h));
        }
    }

    // =========================================================
    //  DUCK CHESS — completely separate root search
    // =========================================================
#ifdef DUCK_CHESS
    if (board.isDuckChess) {
        MoveList chessMoves; MoveGen::getDuckChessMoves(board, chessMoves);
        if (chessMoves.empty()) return Move{};

        Move bestMove = chessMoves[0];
        int bestScore = -INF;

        {
            std::lock_guard<std::mutex> lock(livePVMutex_);
            livePV_.clear();
        }
        liveDepth_.store(0, std::memory_order_relaxed);
        liveEval_.store(0, std::memory_order_relaxed);

        for (int depth = 1; depth <= maxDepth; depth++) {
            if (stop_.load()) break;

            // Use SOFT limit for "should I start a new iteration?"
            if (depth > 1) {
                if (elapsedMs() > softLimitMs_) break;
            }

            int alpha = -INF, beta = INF;
            int bestScoreIter = -INF;
            Move bestMoveIter = chessMoves[0];

            for (int i = 0; i < (int)chessMoves.size(); i++) {
                if (stop_.load()) break;

                Board temp = board;
                temp.applyMove(chessMoves[i]);

                Color opponent = (board.turn == Color::White) ? Color::Black : Color::White;
                if (MoveGen::isKingCaptured(temp, opponent)) {
                    SquareList ducks; MoveGen::getDuckPlacements(temp, ducks);
                    Move winMove = chessMoves[i];
                    winMove.duckTo = ducks.empty() ? Square{0,0} : ducks[0];
                    return winMove;
                }

                SquareList duckSquares; MoveGen::getDuckPlacements(temp, duckSquares);
                orderDuckPlacements(duckSquares, temp, board.turn);

                for (int d = 0; d < (int)duckSquares.size(); d++) {
                    if (stop_.load()) break;

                    Board duckBoard = temp;
                    duckBoard.placeDuck(duckSquares[d]);

                    int score = -searchDuck(duckBoard, depth - 1, -beta, -alpha, 1);

                    if (score > bestScoreIter) {
                        bestScoreIter = score;
                        bestMoveIter = chessMoves[i];
                        bestMoveIter.duckTo = duckSquares[d];
                        if (score > alpha) alpha = score;
                        if (alpha >= beta) break;
                    }
                }
                if (alpha >= beta) break;
            }

            if (!stop_.load()) {
                bestMove = bestMoveIter;
                bestScore = bestScoreIter;
                lastDepth_ = depth;

                liveDepth_.store(depth, std::memory_order_relaxed);
                int whiteEval = (board.turn == Color::White) ? bestScore : -bestScore;
                liveEval_.store(whiteEval, std::memory_order_relaxed);

                {
                    std::lock_guard<std::mutex> lock(livePVMutex_);
                    livePV_.clear();
                    livePV_.push_back(bestMove);
                }

                // Info output for duck chess
                if (onInfoCallback) {
                    int64_t elapsed = elapsedMs();
                    uint64_t nps = elapsed > 0 ? (nodes_ * 1000 / elapsed) : 0;
                    std::string pvStr = pvToUCI({bestMove});
                    onInfoCallback(depth, whiteEval, nodes_, nps, elapsed, pvStr, 1);
                }
            }
        }

        return bestMove;
    }
#endif // DUCK_CHESS

    // =========================================================
    //  STANDARD CHESS — iterative deepening with soft/hard time
    // =========================================================
    MoveList legalMoves; MoveGen::getLegalMoves(board, legalMoves);
    if (legalMoves.empty()) return Move{};
    if (legalMoves.size() == 1) return legalMoves[0];

    Move bestMove = legalMoves[0];
    Move prevBestMove{};              // NEW: track best move stability
    int  bestMoveStableCount = 0;     // NEW: how many iterations the best move hasn't changed
    int  prevScore = 0;
    lastPV_.clear();
    lastDepth_ = 0;
    std::memset(pvLength_, 0, sizeof(pvLength_));
    std::memset(pvTable_, 0, sizeof(pvTable_));
    std::memset(searchStack_, 0, sizeof(searchStack_));

    rootEval_ = evaluate(board);

    {
        std::lock_guard<std::mutex> lock(livePVMutex_);
        livePV_.clear();
    }
    liveDepth_.store(0, std::memory_order_relaxed);
    liveEval_.store(0, std::memory_order_relaxed);

    for (int depth = 1; depth <= maxDepth; depth++) {
        if (stop_.load(std::memory_order_relaxed)) break;

        // === SOFT TIME CHECK ===
        // Don't start a new iteration if we've used more than the soft limit.
        // Exception: extend time if best move is unstable (changed recently).
        if (depth > 1) {
            int64_t elapsed = elapsedMs();
            int effectiveSoft = softLimitMs_;

            // If best move changed in last 2 iterations, extend soft limit by 50%
            if (bestMoveStableCount < 2)
                effectiveSoft = softLimitMs_ * 3 / 2;

            // If best move has been stable for 4+ iterations, reduce soft limit
            if (bestMoveStableCount >= 4)
                effectiveSoft = softLimitMs_ * 2 / 3;

            // Never exceed hard limit
            effectiveSoft = std::min(effectiveSoft, hardLimitMs_);

            if (elapsed > effectiveSoft) break;
        }

        int alpha, beta;

        if (depth >= 4) {
            alpha = prevScore - 25;
            beta  = prevScore + 25;
        } else {
            alpha = -INF;
            beta  = INF;
        }

        MoveList moves; MoveGen::getLegalMoves(board, moves);
        uint64_t hash = computeHash(board);
        size_t ttIdx = hash % activeTT().size();
        Move hashMove = (activeTT()[ttIdx].key == hash) ? activeTT()[ttIdx].best : Move{};
        orderMoves(moves, board, 0, hashMove);

        int bestScoreIter = -INF;
        Move bestMoveIter = moves[0];

        for (int i = 0; i < (int)moves.size(); i++) {
            if (stop_.load(std::memory_order_relaxed)) break;

            Board::UndoInfo undo;
            board.makeMove(moves[i], undo);
            previousMove_ = moves[i];

            int score;
            if (i == 0) {
                score = -search(board, depth - 1, -beta, -alpha, 1, true, true);
            } else {
                score = -search(board, depth - 1, -alpha - 1, -alpha, 1, true, false);
                if (score > alpha && score < beta)
                    score = -search(board, depth - 1, -beta, -alpha, 1, true, true);
            }

            board.unmakeMove(moves[i], undo);

            if (score > bestScoreIter) {
                bestScoreIter = score;
                bestMoveIter  = moves[i];
                if (score > alpha) alpha = score;
                if (alpha >= beta) break;
            }
        }

        // Aspiration window failed — re-search with full window
        if (!stop_.load(std::memory_order_relaxed) && depth >= 4 &&
            (bestScoreIter <= prevScore - 25 || bestScoreIter >= prevScore + 25)) {

            alpha = -INF;
            beta  = INF;
            bestScoreIter = -INF;

            MoveList moves2; MoveGen::getLegalMoves(board, moves2);
            orderMoves(moves2, board, 0, hashMove);

            for (int i = 0; i < (int)moves2.size(); i++) {
                if (stop_.load(std::memory_order_relaxed)) break;

                Board::UndoInfo undo2;
                board.makeMove(moves2[i], undo2);
                previousMove_ = moves2[i];

                int score;
                if (i == 0) {
                    score = -search(board, depth - 1, -beta, -alpha, 1, true, true);
                } else {
                    score = -search(board, depth - 1, -alpha - 1, -alpha, 1, true, false);
                    if (score > alpha && score < beta)
                        score = -search(board, depth - 1, -beta, -alpha, 1, true, true);
                }

                board.unmakeMove(moves2[i], undo2);

                if (score > bestScoreIter) {
                    bestScoreIter = score;
                    bestMoveIter  = moves2[i];
                    if (score > alpha) alpha = score;
                    if (alpha >= beta) break;
                }
            }
        }

        if (!stop_.load(std::memory_order_relaxed)) {
            // === TRACK BEST MOVE STABILITY ===
            if (bestMoveIter == prevBestMove) {
                bestMoveStableCount++;
            } else {
                bestMoveStableCount = 0;
            }
            prevBestMove = bestMoveIter;

            bestMove  = bestMoveIter;
            prevScore = bestScoreIter;
            rootEval_ = bestScoreIter;
            lastDepth_ = depth;

            lastPV_.clear();
            for (int j = 0; j < pvLength_[0]; j++)
                lastPV_.push_back(pvTable_[0][j]);
            if (lastPV_.empty())
                lastPV_.push_back(bestMove);

            {
                std::lock_guard<std::mutex> lock(livePVMutex_);
                livePV_ = lastPV_;
            }

            liveDepth_.store(depth, std::memory_order_relaxed);
            int whiteEval = (board.turn == Color::White) ? prevScore : -prevScore;
            liveEval_.store(whiteEval, std::memory_order_relaxed);

            // === OUTPUT INFO LINE AT EACH DEPTH ===
            if (onInfoCallback) {
                int64_t elapsed = elapsedMs();
                uint64_t nps = elapsed > 0 ? (nodes_ * 1000 / elapsed) : 0;
                std::string pvStr = pvToUCI(lastPV_);
                onInfoCallback(depth, whiteEval, nodes_, nps, elapsed, pvStr, 1);
            }
        }
    }

    // Stop helpers and join threads, aggregate node counts
    if (!helperThreads.empty()) {
        for (auto& h : helpers) h->stop_.store(true, std::memory_order_relaxed);
        for (auto& t : helperThreads) if (t.joinable()) t.join();
        for (auto& h : helpers) nodes_ += h->nodes_;
        cumulativeNodes_ += nodes_;
    } else {
        cumulativeNodes_ += nodes_;
    }

    return bestMove;
}
