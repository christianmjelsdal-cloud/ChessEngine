#include "Board.h"
#include "Zobrist.h"
#include <iostream>
#include <sstream>

// Forward declaration — defined later in this file
static int piecePhaseWeight(PieceType pt);

Board::Board() {
    setStartingPosition();
}

void Board::clearBoard() {
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            squares[r][c] = Piece{};
    duckSquare = { -1, -1 };
    // Reset Automate Chess state
    isAutomateChess = false;
    automateSetupComplete = false;
    automateBudget[0] = automateBudget[1] = 35;
    automatePawnsPlaced[0] = automatePawnsPlaced[1] = 0;
    automateKingPlaced[0] = automateKingPlaced[1] = false;
    automateSetupTurn = Color::White;
}

void Board::setStartingPosition() {
    clearBoard();

    // Piece order for back rank
    PieceType backRank[] = {
        PieceType::Rook, PieceType::Knight, PieceType::Bishop,
        PieceType::Queen, PieceType::King, PieceType::Bishop,
        PieceType::Knight, PieceType::Rook
    };

    for (int c = 0; c < 8; c++) {
        // White back rank (rank 0)
        squares[0][c] = { backRank[c], Color::White };
        // White pawns (rank 1)
        squares[1][c] = { PieceType::Pawn, Color::White };

        // Black pawns (rank 6)
        squares[6][c] = { PieceType::Pawn, Color::Black };
        // Black back rank (rank 7)
        squares[7][c] = { backRank[c], Color::Black };
    }

    // Duck starts off the board
    duckSquare = { -1, -1 };

    // Reset game state
    turn = Color::White;
    castlingRights[0][0] = castlingRights[0][1] = true;
    castlingRights[1][0] = castlingRights[1][1] = true;
    enPassantTarget = { -1, -1 };
    halfMoveClock = 0;
    fullMoveNumber = 1;
    hash = 0;

    recomputeBitboards();
}

Piece Board::getPiece(Square sq) const {
    return squares[sq.rank][sq.col];
}

void Board::setPiece(Square sq, Piece piece) {
    squares[sq.rank][sq.col] = piece;
}

bool Board::isDuckSquare(Square sq) const {
    return isDuckChess && duckSquare.isValid() &&
           duckSquare.rank == sq.rank && duckSquare.col == sq.col;
}

bool Board::isDuckSquare(int rank, int col) const {
    return isDuckChess && duckSquare.isValid() &&
           duckSquare.rank == rank && duckSquare.col == col;
}

void Board::placeDuck(Square sq) {
    // Remove duck from old position
    if (duckSquare.isValid()) {
        squares[duckSquare.rank][duckSquare.col] = Piece{};
    }
    // Place duck on new position
    duckSquare = sq;
#ifdef DUCK_CHESS
    squares[sq.rank][sq.col] = { PieceType::Duck, Color::White }; // color is irrelevant
#endif
}

void Board::printBoard() const {
    std::cout << "\n  a b c d e f g h\n";
    for (int r = 7; r >= 0; r--) {
        std::cout << (r + 1) << " ";
        for (int c = 0; c < 8; c++) {
            Piece p = squares[r][c];
            char symbol = '.';
            if (p.type != PieceType::None) {
                switch (p.type) {
                case PieceType::Pawn:   symbol = 'P'; break;
                case PieceType::Knight: symbol = 'N'; break;
                case PieceType::Bishop: symbol = 'B'; break;
                case PieceType::Rook:   symbol = 'R'; break;
                case PieceType::Queen:  symbol = 'Q'; break;
                case PieceType::King:   symbol = 'K'; break;
                #ifdef DUCK_CHESS
                case PieceType::Duck:   symbol = '@'; break; // duck
#endif
                default: break;
                }
                if (p.color == Color::Black
#ifdef DUCK_CHESS
                    && p.type != PieceType::Duck
#endif
                    )
                    symbol = tolower(symbol);
            }
            std::cout << symbol << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void Board::applyMove(const Move& move) {
    Piece moving = getPiece(move.from);

    // ── Incremental Zobrist hash update ──────────────────────────────────────
    // Remove old castling/EP contributions before state changes
    if (Zobrist::ready) {
        int oldCi = (castlingRights[0][0] ? 1 : 0)
                  | (castlingRights[0][1] ? 2 : 0)
                  | (castlingRights[1][0] ? 4 : 0)
                  | (castlingRights[1][1] ? 8 : 0);
        hash ^= Zobrist::castle[oldCi];
        if (enPassantTarget.isValid())
            hash ^= Zobrist::ep[enPassantTarget.col];
        if (turn == Color::Black)
            hash ^= Zobrist::side;
        // Remove moving piece from its source square
        if (!moving.isNone() && !moving.isDuck())
            hash ^= Zobrist::piece[(int)moving.color][(int)moving.type]
                                  [move.from.rank * 8 + move.from.col];
        // Remove captured piece (if any)
        Piece captured = getPiece(move.to);
        if (!captured.isNone() && !captured.isDuck())
            hash ^= Zobrist::piece[(int)captured.color][(int)captured.type]
                                  [move.to.rank * 8 + move.to.col];
    }

    // Update half-move clock (for 50-move rule)
    bool isCapture = !getPiece(move.to).isNone() && !getPiece(move.to).isDuck();
    if (!isCapture && moving.type == PieceType::Pawn &&
        move.to.rank == enPassantTarget.rank &&
        move.to.col == enPassantTarget.col) {
        isCapture = true; // en passant is a capture
    }
    if (moving.type == PieceType::Pawn || isCapture) {
        halfMoveClock = 0;
    } else {
        halfMoveClock++;
    }

    // Update full move number (after black moves)
    if (turn == Color::Black)
        fullMoveNumber++;

    // En passant capture
    if (moving.type == PieceType::Pawn &&
        move.to.rank == enPassantTarget.rank &&
        move.to.col == enPassantTarget.col) {
        int capturedRank = move.from.rank;
        squares[capturedRank][move.to.col] = Piece{};
    }

    // Set new en passant target
    enPassantTarget = { -1, -1 };
    if (moving.type == PieceType::Pawn &&
        abs(move.to.rank - move.from.rank) == 2) {
        enPassantTarget = {
            (move.from.rank + move.to.rank) / 2,
            move.from.col
        };
    }

    // Castling — move the rook
    if (moving.type == PieceType::King) {
        int backRank = move.from.rank;
        if (move.to.col - move.from.col == 2) { // kingside
            squares[backRank][5] = squares[backRank][7];
            squares[backRank][7] = Piece{};
        }
        else if (move.from.col - move.to.col == 2) { // queenside
            squares[backRank][3] = squares[backRank][0];
            squares[backRank][0] = Piece{};
        }
        int ci = (moving.color == Color::White) ? 0 : 1;
        castlingRights[ci][0] = castlingRights[ci][1] = false;
    }

    // Revoke castling rights if rook moves or is captured
    if (moving.type == PieceType::Rook) {
        int ci = (moving.color == Color::White) ? 0 : 1;
        if (move.from.col == 7) castlingRights[ci][0] = false;
        if (move.from.col == 0) castlingRights[ci][1] = false;
    }
    {
        Piece captured = getPiece(move.to);
        if (captured.type == PieceType::Rook) {
            int ci = (captured.color == Color::White) ? 0 : 1;
            if (move.to.rank == (captured.color == Color::White ? 0 : 7)) {
                if (move.to.col == 7) castlingRights[ci][0] = false;
                if (move.to.col == 0) castlingRights[ci][1] = false;
            }
        }
    }

    // Move the piece
    setPiece(move.to, moving);
    setPiece(move.from, Piece{});

    // Promotion
    if (move.promotion != PieceType::None)
        squares[move.to.rank][move.to.col].type = move.promotion;

    // Handle duck placement if specified in the move (used by engine)
#ifdef DUCK_CHESS
    if (isDuckChess && move.duckTo.isValid()) {
        placeDuck(move.duckTo);
    }
#endif

    // Flip turn
    turn = (turn == Color::White) ? Color::Black : Color::White;

    // ── Finalize incremental hash ─────────────────────────────────────────────
    if (Zobrist::ready) {
        // En passant: remove the captured pawn (it's NOT on move.to)
        if (moving.type == PieceType::Pawn &&
            move.to.rank != move.from.rank &&
            squares[move.from.rank][move.to.col].isNone()) {
            // The EP pawn was already cleared from squares[] above;
            // XOR it out of the hash using the opponent's pawn key
            Color opp = (moving.color == Color::White) ? Color::Black : Color::White;
            hash ^= Zobrist::piece[(int)opp][(int)PieceType::Pawn]
                                  [move.from.rank * 8 + move.to.col];
        }
        // Castling: update rook hash
        if (moving.type == PieceType::King) {
            int backRank = move.from.rank;
            Color col = moving.color;
            if (move.to.col - move.from.col == 2) { // kingside
                hash ^= Zobrist::piece[(int)col][(int)PieceType::Rook][backRank * 8 + 7];
                hash ^= Zobrist::piece[(int)col][(int)PieceType::Rook][backRank * 8 + 5];
            } else if (move.from.col - move.to.col == 2) { // queenside
                hash ^= Zobrist::piece[(int)col][(int)PieceType::Rook][backRank * 8 + 0];
                hash ^= Zobrist::piece[(int)col][(int)PieceType::Rook][backRank * 8 + 3];
            }
        }
        // Add moving piece at destination (after promotion, use promoted type)
        PieceType finalType = (move.promotion != PieceType::None) ? move.promotion : moving.type;
        hash ^= Zobrist::piece[(int)moving.color][(int)finalType]
                              [move.to.rank * 8 + move.to.col];
        // New castling rights
        int newCi = (castlingRights[0][0] ? 1 : 0)
                  | (castlingRights[0][1] ? 2 : 0)
                  | (castlingRights[1][0] ? 4 : 0)
                  | (castlingRights[1][1] ? 8 : 0);
        hash ^= Zobrist::castle[newCi];
        if (enPassantTarget.isValid())
            hash ^= Zobrist::ep[enPassantTarget.col];
        if (turn == Color::Black)  // turn already flipped above
            hash ^= Zobrist::side;
    }

    // Rebuild bitboards from squares[][] (needed by getLegalMoves and evaluate)
    recomputeBitboards();
}

bool Board::isSquareAttacked(Square sq, Color byColor) const {
    // Check all opponent piece attacks on this square

    // Pawn attacks
    int pawnDir = (byColor == Color::White) ? -1 : 1; // direction pawns attack FROM
    for (int dc : {-1, 1}) {
        int pr = sq.rank + pawnDir;
        int pc = sq.col + dc;
        if (pr >= 0 && pr < 8 && pc >= 0 && pc < 8) {
            Piece p = squares[pr][pc];
            if (p.type == PieceType::Pawn && p.color == byColor)
                return true;
        }
    }

    // Knight attacks
    int kOff[8][2] = {{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
    for (auto& o : kOff) {
        int nr = sq.rank + o[0], nc = sq.col + o[1];
        if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            Piece p = squares[nr][nc];
            if (p.type == PieceType::Knight && p.color == byColor)
                return true;
        }
    }

    // Sliding attacks (bishop/rook/queen)
    auto checkSlider = [&](const int dirs[][2], int numDirs, PieceType slider1, PieceType slider2) -> bool {
        for (int i = 0; i < numDirs; i++) {
            int nr = sq.rank + dirs[i][0], nc = sq.col + dirs[i][1];
            while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                // Duck blocks line of sight
                if (isDuckSquare(nr, nc)) break;
                Piece p = squares[nr][nc];
                if (!p.isNone()) {
                    if (p.color == byColor && (p.type == slider1 || p.type == slider2))
                        return true;
                    break;
                }
                nr += dirs[i][0]; nc += dirs[i][1];
            }
        }
        return false;
    };

    const int diagDirs[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    const int straightDirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

    if (checkSlider(diagDirs, 4, PieceType::Bishop, PieceType::Queen)) return true;
    if (checkSlider(straightDirs, 4, PieceType::Rook, PieceType::Queen)) return true;

    // King attacks
    for (int dr = -1; dr <= 1; dr++)
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;
            int nr = sq.rank + dr, nc = sq.col + dc;
            if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                Piece p = squares[nr][nc];
                if (p.type == PieceType::King && p.color == byColor)
                    return true;
            }
        }

    return false;
}

// ── FEN support ──────────────────────────────────────────────────

std::string Board::toFEN() const {
    std::string fen;

    // Piece placement
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int c = 0; c < 8; c++) {
            Piece p = squares[r][c];
            if (p.isNone() || p.isDuck()) {
                empty++;
            } else {
                if (empty > 0) { fen += std::to_string(empty); empty = 0; }
                char ch = '.';
                switch (p.type) {
                    case PieceType::Pawn:   ch = 'P'; break;
                    case PieceType::Knight: ch = 'N'; break;
                    case PieceType::Bishop: ch = 'B'; break;
                    case PieceType::Rook:   ch = 'R'; break;
                    case PieceType::Queen:  ch = 'Q'; break;
                    case PieceType::King:   ch = 'K'; break;
                    default: break;
                }
                if (p.color == Color::Black) ch = static_cast<char>(tolower(ch));
                fen += ch;
            }
        }
        if (empty > 0) fen += std::to_string(empty);
        if (r > 0) fen += '/';
    }

    // Side to move
    fen += (turn == Color::White) ? " w " : " b ";

    // Castling
    std::string castling;
    if (castlingRights[0][0]) castling += 'K';
    if (castlingRights[0][1]) castling += 'Q';
    if (castlingRights[1][0]) castling += 'k';
    if (castlingRights[1][1]) castling += 'q';
    fen += castling.empty() ? "-" : castling;

    // En passant
    if (enPassantTarget.isValid()) {
        fen += ' ';
        fen += static_cast<char>('a' + enPassantTarget.col);
        fen += static_cast<char>('1' + enPassantTarget.rank);
    } else {
        fen += " -";
    }

    // Half-move clock and full move number
    fen += ' ' + std::to_string(halfMoveClock);
    fen += ' ' + std::to_string(fullMoveNumber);

    return fen;
}

bool Board::fromFEN(const std::string& fen) {
    clearBoard();

    std::istringstream iss(fen);
    std::string pieces, turnStr, castling, ep, halfmoveStr, fullmoveStr;
    if (!(iss >> pieces >> turnStr >> castling >> ep)) return false;
    iss >> halfmoveStr >> fullmoveStr; // optional

    int rank = 7, col = 0;
    for (char c : pieces) {
        if (c == '/') {
            rank--;
            col = 0;
        }
        else if (c >= '1' && c <= '8') {
            col += (c - '0');
        }
        else {
            if (rank < 0 || rank > 7 || col < 0 || col > 7) return false;
            Color color = std::isupper(c) ? Color::White : Color::Black;
            PieceType pt = PieceType::None;
            switch (std::tolower(c)) {
                case 'p': pt = PieceType::Pawn;   break;
                case 'n': pt = PieceType::Knight; break;
                case 'b': pt = PieceType::Bishop; break;
                case 'r': pt = PieceType::Rook;   break;
                case 'q': pt = PieceType::Queen;  break;
                case 'k': pt = PieceType::King;   break;
                default: return false;
            }
            squares[rank][col] = Piece{pt, color};
            col++;
        }
    }

    turn = (turnStr == "w") ? Color::White : Color::Black;

    castlingRights[0][0] = false;
    castlingRights[0][1] = false;
    castlingRights[1][0] = false;
    castlingRights[1][1] = false;
    if (castling != "-") {
        for (char c : castling) {
            switch (c) {
                case 'K': castlingRights[0][0] = true; break;
                case 'Q': castlingRights[0][1] = true; break;
                case 'k': castlingRights[1][0] = true; break;
                case 'q': castlingRights[1][1] = true; break;
            }
        }
    }

    if (ep != "-" && ep.size() == 2) {
        enPassantTarget = { ep[1] - '1', ep[0] - 'a' };
    } else {
        enPassantTarget = { -1, -1 };
    }

    try {
        halfMoveClock = halfmoveStr.empty() ? 0 : std::stoi(halfmoveStr);
        fullMoveNumber = fullmoveStr.empty() ? 1 : std::stoi(fullmoveStr);
    } catch (...) {
        halfMoveClock = 0;
        fullMoveNumber = 1;
    }

    recomputeBitboards();
    return true;
}

bool Board::hasValidKings() const {
    bool whiteKing = false, blackKing = false;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (squares[r][c].type == PieceType::King) {
                if (squares[r][c].color == Color::White) whiteKing = true;
                else blackKing = true;
            }
        }
    }
    return whiteKing && blackKing;
}

// ============================================================
// Bitboard helpers
// ============================================================

static inline Bitboard sqBB(int rank, int col) {
    return 1ULL << (rank * 8 + col);
}

static int piecePhaseWeight(PieceType pt) {
    switch (pt) {
        case PieceType::Knight: return 1;
        case PieceType::Bishop: return 1;
        case PieceType::Rook:   return 2;
        case PieceType::Queen:  return 4;
        default: return 0;
    }
}

void Board::bbSet(int rank, int col, Piece p) {
    Bitboard bb = sqBB(rank, col);
    occupiedBB          |= bb;
    colorBB[(int)p.color] |= bb;
    pieceBBs[(int)p.type] |= bb;
    if (p.type == PieceType::King) {
        if (p.color == Color::White) whiteKingSq = {rank, col};
        else                         blackKingSq = {rank, col};
    }
    phase += piecePhaseWeight(p.type);
}

void Board::bbClear(int rank, int col) {
    Piece p = squares[rank][col];
    if (p.isNone()) return;
    Bitboard bb = sqBB(rank, col);
    occupiedBB              &= ~bb;
    colorBB[(int)p.color]   &= ~bb;
    pieceBBs[(int)p.type]   &= ~bb;
    phase -= piecePhaseWeight(p.type);
}

void Board::recomputeBitboards() {
    occupiedBB = 0;
    colorBB[0] = colorBB[1] = 0;
    for (int i = 0; i < 7; i++) pieceBBs[i] = 0;
    whiteKingSq = {0, 4};
    blackKingSq = {7, 4};
    phase = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            Piece p = squares[r][c];
            if (p.isNone()) continue;
#ifdef DUCK_CHESS
            if (p.isDuck()) continue;
#endif
            Bitboard bb = sqBB(r, c);
            occupiedBB             |= bb;
            colorBB[(int)p.color]  |= bb;
            pieceBBs[(int)p.type]  |= bb;
            if (p.type == PieceType::King) {
                if (p.color == Color::White) whiteKingSq = {r, c};
                else                         blackKingSq = {r, c};
            }
            phase += piecePhaseWeight(p.type);
        }
    }
}

// ============================================================
// makeMove / unmakeMove (reversible — minimal undo struct)
// ============================================================

void Board::makeMove(const Move& m, UndoInfo& undo) {
    // Save minimal state needed for unmake
    undo.enPassantTarget = enPassantTarget;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            undo.castlingRights[i][j] = castlingRights[i][j];
    undo.halfMoveClock  = halfMoveClock;
    undo.fullMoveNumber = fullMoveNumber;
    undo.hash           = hash;
    undo.duckSquare     = duckSquare;
    undo.occupiedBB     = occupiedBB;
    undo.colorBB[0]     = colorBB[0];
    undo.colorBB[1]     = colorBB[1];
    for (int i = 0; i < 7; i++) undo.pieceBBs[i] = pieceBBs[i];
    undo.whiteKingSq   = whiteKingSq;
    undo.blackKingSq   = blackKingSq;
    undo.phase         = phase;
    undo.movedPiece    = getPiece(m.from);
    undo.capturedPiece = getPiece(m.to);
    // En-passant capture square
    undo.capturedEP   = Piece{};
    undo.capturedEPSq = {-1, -1};
    if (undo.movedPiece.type == PieceType::Pawn &&
        m.to.col != m.from.col && undo.capturedPiece.isNone()) {
        undo.capturedEPSq = {m.from.rank, m.to.col};
        undo.capturedEP   = getPiece(undo.capturedEPSq);
    }

    // Full squares[][] snapshot for reliable unmake
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            undo.squares[r][c] = squares[r][c];

    applyMove(m);
}

void Board::unmakeMove(const Move& /*m*/, const UndoInfo& undo) {
    // Restore all state from snapshots
    enPassantTarget = undo.enPassantTarget;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            castlingRights[i][j] = undo.castlingRights[i][j];
    halfMoveClock  = undo.halfMoveClock;
    fullMoveNumber = undo.fullMoveNumber;
    hash           = undo.hash;
    duckSquare     = undo.duckSquare;
    occupiedBB     = undo.occupiedBB;
    colorBB[0]     = undo.colorBB[0];
    colorBB[1]     = undo.colorBB[1];
    for (int i = 0; i < 7; i++) pieceBBs[i] = undo.pieceBBs[i];
    whiteKingSq    = undo.whiteKingSq;
    blackKingSq    = undo.blackKingSq;
    phase          = undo.phase;
    turn           = (turn == Color::White) ? Color::Black : Color::White;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            squares[r][c] = undo.squares[r][c];
}

// ============================================================
// Automate Chess — piece costs and placement validation
// ============================================================

// Piece costs: Queen=7, Rook=4, Knight=3, Bishop=3, Pawn=1, King=free
int Board::automatePieceCost(PieceType pt) {
    switch (pt) {
        case PieceType::Queen:  return 7;
        case PieceType::Rook:   return 4;
        case PieceType::Knight: return 3;
        case PieceType::Bishop: return 3;
        case PieceType::Pawn:   return 1;
        case PieceType::King:   return 0;
        default:                return 0;
    }
}

// Check whether a side can legally place a piece on a given square.
// Rules:
//   - Pawns: ranks 2-3 (indices 1-2 for White, 5-6 for Black), max 2 per file
//   - Pieces (Q/R/N/B): ranks 1-2 (indices 0-1 for White, 6-7 for Black),
//     only after 6 mandatory pawns placed
//   - King: ranks 1-2 (same as pieces), only after 6 mandatory pawns placed,
//     must be placed last (after all other pieces), must not be in check
//   - Square must be empty
//   - Must have enough budget
bool Board::automateCanPlace(Color side, PieceType pt, Square sq) const {
    if (!isAutomateChess || automateSetupComplete) return false;
    if (automateSetupTurn != side) return false;
    if (!sq.isValid()) return false;
    if (!squares[sq.rank][sq.col].isNone()) return false; // square occupied

    int ci = (int)side;
    int cost = automatePieceCost(pt);
    if (cost > automateBudget[ci]) return false;

    // King already placed — can't place again
    if (pt == PieceType::King && automateKingPlaced[ci]) return false;

    // Must place 6 pawns before any non-pawn non-king piece
    bool minPawnsMet = (automatePawnsPlaced[ci] >= 6);

    // Rank constraints depend on side
    int backRank  = (side == Color::White) ? 0 : 7;
    int pawnRank1 = (side == Color::White) ? 1 : 6;
    int pawnRank2 = (side == Color::White) ? 2 : 5;

    if (pt == PieceType::Pawn) {
        // Pawns go on ranks 2-3 (pawnRank1 or pawnRank2)
        if (sq.rank != pawnRank1 && sq.rank != pawnRank2) return false;
        // Max 2 pawns per file
        int pawnsInFile = 0;
        for (int r = 0; r < 8; r++) {
            Piece p = squares[r][sq.col];
            if (p.type == PieceType::Pawn && p.color == side) pawnsInFile++;
        }
        if (pawnsInFile >= 2) return false;
    } else if (pt == PieceType::King) {
        // King: must have met pawn minimum, placed on back rank or pawn rank 1
        if (!minPawnsMet) return false;
        if (sq.rank != backRank && sq.rank != pawnRank1) return false;
        // King-in-check = instant loss (caller handles this)
    } else {
        // Q/R/N/B: must have met pawn minimum, placed on back rank or pawn rank 1
        if (!minPawnsMet) return false;
        if (sq.rank != backRank && sq.rank != pawnRank1) return false;
    }

    return true;
}

// Place a piece during Automate Chess setup phase.
// Assumes automateCanPlace() returned true.
void Board::automatePlacePiece(Color side, PieceType pt, Square sq) {
    int ci = (int)side;
    squares[sq.rank][sq.col] = { pt, side };
    automateBudget[ci] -= automatePieceCost(pt);
    if (pt == PieceType::Pawn) automatePawnsPlaced[ci]++;
    if (pt == PieceType::King) automateKingPlaced[ci] = true;

    // Advance setup turn to the other side
    automateSetupTurn = (side == Color::White) ? Color::Black : Color::White;

    // Check if setup is complete: both kings placed
    if (automateKingPlaced[0] && automateKingPlaced[1]) {
        automateSetupComplete = true;
        // Set up for normal play: White moves first
        turn = Color::White;
        // Disable castling (non-standard armies)
        castlingRights[0][0] = castlingRights[0][1] = false;
        castlingRights[1][0] = castlingRights[1][1] = false;
    }

    recomputeBitboards();
}
