#include "MoveGen.h"
#include "Bitboard.h"

// Helper: is a square on the board?
static bool inBounds(int r, int c) {
    return r >= 0 && r < 8 && c >= 0 && c < 8;
}

// Helper: is a square blocked by the duck? (can't land on or pass through)
static bool isDuck(const Board& board, int r, int c) {
    return board.isDuckSquare(r, c);
}

// Helper: can a piece land on this square? (not duck, and either empty or enemy)
static bool canLandOn(const Board& board, int r, int c, Color myColor) {
    if (isDuck(board, r, c)) return false;
    Piece target = board.squares[r][c];
    return target.isNone() || (target.color != myColor && !target.isDuck());
}

// -------------------------------------------------------
// PAWN (bitboard-based)
// -------------------------------------------------------
void MoveGen::generatePawnMoves(const Board& board, Square sq, MoveList& moves) {
    Piece p = board.getPiece(sq);
    const Color color    = p.color;
    const int   sqIdx    = BB::toSquareIndex(sq);
    const int   promoRank = (color == Color::White) ? 7 : 0;
    const int   startRank = (color == Color::White) ? 1 : 6;

    // Occupancy (empty squares = ~occupied, duck blocks pushes)
    Bitboard occ = board.occupied();
#ifdef DUCK_CHESS
    if (board.isDuckChess && board.duckSquare.isValid())
        occ |= BB::squareBB(board.duckSquare);
#endif
    Bitboard empty = ~occ;

    // ── Pushes ──────────────────────────────────────────────────────────────
    // Single push: shift one rank in pawn direction
    Bitboard single;
    if (color == Color::White)
        single = (BB::squareBB(sqIdx) << 8) & empty;
    else
        single = (BB::squareBB(sqIdx) >> 8) & empty;

    if (single) {
        int toSq = BB::lsb(single);
        Square to = BB::toSquare(toSq);
        if (to.rank == promoRank) {
            for (auto pt : { PieceType::Queen, PieceType::Rook,
                             PieceType::Bishop, PieceType::Knight })
                moves.push_back({ sq, to, pt });
        } else {
            moves.push_back({ sq, to });
            // Double push from starting rank
            if (sq.rank == startRank) {
                Bitboard dbl;
                if (color == Color::White)
                    dbl = (single << 8) & empty;
                else
                    dbl = (single >> 8) & empty;
                if (dbl)
                    moves.push_back({ sq, BB::toSquare(BB::lsb(dbl)) });
            }
        }
    }

    // ── Captures ────────────────────────────────────────────────────────────
    Bitboard enemies = board.pieces(color == Color::White ? Color::Black : Color::White);
#ifdef DUCK_CHESS
    if (board.isDuckChess && board.duckSquare.isValid())
        enemies &= ~BB::squareBB(board.duckSquare); // can't capture duck
#endif

    // En passant target
    Bitboard epBB = 0;
    if (board.enPassantTarget.isValid())
        epBB = BB::squareBB(board.enPassantTarget);

    Bitboard captureTargets = BB::pawnAttacks(color, sqIdx) & (enemies | epBB);
    while (captureTargets) {
        int toSq = BB::popLsb(captureTargets);
        Square to = BB::toSquare(toSq);
        if (to.rank == promoRank) {
            for (auto pt : { PieceType::Queen, PieceType::Rook,
                             PieceType::Bishop, PieceType::Knight })
                moves.push_back({ sq, to, pt });
        } else {
            moves.push_back({ sq, to });
        }
    }
}

// -------------------------------------------------------
// KNIGHT (bitboard-based)
// -------------------------------------------------------
void MoveGen::generateKnightMoves(const Board& board, Square sq, MoveList& moves) {
    Piece p = board.getPiece(sq);
    int sqIdx = BB::toSquareIndex(sq);
    Bitboard attacks = BB::KnightAttacks[sqIdx] & ~board.pieces(p.color);
#ifdef DUCK_CHESS
    if (board.isDuckChess && board.duckSquare.isValid())
        attacks &= ~BB::squareBB(board.duckSquare);
#endif
    while (attacks) {
        int to = BB::popLsb(attacks);
        moves.push_back({ sq, BB::toSquare(to) });
    }
}

// -------------------------------------------------------
// SLIDING PIECES — bitboard magic attacks (Bishop, Rook, Queen)
// -------------------------------------------------------

// Helper: get board occupancy including the duck (if present) for sliding piece blockers.
static Bitboard slidingOccupancy(const Board& board) {
    Bitboard occ = board.occupied();
#ifdef DUCK_CHESS
    if (board.isDuckChess && board.duckSquare.isValid())
        occ |= BB::squareBB(board.duckSquare);
#endif
    return occ;
}

void MoveGen::generateBishopMoves(const Board& board, Square sq, MoveList& moves) {
    Piece p = board.getPiece(sq);
    int sqIdx = BB::toSquareIndex(sq);
    Bitboard occ = slidingOccupancy(board);
    Bitboard attacks = BB::bishopAttacks(sqIdx, occ);
    attacks &= ~board.pieces(p.color); // can't capture own pieces
#ifdef DUCK_CHESS
    if (board.isDuckChess && board.duckSquare.isValid())
        attacks &= ~BB::squareBB(board.duckSquare); // can't land on duck
#endif
    while (attacks) {
        int to = BB::popLsb(attacks);
        moves.push_back({ sq, BB::toSquare(to) });
    }
}

void MoveGen::generateRookMoves(const Board& board, Square sq, MoveList& moves) {
    Piece p = board.getPiece(sq);
    int sqIdx = BB::toSquareIndex(sq);
    Bitboard occ = slidingOccupancy(board);
    Bitboard attacks = BB::rookAttacks(sqIdx, occ);
    attacks &= ~board.pieces(p.color); // can't capture own pieces
#ifdef DUCK_CHESS
    if (board.isDuckChess && board.duckSquare.isValid())
        attacks &= ~BB::squareBB(board.duckSquare); // can't land on duck
#endif
    while (attacks) {
        int to = BB::popLsb(attacks);
        moves.push_back({ sq, BB::toSquare(to) });
    }
}

void MoveGen::generateQueenMoves(const Board& board, Square sq, MoveList& moves) {
    Piece p = board.getPiece(sq);
    int sqIdx = BB::toSquareIndex(sq);
    Bitboard occ = slidingOccupancy(board);
    Bitboard attacks = BB::queenAttacks(sqIdx, occ);
    attacks &= ~board.pieces(p.color); // can't capture own pieces
#ifdef DUCK_CHESS
    if (board.isDuckChess && board.duckSquare.isValid())
        attacks &= ~BB::squareBB(board.duckSquare); // can't land on duck
#endif
    while (attacks) {
        int to = BB::popLsb(attacks);
        moves.push_back({ sq, BB::toSquare(to) });
    }
}

// -------------------------------------------------------
// KING (bitboard-based normal moves + castling with bitboard attack checks)
// -------------------------------------------------------
void MoveGen::generateKingMoves(const Board& board, Square sq, MoveList& moves) {
    Piece p = board.getPiece(sq);
    int sqIdx = BB::toSquareIndex(sq);

    // Normal king moves via bitboard lookup
    Bitboard attacks = BB::KingAttacks[sqIdx] & ~board.pieces(p.color);
#ifdef DUCK_CHESS
    if (board.isDuckChess && board.duckSquare.isValid())
        attacks &= ~BB::squareBB(board.duckSquare);
#endif
    while (attacks) {
        int to = BB::popLsb(attacks);
        moves.push_back({ sq, BB::toSquare(to) });
    }

    // Castling
    int backRank = (p.color == Color::White) ? 0 : 7;
    int colorIdx = (p.color == Color::White) ? 0 : 1;
    Color opponent = (p.color == Color::White) ? Color::Black : Color::White;

    if (sq.rank == backRank && sq.col == 4) {
        // Kingside
        // AUDIT FIX CS-4: Verify rook is on starting square before generating castling
        if (board.castlingRights[colorIdx][0] &&
            board.squares[backRank][7].type == PieceType::Rook &&
            board.squares[backRank][7].color == p.color &&
            board.squares[backRank][5].isNone() && !isDuck(board, backRank, 5) &&
            board.squares[backRank][6].isNone() && !isDuck(board, backRank, 6)) {

            if (board.isDuckChess) {
                // In duck chess: no check concept, just need clear path (no duck blocking)
                moves.push_back({ sq, {backRank, 6} });
            } else {
                // Standard chess: use bitboard attack detection for castling legality
                if (!board.isAttackedBy(sqIdx, opponent) &&
                    !board.isAttackedBy(BB::toSquareIndex(backRank, 5), opponent) &&
                    !board.isAttackedBy(BB::toSquareIndex(backRank, 6), opponent)) {
                    moves.push_back({ sq, {backRank, 6} });
                }
            }
        }

        // Queenside
        // AUDIT FIX CS-4: Verify rook is on starting square before generating castling
        if (board.castlingRights[colorIdx][1] &&
            board.squares[backRank][0].type == PieceType::Rook &&
            board.squares[backRank][0].color == p.color &&
            board.squares[backRank][3].isNone() && !isDuck(board, backRank, 3) &&
            board.squares[backRank][2].isNone() && !isDuck(board, backRank, 2) &&
            board.squares[backRank][1].isNone() && !isDuck(board, backRank, 1) /* INFO [8.11]: isDuck check is redundant here — isNone() already rejects duck squares */) {

            if (board.isDuckChess) {
                moves.push_back({ sq, {backRank, 2} });
            } else {
                if (!board.isAttackedBy(sqIdx, opponent) &&
                    !board.isAttackedBy(BB::toSquareIndex(backRank, 3), opponent) &&
                    !board.isAttackedBy(BB::toSquareIndex(backRank, 2), opponent)) {
                    moves.push_back({ sq, {backRank, 2} });
                }
            }
        }
    }
}

// -------------------------------------------------------
// PSEUDO-LEGAL (all moves, may leave king in check)
// -------------------------------------------------------
// §1.2: MoveList-based implementation (primary - no heap allocation)
// Iterates over own pieces via bitboard instead of scanning all 64 squares.
void MoveGen::getPseudoLegalMoves(const Board& board, MoveList& moves) {
    Bitboard ourPieces = board.pieces(board.turn);
    while (ourPieces) {
        int sqIdx = BB::popLsb(ourPieces);
        Square sq = BB::toSquare(sqIdx);
        Piece p = board.squares[sq.rank][sq.col];
        switch (p.type) {
        case PieceType::Pawn:   generatePawnMoves(board, sq, moves);   break;
        case PieceType::Knight: generateKnightMoves(board, sq, moves); break;
        case PieceType::Bishop: generateBishopMoves(board, sq, moves); break;
        case PieceType::Rook:   generateRookMoves(board, sq, moves);   break;
        case PieceType::Queen:  generateQueenMoves(board, sq, moves);  break;
        case PieceType::King:   generateKingMoves(board, sq, moves);   break;
        default: break;
        }
    }
}

// (vector-returning getPseudoLegalMoves removed — use MoveList overload)

// -------------------------------------------------------
// CHECK DETECTION (bitboard-based — O(1) via magic bitboard attack lookups)
// -------------------------------------------------------
bool MoveGen::isInCheck(const Board& board, Color color) {
    // In duck chess, there is no check concept
    if (board.isDuckChess) return false;

    // Use bitboard to find king square — O(1) via lsb
    Bitboard kingBB = board.pieces(color, PieceType::King);
    if (!kingBB) return false;
    int kingSq = BB::lsb(kingBB);

    Color opponent = (color == Color::White) ? Color::Black : Color::White;
    return board.isAttackedBy(kingSq, opponent);
}

// -------------------------------------------------------
// §1.2: MoveList-based LEGAL MOVES (no heap allocation)
// -------------------------------------------------------
// AUDIT FIX C-4: Accept Board& (not const) — we use makeMove/unmakeMove internally
void MoveGen::getLegalMoves(Board& board, MoveList& out) {
#ifdef DUCK_CHESS
    if (board.isDuckChess) {
        getDuckChessMoves(board, out);
        return;
    }
#endif
    MoveList pseudo;
    getPseudoLegalMoves(board, pseudo);
    for (int i = 0; i < pseudo.count; i++) {
        Color sideToMove = board.turn;
        Board::UndoInfo undo;
        board.makeMove(pseudo[i], undo);
        bool legal = !isInCheck(board, sideToMove);
        board.unmakeMove(pseudo[i], undo);
        if (legal) out.add(pseudo[i]);
    }
}

// -------------------------------------------------------
// §1.2: MoveList-based LEGAL CAPTURES (no heap allocation)
// -------------------------------------------------------
// AUDIT FIX C-4: Accept Board& (not const) — we use makeMove/unmakeMove internally
void MoveGen::getLegalCaptures(Board& board, MoveList& out) {
    MoveList pseudo;
    getPseudoLegalMoves(board, pseudo);
    for (int i = 0; i < pseudo.count; i++) {
        const Move& m = pseudo[i];
        bool isCapture = false;
        bool isPromotion = (m.promotion != PieceType::None);

        Piece target = board.squares[m.to.rank][m.to.col];
        if (!target.isNone() && target.color != board.turn && !target.isDuck())
            isCapture = true;

        Piece mover = board.squares[m.from.rank][m.from.col];
        if (mover.type == PieceType::Pawn &&
            m.from.col != m.to.col && target.isNone())
            isCapture = true;

        if (!isCapture && !isPromotion) continue;

        Color sideToMove = board.turn;
        Board::UndoInfo undo;
        board.makeMove(m, undo);
#ifdef DUCK_CHESS
        if (board.isDuckChess) {
            board.unmakeMove(m, undo);
            out.add(m);
        } else
#endif
        {
            bool legal = !MoveGen::isInCheck(board, sideToMove);
            board.unmakeMove(m, undo);
            if (legal) out.add(m);
        }
    }
}

// (vector-returning getLegalMoves removed — use MoveList overload)

// (vector-returning getLegalCaptures removed — use MoveList overload)

// -------------------------------------------------------
// DUCK CHESS: LEGAL CHESS MOVES (no check filtering)
// -------------------------------------------------------
#ifdef DUCK_CHESS
void MoveGen::getDuckChessMoves(const Board& board, MoveList& out) {
    // In duck chess, all pseudo-legal moves are legal
    // (there is no check concept - you CAN leave your king in danger)
    getPseudoLegalMoves(board, out);
}


// -------------------------------------------------------
// DUCK CHESS: VALID DUCK PLACEMENTS
// Returns all empty squares where the duck can be placed
// -------------------------------------------------------
void MoveGen::getDuckPlacements(const Board& board, SquareList& out) {
    // All empty squares — use bitboard complement of occupied
    Bitboard empty = ~board.occupied();
#ifdef DUCK_CHESS
    // Also exclude the current duck square (it's already in occupied via squares[][])
    // but include it as a valid placement target (duck can stay put or move)
    if (board.duckSquare.isValid())
        empty |= BB::squareBB(board.duckSquare);
#endif
    while (empty) {
        int sq = BB::popLsb(empty);
        out.add(BB::toSquare(sq));
    }
}


// -------------------------------------------------------
// DUCK CHESS: CHECK IF A KING WAS CAPTURED
// -------------------------------------------------------
bool MoveGen::isKingCaptured(const Board& board, Color color) {
    // FIX 1.12: Use bitboard instead of scanning 64 squares
    return board.pieces(color, PieceType::King) == 0;
}
#endif

