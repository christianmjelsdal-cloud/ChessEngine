#include "MoveGen.h"

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
// PAWN
// -------------------------------------------------------
void MoveGen::generatePawnMoves(const Board& board, Square sq, std::vector<Move>& moves) {
    Piece p = board.getPiece(sq);
    int dir = (p.color == Color::White) ? 1 : -1;
    int startRank = (p.color == Color::White) ? 1 : 6;
    int promoRank = (p.color == Color::White) ? 7 : 0;

    // One square forward
    int nr = sq.rank + dir;
    if (inBounds(nr, sq.col) && board.squares[nr][sq.col].isNone() && !isDuck(board, nr, sq.col)) {
        if (nr == promoRank) {
            for (auto pt : { PieceType::Queen, PieceType::Rook,
                             PieceType::Bishop, PieceType::Knight })
                moves.push_back({ sq, {nr, sq.col}, pt });
        }
        else {
            moves.push_back({ sq, {nr, sq.col} });

            // Two squares forward from starting rank
            int nr2 = nr + dir;
            if (sq.rank == startRank && board.squares[nr2][sq.col].isNone() && !isDuck(board, nr2, sq.col))
                moves.push_back({ sq, {nr2, sq.col} });
        }
    }

    // Captures
    for (int dc : {-1, 1}) {
        int nc = sq.col + dc;
        if (!inBounds(nr, nc)) continue;

        // Can't capture the duck
        if (isDuck(board, nr, nc)) continue;

        Piece target = board.squares[nr][nc];
        bool isEnPassant = (board.enPassantTarget.rank == nr &&
            board.enPassantTarget.col == nc);

        if ((!target.isNone() && target.color != p.color && !target.isDuck()) || isEnPassant) {
            if (nr == promoRank) {
                for (auto pt : { PieceType::Queen, PieceType::Rook,
                                 PieceType::Bishop, PieceType::Knight })
                    moves.push_back({ sq, {nr, nc}, pt });
            }
            else {
                moves.push_back({ sq, {nr, nc} });
            }
        }
    }
}

// -------------------------------------------------------
// KNIGHT
// -------------------------------------------------------
void MoveGen::generateKnightMoves(const Board& board, Square sq, std::vector<Move>& moves) {
    Piece p = board.getPiece(sq);
    int offsets[8][2] = { {2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2} };
    for (auto& o : offsets) {
        int nr = sq.rank + o[0], nc = sq.col + o[1];
        if (!inBounds(nr, nc)) continue;
        if (isDuck(board, nr, nc)) continue; // can't land on duck
        Piece target = board.squares[nr][nc];
        if (target.isNone() || (target.color != p.color && !target.isDuck()))
            moves.push_back({ sq, {nr, nc} });
    }
}

// -------------------------------------------------------
// SLIDING PIECES (Bishop, Rook, Queen)
// -------------------------------------------------------
static void generateSlidingMoves(const Board& board, Square sq,
    const int dirs[][2], int numDirs,
    std::vector<Move>& moves) {
    Piece p = board.getPiece(sq);
    for (int i = 0; i < numDirs; i++) {
        int nr = sq.rank + dirs[i][0];
        int nc = sq.col + dirs[i][1];
        while (inBounds(nr, nc)) {
            // Duck blocks sliding pieces completely
            if (isDuck(board, nr, nc)) break;

            Piece target = board.squares[nr][nc];
            if (target.isNone()) {
                moves.push_back({ sq, {nr, nc} });
            }
            else {
                if (target.color != p.color && !target.isDuck())
                    moves.push_back({ sq, {nr, nc} }); // capture
                break; // blocked
            }
            nr += dirs[i][0];
            nc += dirs[i][1];
        }
    }
}

void MoveGen::generateBishopMoves(const Board& board, Square sq, std::vector<Move>& moves) {
    const int dirs[4][2] = { {1,1},{1,-1},{-1,1},{-1,-1} };
    generateSlidingMoves(board, sq, dirs, 4, moves);
}

void MoveGen::generateRookMoves(const Board& board, Square sq, std::vector<Move>& moves) {
    const int dirs[4][2] = { {1,0},{-1,0},{0,1},{0,-1} };
    generateSlidingMoves(board, sq, dirs, 4, moves);
}

void MoveGen::generateQueenMoves(const Board& board, Square sq, std::vector<Move>& moves) {
    generateBishopMoves(board, sq, moves);
    generateRookMoves(board, sq, moves);
}

// -------------------------------------------------------
// KING
// -------------------------------------------------------
void MoveGen::generateKingMoves(const Board& board, Square sq, std::vector<Move>& moves) {
    Piece p = board.getPiece(sq);
    for (int dr = -1; dr <= 1; dr++)
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;
            int nr = sq.rank + dr, nc = sq.col + dc;
            if (!inBounds(nr, nc)) continue;
            if (isDuck(board, nr, nc)) continue; // can't land on duck
            Piece target = board.squares[nr][nc];
            if (target.isNone() || (target.color != p.color && !target.isDuck()))
                moves.push_back({ sq, {nr, nc} });
        }

    // Castling
    int backRank = (p.color == Color::White) ? 0 : 7;
    int colorIdx = (p.color == Color::White) ? 0 : 1;
    Color opponent = (p.color == Color::White) ? Color::Black : Color::White;

    if (sq.rank == backRank && sq.col == 4) {
        // Kingside
        if (board.castlingRights[colorIdx][0] &&
            board.squares[backRank][5].isNone() && !isDuck(board, backRank, 5) &&
            board.squares[backRank][6].isNone() && !isDuck(board, backRank, 6)) {

            if (board.isDuckChess) {
                // In duck chess: no check concept, just need clear path (no duck blocking)
                moves.push_back({ sq, {backRank, 6} });
            } else {
                // Standard chess: king must not be in check, pass through check, or land in check
                if (!board.isSquareAttacked(sq, opponent) &&
                    !board.isSquareAttacked({backRank, 5}, opponent) &&
                    !board.isSquareAttacked({backRank, 6}, opponent)) {
                    moves.push_back({ sq, {backRank, 6} });
                }
            }
        }

        // Queenside
        if (board.castlingRights[colorIdx][1] &&
            board.squares[backRank][3].isNone() && !isDuck(board, backRank, 3) &&
            board.squares[backRank][2].isNone() && !isDuck(board, backRank, 2) &&
            board.squares[backRank][1].isNone() && !isDuck(board, backRank, 1)) {

            if (board.isDuckChess) {
                moves.push_back({ sq, {backRank, 2} });
            } else {
                if (!board.isSquareAttacked(sq, opponent) &&
                    !board.isSquareAttacked({backRank, 3}, opponent) &&
                    !board.isSquareAttacked({backRank, 2}, opponent)) {
                    moves.push_back({ sq, {backRank, 2} });
                }
            }
        }
    }
}

// -------------------------------------------------------
// PSEUDO-LEGAL (all moves, may leave king in check)
// -------------------------------------------------------
std::vector<Move> MoveGen::getPseudoLegalMoves(const Board& board) {
    std::vector<Move> moves;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            Piece p = board.squares[r][c];
            if (p.isNone() || p.isDuck() || p.color != board.turn) continue;
            Square sq = { r, c };
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
    return moves;
}

// -------------------------------------------------------
// CHECK DETECTION
// -------------------------------------------------------
bool MoveGen::isInCheck(const Board& board, Color color) {
    // In duck chess, there is no check concept
    if (board.isDuckChess) return false;

    // Find the king
    Square kingSq = { -1, -1 };
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            if (board.squares[r][c].type == PieceType::King &&
                board.squares[r][c].color == color)
                kingSq = { r, c };

    if (!kingSq.isValid()) return false;

    Color opponent = (color == Color::White) ? Color::Black : Color::White;
    return board.isSquareAttacked(kingSq, opponent);
}

// -------------------------------------------------------
// LEGAL MOVES (filters out moves that leave king in check)
// -------------------------------------------------------
std::vector<Move> MoveGen::getLegalMoves(const Board& board) {
    // In duck chess, use the duck chess move generator
    if (board.isDuckChess)
        return getDuckChessMoves(board);

    auto pseudoMoves = getPseudoLegalMoves(board);
    std::vector<Move> legal;

    for (auto& move : pseudoMoves) {
        Board temp = board;
        temp.applyMove(move);
        if (!isInCheck(temp, board.turn))
            legal.push_back(move);
    }

    return legal;
}

// -------------------------------------------------------
// DUCK CHESS: LEGAL CHESS MOVES (no check filtering)
// -------------------------------------------------------
std::vector<Move> MoveGen::getDuckChessMoves(const Board& board) {
    // In duck chess, all pseudo-legal moves are legal
    // (there is no check concept - you CAN leave your king in danger)
    return getPseudoLegalMoves(board);
}

// -------------------------------------------------------
// DUCK CHESS: VALID DUCK PLACEMENTS
// Returns all empty squares where the duck can be placed
// -------------------------------------------------------
std::vector<Square> MoveGen::getDuckPlacements(const Board& board) {
    std::vector<Square> placements;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            if (board.squares[r][c].isNone())
                placements.push_back({ r, c });
        }
    return placements;
}

// -------------------------------------------------------
// DUCK CHESS: CHECK IF A KING WAS CAPTURED
// -------------------------------------------------------
bool MoveGen::isKingCaptured(const Board& board, Color color) {
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            if (board.squares[r][c].type == PieceType::King &&
                board.squares[r][c].color == color)
                return false;
    return true; // king not found = captured
}
