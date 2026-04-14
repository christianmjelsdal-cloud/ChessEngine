#pragma once
#include "Types.h"
#include "Bitboard.h"
#include <string>
#include <cstdint>

class Board {
public:
    // The 8x8 grid of pieces
    Piece squares[8][8];

    // Game state
    Color turn = Color::White;
    bool castlingRights[2][2] = { {true, true}, {true, true} };
    // castlingRights[Color][0] = kingside, [1] = queenside

    Square enPassantTarget = { -1, -1 }; // invalid by default

    int halfMoveClock = 0;  // for 50-move rule
    int fullMoveNumber = 1;

    // === Duck Chess ===
    bool isDuckChess = false;
    Square duckSquare = { -1, -1 }; // current duck position ({-1,-1} = not yet placed)

    // === Automate Chess ===
    // Automate Chess: each side has 35 points to spend on pieces.
    // Setup is turn-based (White places one piece, then Black, alternating).
    // After both kings are placed the game transitions to normal play.
    bool isAutomateChess = false;
    bool automateSetupComplete = false;  // true once both kings placed
    int  automateBudget[2] = {35, 35};   // remaining points per side [White=0, Black=1]
    int  automatePawnsPlaced[2] = {0, 0}; // pawns placed per side (need 6 before pieces)
    bool automateKingPlaced[2] = {false, false}; // king placed per side
    // Whose turn it is to place during setup (alternates White/Black)
    Color automateSetupTurn = Color::White;

    // ----------------------------------------------------------------
    // Bitboard redundancy (derived from squares[][], kept in sync)
    // ----------------------------------------------------------------
    Bitboard occupiedBB = 0;       // all occupied squares (used by Syzygy etc.)
    Bitboard colorBB[2]  = {0,0};  // [0]=White, [1]=Black
    Bitboard pieceBBs[7] = {};     // indexed by (int)PieceType (0=None,1=Pawn,...,6=King)

    // Cached king squares (avoid scanning 64 squares to find king)
    Square whiteKingSq = {0, 4};   // e1 by default
    Square blackKingSq = {7, 4};   // e8 by default

    // Zobrist hash (currently a simple field; set externally or via recomputeHash)
    uint64_t hash = 0;

    // Game phase (0=endgame, 24=opening): sum of piece weights
    int phase = 0;

    // ----------------------------------------------------------------
    // Undo information for makeMove / unmakeMove
    // ----------------------------------------------------------------
    struct UndoInfo {
        // Scalar state that changes every move
        Square   enPassantTarget;
        bool     castlingRights[2][2];
        int      halfMoveClock;
        int      fullMoveNumber;
        uint64_t hash;
        Square   duckSquare;
        // Piece state: what was on the from/to squares (and EP capture square)
        Piece    movedPiece;
        Piece    capturedPiece;
        Piece    capturedEP;
        Square   capturedEPSq;
        // Bitboard snapshot
        Bitboard occupiedBB;
        Bitboard colorBB[2];
        Bitboard pieceBBs[7];
        Square   whiteKingSq;
        Square   blackKingSq;
        int      phase;
        // Full squares[][] snapshot — guarantees correct unmake for all edge cases
        Piece    squares[8][8];
    };

    // Constructor
    Board();

    // Setup
    void setStartingPosition();
    void clearBoard();

    // Access
    Piece getPiece(Square sq) const;
    void  setPiece(Square sq, Piece piece);

    // Move (two flavors)
    void applyMove(const Move& move);          // permanent (no undo)
    void makeMove(const Move& m, UndoInfo& undo);  // reversible
    void unmakeMove(const Move& m, const UndoInfo& undo);

    // Duck chess: place (or move) the duck to a new square
    void placeDuck(Square sq);

    // Automate Chess: piece costs and placement validation
    static int automatePieceCost(PieceType pt);
    bool automateCanPlace(Color side, PieceType pt, Square sq) const;
    void automatePlacePiece(Color side, PieceType pt, Square sq);

    // Utility
    void printBoard() const;
    bool isSquareAttacked(Square sq, Color byColor) const;

    // Aliases used by older call sites
    bool isAttackedBy(Square sq, Color c) const { return isSquareAttacked(sq, c); }
    bool isAttackedBy(int sqIdx, Color c) const { return isSquareAttacked({sqIdx/8, sqIdx%8}, c); }
    bool setFromFEN(const std::string& fen)     { return fromFEN(fen); }

    // Check if a square is blocked by the duck
    bool isDuckSquare(Square sq) const;
    bool isDuckSquare(int rank, int col) const;

    // FEN support
    std::string toFEN() const;
    bool fromFEN(const std::string& fen);

    // Validation: checks that both kings are present
    bool hasValidKings() const;

    // ----------------------------------------------------------------
    // Bitboard query helpers (derived from pieceBBs / colorBB)
    // ----------------------------------------------------------------
    Bitboard occupied()                              const { return occupiedBB; }
    Bitboard pieces(Color c)                         const { return colorBB[(int)c]; }
    Bitboard pieces(PieceType pt)                    const { return pieceBBs[(int)pt]; }
    Bitboard pieces(Color c, PieceType pt)           const { return colorBB[(int)c] & pieceBBs[(int)pt]; }

    // ----------------------------------------------------------------
    // Bitboard maintenance
    // ----------------------------------------------------------------
    // Recompute all bitboard fields from the squares[][] array.
    // Call after any bulk change (FEN parsing, setup mode, etc.).
    void recomputeBitboards();

private:
    // Internal: update bitboards incrementally when placing/removing a piece.
    void bbSet(int rank, int col, Piece p);
    void bbClear(int rank, int col);
};
