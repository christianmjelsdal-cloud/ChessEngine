// ============================================================================
//  GameLogic.cpp — Pure game-rule implementations (no GUI, no SFML)
//
//  These functions were originally embedded in VisualGame::updateStatus().
//  Now they can be used by the GUI, UCI, self-play, and tests alike.
// ============================================================================

#include "GameLogic.h"
#include "Bitboard.h"  // FIX 3.8: needed for BB::popcount()
#include <algorithm>

namespace GameLogic {

// ── Individual checks ────────────────────────────────────────────────────────

bool isCheckmate(const Board& board, const MoveList& legalMoves) {
    return legalMoves.empty() && MoveGen::isInCheck(board, board.turn);
}

bool isStalemate(const Board& board, const MoveList& legalMoves) {
    return legalMoves.empty() && !MoveGen::isInCheck(board, board.turn);
}

bool isDraw50Move(const Board& board) {
    return board.halfMoveClock >= FIFTY_MOVE_CLOCK_LIMIT;
}

bool isThreefoldRepetition(const std::vector<uint64_t>& posHashes, int halfMoveClock) {
    if (posHashes.empty()) return false;
    uint64_t current = posHashes.back();
    // FIX 3.22: Only scan back to last irreversible move (halfMoveClock reset).
    // Repetitions cannot cross a pawn move or capture boundary.
    int start = std::max(0, (int)posHashes.size() - halfMoveClock - 1);
    int count = 0;
    for (int i = start; i < (int)posHashes.size(); i++) {
        if (posHashes[i] == current) {
            count++;
            if (count >= THREEFOLD_REP_COUNT) return true;
        }
    }
    return false;
}

bool isInsufficientMaterial(const Board& board) {
    // FIX 3.8: Use bitboards instead of scanning 64 squares via mailbox.
    // popcount on pieceBBs is O(1) per piece type vs O(64) mailbox scan.


    // If any pawns, rooks, or queens exist → sufficient material
    if (board.pieces(PieceType::Pawn) | board.pieces(PieceType::Rook) | board.pieces(PieceType::Queen))
        return false;

    int wN = BB::popcount(board.pieces(Color::White, PieceType::Knight));
    int bN = BB::popcount(board.pieces(Color::Black, PieceType::Knight));
    Bitboard wB = board.pieces(Color::White, PieceType::Bishop);
    Bitboard bB = board.pieces(Color::Black, PieceType::Bishop);
    int wBc = BB::popcount(wB);
    int bBc = BB::popcount(bB);

    int totalW = wN + wBc;
    int totalB = bN + bBc;

    // K vs K
    if (totalW == 0 && totalB == 0) return true;

    // K+B vs K  or  K vs K+B
    if (totalW == 0 && totalB == 1 && bBc == 1) return true;
    if (totalB == 0 && totalW == 1 && wBc == 1) return true;

    // K+N vs K  or  K vs K+N
    if (totalW == 0 && totalB == 1 && bN == 1) return true;
    if (totalB == 0 && totalW == 1 && wN == 1) return true;

    // K+B vs K+B (same-coloured bishops)
    if (totalW == 1 && wBc == 1 && totalB == 1 && bBc == 1) {
        // Light squares = 0x55AA55AA55AA55AA, dark = 0xAA55AA55AA55AA55
        constexpr Bitboard lightSquares = 0x55AA55AA55AA55AA;
        bool wOnLight = (wB & lightSquares) != 0;
        bool bOnLight = (bB & lightSquares) != 0;
        if (wOnLight == bOnLight) return true;
    }

    return false;
}

// ── Main classification ──────────────────────────────────────────────────────

GameResult classify(const Board& board,
                    const MoveList& legalMoves,
                    const std::vector<uint64_t>& posHashes,
                    bool isDuckChess) {
#ifdef DUCK_CHESS
    if (isDuckChess) {
        // Duck chess: king capture check
        if (MoveGen::isKingCaptured(board, Color::White))
            return GameResult::DuckKingCapturedBlackWins;
        if (MoveGen::isKingCaptured(board, Color::Black))
            return GameResult::DuckKingCapturedWhiteWins;
    }
#endif

    // Threefold repetition (check before move legality)
    if (isThreefoldRepetition(posHashes, board.halfMoveClock))
        return GameResult::DrawByRepetition;

    // 50-move rule
    if (isDraw50Move(board))
        return GameResult::DrawBy50Move;

    // Insufficient material
    if (!isDuckChess && isInsufficientMaterial(board))
        return GameResult::DrawByInsufficientMaterial;

    // No legal moves
    if (legalMoves.empty()) {
#ifdef DUCK_CHESS
        if (isDuckChess) {
            // In duck chess, no legal moves = loss (no stalemate)
            return (board.turn == Color::White)
                ? GameResult::DuckNoMovesBlackWins
                : GameResult::DuckNoMovesWhiteWins;
        }
#endif
        if (MoveGen::isInCheck(board, board.turn)) {
            return (board.turn == Color::White)
                ? GameResult::CheckmateBlackWins
                : GameResult::CheckmateWhiteWins;
        }
        return GameResult::Stalemate;
    }

    return GameResult::Ongoing;
}

// ── Status string ────────────────────────────────────────────────────────────

std::string statusString(const Board& board,
                         const MoveList& legalMoves,
                         GameResult result) {
    std::string side = (board.turn == Color::White) ? "White" : "Black";

    switch (result) {
        case GameResult::CheckmateWhiteWins:  return "Checkmate! White wins!";
        case GameResult::CheckmateBlackWins:  return "Checkmate! Black wins!";
        case GameResult::Stalemate:           return "Stalemate -- Draw!";
        case GameResult::DrawByRepetition:    return "Draw by threefold repetition!";
        case GameResult::DrawBy50Move:        return "Draw by 50-move rule!";
        case GameResult::DrawByInsufficientMaterial:
            return "Draw by insufficient material!";
#ifdef DUCK_CHESS
        case GameResult::DuckNoMovesWhiteWins:
            return "White wins! Black has no legal moves!";
        case GameResult::DuckNoMovesBlackWins:
            return "Black wins! White has no legal moves!";
        case GameResult::DuckKingCapturedWhiteWins:
            return "White wins! Black's king was captured!";
        case GameResult::DuckKingCapturedBlackWins:
            return "Black wins! White's king was captured!";
#endif
        case GameResult::Ongoing:
        default: {
            std::string msg = side + "'s turn";
            if (MoveGen::isInCheck(board, board.turn))
                msg += "  --  CHECK!";
            return msg;
        }
    }
}

}  // namespace GameLogic
