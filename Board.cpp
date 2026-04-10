#include "Board.h"
#include <iostream>
#include <sstream>

Board::Board() {
    setStartingPosition();
}

void Board::clearBoard() {
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            squares[r][c] = Piece{};
    duckSquare = { -1, -1 };
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
    squares[sq.rank][sq.col] = { PieceType::Duck, Color::White }; // color is irrelevant
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
                case PieceType::Duck:   symbol = '@'; break; // duck
                default: break;
                }
                if (p.color == Color::Black && p.type != PieceType::Duck)
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
        // Revoke castling rights
        int ci = (moving.color == Color::White) ? 0 : 1;
        castlingRights[ci][0] = castlingRights[ci][1] = false;
    }

    // Revoke castling rights if rook moves or is captured
    if (moving.type == PieceType::Rook) {
        int ci = (moving.color == Color::White) ? 0 : 1;
        if (move.from.col == 7) castlingRights[ci][0] = false;
        if (move.from.col == 0) castlingRights[ci][1] = false;
    }
    // Revoke opponent castling rights if their rook is captured
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

    // In duck chess, check if king was captured (game-ending)
    // (We don't prevent this — it's how you win in duck chess)

    // Move the piece
    setPiece(move.to, moving);
    setPiece(move.from, Piece{});

    // Promotion
    if (move.promotion != PieceType::None)
        squares[move.to.rank][move.to.col].type = move.promotion;

    // Handle duck placement if specified in the move (used by engine)
    if (isDuckChess && move.duckTo.isValid()) {
        placeDuck(move.duckTo);
    }

    // Flip turn
    turn = (turn == Color::White) ? Color::Black : Color::White;
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

    halfMoveClock = halfmoveStr.empty() ? 0 : std::stoi(halfmoveStr);
    fullMoveNumber = fullmoveStr.empty() ? 1 : std::stoi(fullmoveStr);

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
