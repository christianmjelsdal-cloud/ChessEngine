#include "UCI.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

// ============================================================
//  UCI Protocol Implementation
//  Reference: https://backscattering.de/chess/uci/
// ============================================================

UCI::UCI() {
    Engine::initZobrist();
    // Load NNUE weights if available
    if (nnue_.loadWeights("assets/nnue_weights.bin")) {
        engine_.setNNUE(&nnue_);
    }

    // Set up info callback so Engine outputs UCI info lines during search
    engine_.onInfoCallback = [](int depth, int scoreCp, uint64_t nodes,
                                uint64_t nps, int64_t elapsedMs,
                                const std::string& pvStr) {
        std::cout << "info depth " << depth
                  << " score cp " << scoreCp
                  << " nodes " << nodes
                  << " nps " << nps
                  << " time " << elapsedMs;
        if (!pvStr.empty())
            std::cout << " pv " << pvStr;
        std::cout << std::endl;
    };
}

UCI::~UCI() {
    if (searchThread_.joinable()) {
        engine_.stop();
        searchThread_.join();
    }
}

// ------------------------------------------------------------
//  Main loop
// ------------------------------------------------------------
void UCI::loop() {
    std::string line;
    while (!quit_ && std::getline(std::cin, line)) {
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front())))
            line.erase(line.begin());
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
            line.pop_back();
        if (line.empty()) continue;

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
        else if (token == "quit")       cmdQuit();
    }
}

// ------------------------------------------------------------
//  uci
// ------------------------------------------------------------
void UCI::cmdUCI() {
    std::cout << "id name ChessEngine NNUE" << std::endl;
    std::cout << "id author Christian Mjelsdal" << std::endl;
    std::cout << "uciok" << std::endl;
}

// ------------------------------------------------------------
//  isready
// ------------------------------------------------------------
void UCI::cmdIsReady() {
    std::cout << "readyok" << std::endl;
}

// ------------------------------------------------------------
//  ucinewgame
// ------------------------------------------------------------
void UCI::cmdNewGame() {
    if (searchThread_.joinable()) {
        engine_.stop();
        searchThread_.join();
    }
    board_ = Board();
    board_.setStartingPosition();
    positionHistory_.clear();
}

// ------------------------------------------------------------
//  position startpos [moves ...]
//  position fen <fen> [moves ...]
// ------------------------------------------------------------
void UCI::cmdPosition(const std::string& line) {
    std::istringstream iss(line);
    std::string token;
    iss >> token; // "position"
    iss >> token; // "startpos" or "fen"

    positionHistory_.clear();

    if (token == "startpos") {
        board_ = Board();
        board_.setStartingPosition();
        iss >> token;
    }
    else if (token == "fen") {
        std::string fen;
        for (int i = 0; i < 6 && iss >> token; ++i) {
            if (i > 0) fen += " ";
            fen += token;
        }
        parseFEN(fen);
        iss >> token;
    }

    if (token == "moves") {
        while (iss >> token) {
            positionHistory_.push_back(Engine::computeHash(board_));
            Move m = parseMove(token);
            board_.applyMove(m);
        }
    }
}

// ------------------------------------------------------------
//  go — IMPROVED TIME MANAGEMENT
//  Now parses movestogo, computes soft/hard limits separately,
//  and uses a smarter allocation formula.
// ------------------------------------------------------------
void UCI::cmdGo(const std::string& line) {
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
    int movestogo = 0;   // NEW: 0 = sudden death (no movestogo sent)
    bool infinite = false;

    while (iss >> token) {
        if      (token == "depth")     iss >> depth;
        else if (token == "movetime")  iss >> moveTime;
        else if (token == "wtime")     iss >> wtime;
        else if (token == "btime")     iss >> btime;
        else if (token == "winc")      iss >> winc;
        else if (token == "binc")      iss >> binc;
        else if (token == "movestogo") iss >> movestogo;   // NEW
        else if (token == "infinite")  infinite = true;
    }

    // Calculate time for this move
    if (moveTime > 0) {
        // Exact move time — use most of it (leave 50ms safety margin)
        int safe = std::max(50, moveTime - 50);
        engine_.setTimeLimits(safe, safe);
    }
    else if (wtime > 0 || btime > 0) {
        int myTime = (board_.turn == Color::White) ? wtime : btime;
        int myInc  = (board_.turn == Color::White) ? winc  : binc;

        // ========================================
        //  IMPROVED TIME ALLOCATION
        // ========================================
        int softMs, hardMs;

        if (movestogo > 0) {
            // Tournament time control: X moves in Y time
            // Allocate time/moves + a portion of increment
            // Leave a buffer so we don't flag
            softMs = myTime / (movestogo + 2) + myInc * 3 / 4;
            hardMs = myTime / std::max(1, movestogo / 2 + 1) + myInc;
        } else {
            // Sudden death (or Fischer): estimate ~25 moves remaining
            // Use more time early, taper as clock shrinks
            int estimatedMoves = std::max(10, 25 - (int)positionHistory_.size() / 4);
            // But cap estimate — don't assume too many moves left
            estimatedMoves = std::min(estimatedMoves, 40);

            softMs = myTime / estimatedMoves + myInc * 3 / 4;
            hardMs = myTime / std::max(5, estimatedMoves / 3) + myInc;
        }

        // Safety: never use more than 50% of remaining time as hard limit
        hardMs = std::min(hardMs, myTime / 2);

        // Safety: soft limit should not exceed hard limit
        softMs = std::min(softMs, hardMs);

        // Floor: at least 50ms
        softMs = std::max(50, softMs);
        hardMs = std::max(50, hardMs);

        // Safety margin: always leave at least 100ms on the clock
        if (hardMs > myTime - 100 && myTime > 200)
            hardMs = myTime - 100;
        if (softMs > hardMs)
            softMs = hardMs;

        engine_.setTimeLimits(softMs, hardMs);
    }
    else if (infinite) {
        engine_.setTimeLimits(999999999, 999999999);
    }
    else {
        engine_.setTimeLimits(5000, 5000); // fallback
    }

    engine_.setPositionHistory(positionHistory_);

    searchThread_ = std::thread([this, depth]() {
        Board searchBoard = board_;
        Move best = engine_.getBestMove(searchBoard, depth);

        // The engine already outputs info lines via onInfoCallback during search.
        // Just output the bestmove.
        std::cout << "bestmove " << moveToUCI(best) << std::endl;
    });
}

// ------------------------------------------------------------
//  stop
// ------------------------------------------------------------
void UCI::cmdStop() {
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

// ============================================================
//  FEN Parser
// ============================================================
void UCI::parseFEN(const std::string& fen) {
    board_.clearBoard();

    std::istringstream iss(fen);
    std::string pieces, turn, castling, ep, halfmove, fullmove;
    iss >> pieces >> turn >> castling >> ep >> halfmove >> fullmove;

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
            Color color = std::isupper(c) ? Color::White : Color::Black;
            PieceType pt = PieceType::None;
            switch (std::tolower(c)) {
                case 'p': pt = PieceType::Pawn;   break;
                case 'n': pt = PieceType::Knight; break;
                case 'b': pt = PieceType::Bishop; break;
                case 'r': pt = PieceType::Rook;   break;
                case 'q': pt = PieceType::Queen;  break;
                case 'k': pt = PieceType::King;   break;
            }
            board_.squares[rank][col] = Piece{pt, color};
            col++;
        }
    }

    board_.turn = (turn == "w") ? Color::White : Color::Black;

    board_.castlingRights[0][0] = false;
    board_.castlingRights[0][1] = false;
    board_.castlingRights[1][0] = false;
    board_.castlingRights[1][1] = false;
    if (castling != "-") {
        for (char c : castling) {
            switch (c) {
                case 'K': board_.castlingRights[0][0] = true; break;
                case 'Q': board_.castlingRights[0][1] = true; break;
                case 'k': board_.castlingRights[1][0] = true; break;
                case 'q': board_.castlingRights[1][1] = true; break;
            }
        }
    }

    if (ep != "-" && ep.size() == 2) {
        board_.enPassantTarget = parseSquare(ep);
    }
    else {
        board_.enPassantTarget = {-1, -1};
    }

    if (!halfmove.empty()) {
        try {
            board_.halfMoveClock = std::stoi(halfmove);
        } catch (const std::exception&) {
            board_.halfMoveClock = 0;
        }
    }
    if (!fullmove.empty()) {
        try {
            board_.fullMoveNumber = std::stoi(fullmove);
        } catch (const std::exception&) {
            board_.fullMoveNumber = 1;
        }
    }
}

// ============================================================
//  Move parsing / formatting
// ============================================================
Square UCI::parseSquare(const std::string& s) {
    int col  = s[0] - 'a';
    int rank = s[1] - '1';
    return {rank, col};
}

std::string UCI::squareToUCI(const Square& sq) {
    std::string s;
    s += static_cast<char>('a' + sq.col);
    s += static_cast<char>('1' + sq.rank);
    return s;
}

Move UCI::parseMove(const std::string& uciMove) const {
    Move m;
    m.from = parseSquare(uciMove.substr(0, 2));
    m.to   = parseSquare(uciMove.substr(2, 2));

    if (uciMove.size() == 5) {
        switch (uciMove[4]) {
            case 'q': m.promotion = PieceType::Queen;  break;
            case 'r': m.promotion = PieceType::Rook;   break;
            case 'b': m.promotion = PieceType::Bishop; break;
            case 'n': m.promotion = PieceType::Knight; break;
        }
    }
    else {
        Piece p = board_.getPiece(m.from);
        if (p.type == PieceType::Pawn) {
            if ((p.color == Color::White && m.to.rank == 7) ||
                (p.color == Color::Black && m.to.rank == 0)) {
                m.promotion = PieceType::Queen;
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
