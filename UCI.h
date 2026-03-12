#pragma once
#include "Board.h"
#include "Engine.h"
#include "MoveGen.h"
#include "NNUE.h"
#include <string>
#include <vector>
#include <thread>

class UCI {
public:
    UCI();
    ~UCI();

    /// Main loop: reads from stdin, writes to stdout. Blocks until "quit".
    void loop();

private:
    /* ---------- command handlers ---------- */
    void cmdUCI();
    void cmdIsReady();
    void cmdNewGame();
    void cmdPosition(const std::string& line);
    void cmdGo(const std::string& line);
    void cmdStop();
    void cmdQuit();

    /* ---------- helpers ---------- */
    /// Parse a FEN string into the board.
    void parseFEN(const std::string& fen);

    /// Parse a UCI move string (e.g. "e2e4", "e7e8q") and apply it.
    Move parseMove(const std::string& uciMove) const;

    /// Convert a move to UCI string.
    static std::string moveToUCI(const Move& m);

    /// Convert a square to UCI string (e.g. "e2").
    static std::string squareToUCI(const Square& sq);

    /// Parse a UCI square string (e.g. "e2") to Square.
    static Square parseSquare(const std::string& s);

    /* ---------- state ---------- */
    Board board_;
    Engine engine_;
    NNUE::Network nnue_;
    std::vector<uint64_t> positionHistory_;
    std::thread searchThread_;
    bool quit_ = false;
};
