#pragma once
#include "Board.h"
#include "Engine.h"
#include "MoveGen.h"
#include "NNUE.h"
#include "SelfPlayGen.h"
#include "Syzygy.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

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
    void cmdPonderHit();   // TIER-2 FIX #8: handle "ponderhit" command
    void cmdQuit();

    /// generate games N depth D workers W output path
    /// Runs self-play generation synchronously and writes a binary data file.
    void cmdGenerate(const std::string& line);

    /* ---------- helpers ---------- */
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
    // NOTE: Engine is ~3MB. UCI must be heap-allocated (e.g., via new/unique_ptr) to avoid stack overflow.
    Engine engine_;
    std::unique_ptr<NNUE::Network> nnue_;
    std::vector<uint64_t> positionHistory_;
    std::thread searchThread_;
    std::mutex coutMutex_;
    std::atomic<bool> quit_{false};  // FIX L-11: atomic for thread safety
    bool nnueLoaded_ = false;        // FIX M: track whether NNUE weights loaded

    // TIER-2 FIX #8: Pondering state
    std::atomic<bool> ponderEnabled_{true};    // FIX 13.7: atomic for thread safety
    std::atomic<bool> isPondering_{false};     // FIX 13.7: atomic for thread safety
    int ponderSoftMs_ = 0;        // Saved time limits for ponderhit
    int ponderHardMs_ = 0;
};
