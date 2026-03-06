#pragma once
#include "Board.h"
#include "MoveGen.h"
#include <vector>
#include <string>
#include "Engine.h"

class Game {
public:
    Board board;
    bool  engineEnabled = true;   // 👈 add this
    Color engineColor = Color::Black; // 👈 engine plays Black
    int   engineDepth = 4;      // 👈 search depth (3-5 recommended)

    void run();           // Main game loop

private:
    void printStatus();
    Move getPlayerMove(const std::vector<Move>& legalMoves);

    // Converts "e2e4" style input to a Move
    Move parseMove(const std::string& input);

    // Converts square like "e2" to Square struct
    Square parseSquare(const std::string& s);

    // Converts Square to string like "e2"
    std::string squareToString(Square sq);

    bool isCheckmate(const std::vector<Move>& legalMoves);
    bool isStalemate(const std::vector<Move>& legalMoves);
};