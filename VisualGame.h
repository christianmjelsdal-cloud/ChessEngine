#pragma once
#include <SFML/Graphics.hpp>
#include "Board.h"
#include "MoveGen.h"
#include "Engine.h"
#include "NNUE.h"
#include "DuckNNUE.h"
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <optional>
#include <mutex>
#include <memory>

class VisualGame {
public:
    VisualGame();
    ~VisualGame();
    void run();

private:
    // Window & Assets
    sf::RenderWindow window;
    sf::Texture      pieceTextures[2][6]; // [color][pieceType]
    sf::Font         font;

    // Game State
    Board             board;
    std::vector<Move> legalMoves;
    std::vector<Move> selectedMoves;
    bool              pieceSelected = false;
    Square            selectedSq = { -1, -1 };
    Move              lastMove = { {-1,-1},{-1,-1} };
    bool              hasLastMove = false;
    bool              gameOver = false;
    std::string       statusMsg;
    int               moveNumber = 1;     // full move counter

    // Position history for repetition detection (Zobrist hashes)
    std::vector<uint64_t> positionHistory_;

    // Drag & Drop
    bool         isDragging = false;
    Square       dragFrom = { -1, -1 };
    sf::Vector2f dragPos;

    // Move Animation
    bool         isAnimating = false;
    sf::Vector2f animStartPos;
    sf::Vector2f animEndPos;
    Piece        animPiece;
    Move         animMove;
    float        animElapsed = 0.f;
    static constexpr float ANIM_DURATION = 0.15f;
    sf::Clock    animClock;

    // Promotion Dialog
    bool         isPromoting = false;
    Square       promoFrom = { -1, -1 };
    Square       promoTo   = { -1, -1 };
    Move         promoBaseMove;

    // Engine (threaded)
    Engine            engine_;
    Engine            engine2_;        // second engine for bot-vs-bot (black)
    bool              engineEnabled = true;
    Color             engineColor = Color::Black;
    int               engineTimeMs = 3000;
    bool              engineThinking = false;
    std::thread       engineThread;
    std::atomic<bool> engineDone{false};
    Move              engineResult;
    Engine*           activeEngine_ = nullptr;  // engine currently thinking
    std::vector<Move> cachedPV_;                 // PV snapshot kept during pending-move delay

    // Bot vs Bot mode
    bool              botVsBot = false;
    bool              botPaused = false;
    bool              showArrows = true;
    bool              fastMode = false;        // no delay, no animation
    int               botDelayMs = 1500;       // delay between moves for readability
    int               botThinkMs = 1500;       // engine think time in bot mode
    sf::Clock         botDelayClock;
    bool              botWaiting = false;       // waiting for delay before next move
    std::vector<Move> currentPV;               // PV arrows to display (legacy, kept for non-live fallback)
    std::mutex        pvMutex;
    Move              botPendingMove_;          // move waiting to be applied after delay
    Piece             botPendingPiece_;         // piece for the pending move animation
    bool              botHasPendingMove_ = false;  // true when waiting to apply after delay
    int               lastEval_ = 0;            // last eval from White's perspective (for eval bar)

    // Duck Chess mode
    bool isDuckChess_ = false;        // is duck chess mode active?
    bool placingDuck_ = false;         // waiting for player to place duck
    Move pendingChessMove_;             // the chess move awaiting duck placement
    sf::Texture duckTexture_;           // duck piece texture

    // Board Setup Mode
    bool setupMode_ = false;
    int setupPaletteIdx_ = 0;          // selected palette item
    Board setupSavedBoard_;             // saved board for cancel
    bool setupSavedDuckChess_ = false;
    std::string setupStatus_;

    // NNUE
    std::unique_ptr<NNUE::Network> nnueNet_;           // standard NNUE network
    std::unique_ptr<DuckNNUE::Network> duckNnueNet_;   // duck chess NNUE network
    bool              nnueEnabled_ = false;             // whether to use NNUE eval
    std::string       nnueStatus_;                      // NNUE status text

    // Layout
    static const int SQ = 80;
    static const int EVAL_BAR_W = 28;   // eval bar width
    static const int OX = 100;          // board offset (room for eval bar + coords)
    static const int OY = 40;
    static const int SETUP_PANEL_W = 140;  // extra width for setup palette
    static const int PALETTE_SQ = 48;       // palette cell size

    // Setup
    void loadAssets();

    // Logic
    void handleMouseDown(int x, int y);
    void handleMouseMove(int x, int y);
    void handleMouseUp(int x, int y);
    void handlePromotionClick(int x, int y);
    void handleKeyPress(sf::Keyboard::Key key);
    void selectPiece(Square sq);
    void executeMove(const Move& move, bool animate = true);
    void startAnimation(const Move& move, Piece piece);
    void updateAnimation();
    void finishMove(const Move& move);
    void startEngineThinking();
    void checkEngineResult();
    void updateStatus();
    void resetGame();
    void enterSetupMode();
    void exitSetupMode(bool apply);
    void handleSetupClick(int x, int y);

    // Rendering
    void render();
    void drawBoard();
    void drawHighlights();
    void drawPieces();
    void drawCoordinates();
    void drawStatus();
    void drawPromotionDialog();
    void drawDraggedPiece();
    void drawAnimatingPiece();
    void drawPVArrows();
    void drawArrow(sf::Vector2f from, sf::Vector2f to, sf::Color color);
    void drawEvalBar();
    void drawHUD();
    void drawSetupPanel();

    // Helpers
    Square       screenToSquare(int x, int y);
    sf::Vector2f squareToScreen(Square sq);
    sf::Vector2f squareCenter(Square sq);
    bool         isLegalTarget(Square sq);
    bool         inputLocked() const;
};
