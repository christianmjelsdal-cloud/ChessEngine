#pragma once
#include <SFML/Graphics.hpp>
#include "Board.h"
#include "MoveGen.h"
#include "Engine.h"
#include "NNUE.h"
#include "NNUETrainer.h"
#include "GameLogic.h"   // FIX 3.10: needed for isInsufficientMaterial()
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
    MoveList legalMoves;
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
    std::mutex        engineResultMutex_;   // H12: protects engineResult
    Engine*           activeEngine_ = nullptr;  // engine currently thinking
    std::vector<Move> cachedPV_;                 // PV snapshot kept during pending-move delay

    // Bot vs Bot mode
    bool              botVsBot = false;
    bool              botVsNNUE_ = false;      // Bot (classical) vs NNUE mode

    // Side selection: Default / Swapped / Random
    enum class SideConfig { Default, Swapped, Random };
    SideConfig sideConfig_ = SideConfig::Default;
    bool nnueOnWhite_ = false;  // resolved: true = NNUE plays White in BotVsNNUE
    bool              botPaused = false;
    bool              showArrows = true;
    bool              fastMode = false;        // no delay, no animation
    int               botDelayMs = 1500;       // delay between moves for readability
    int               botThinkMs = 1500;       // engine think time in bot mode
    sf::Clock         botDelayClock;
    std::vector<Move> currentPV;               // PV arrows to display (legacy, kept for non-live fallback)
    Move              botPendingMove_;          // move waiting to be applied after delay
    Piece             botPendingPiece_;         // piece for the pending move animation
    bool              botHasPendingMove_ = false;  // true when waiting to apply after delay
    int               lastEval_ = 0;            // last eval from White's perspective (for eval bar)
    int               lastEval2_ = 0;           // last eval from engine2_ (for dual eval bar)

#ifdef DUCK_CHESS
    // Duck Chess mode
    bool isDuckChess_ = false;        // is duck chess mode active?
    bool placingDuck_ = false;         // waiting for player to place duck
    Move pendingChessMove_;             // the chess move awaiting duck placement
    sf::Texture duckTexture_;           // duck piece texture
#else
    static constexpr bool isDuckChess_ = false;
    static constexpr bool placingDuck_ = false;
#endif

    // NNUE
    std::unique_ptr<NNUE::Network> nnueNet_;     // trained NNUE network
    bool              nnueEnabled_ = false;       // whether to use NNUE eval
    std::atomic<bool> nnueTraining_{false};        // training in progress
    std::atomic<bool> nnueEstimating_{false};     // ELO estimation in progress
    std::thread       nnueThread_;                // background thread for training/estimation
    std::string       nnueStatus_;                // training/ELO progress text
    mutable std::mutex nnueStatusMutex_;          // protects nnueStatus_
    NNUE::EloResult   lastEloResult_;             // last ELO estimation result
    std::atomic<bool> hasEloResult_{false};

    // Cancel flag for training/ELO
    std::atomic<bool> nnueCancelFlag_{false};

    // Text input mode for training config
    bool nnueInputMode_ = false;
    std::string nnueInputBuffer_;
    int nnueInputStep_ = 0;  // 0=games, 1=max positions, 2=epochs
    int nnueConfigGames_ = 50;
    int nnueConfigMaxPositions_ = 0;
    int nnueConfigEpochs_ = 50;
    std::atomic<int64_t> nnueETAEndMs_{0};  // 0 = no countdown, else steady_clock ms when task should finish

    // ELO estimation config input
    bool nnueEloInputMode_ = false;
    int nnueConfigEloGames_ = 100;

    // Layout
    static const int SQ = 80;
    static const int EVAL_BAR_W = 28;   // eval bar width
    static const int OX = 100;          // board offset (room for eval bar + coords)
    static const int OY = 40;

    // Setup
    void loadAssets();

    // Logic
    void handleMouseDown(int x, int y);
    void handleMouseMove(int x, int y);
    void handleMouseUp(int x, int y);
    void handlePromotionClick(int x, int y);
    void handleKeyPress(sf::Keyboard::Key key);
    void handleTextInput(uint32_t unicode);
    void processTrainingInput();
    void startTraining();
    void processEloInput();
    void startEloEstimation();
    void updateETA(std::chrono::steady_clock::time_point startTime, int done, int total);
    void selectPiece(Square sq);
    void executeMove(const Move& move, bool animate = true);
    void startAnimation(const Move& move, Piece piece);
    void updateAnimation();
    void finishMove(const Move& move);
    void startEngineThinking();
    void checkEngineResult();
    void updateStatus();
    void resetGame();

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
    void drawDualEvalBars();
    void drawSingleEvalBar(float barX, float barY, float barW, float barH,
                           int eval, const std::string& label, sf::Color labelColor);
    void resolveSides();  // apply sideConfig_ to engine assignments
    void drawHUD();

    // Helpers
    Square       screenToSquare(int x, int y);
    sf::Vector2f squareToScreen(Square sq);
    sf::Vector2f squareCenter(Square sq);
    bool         isLegalTarget(Square sq);
    bool         inputLocked() const;
};
