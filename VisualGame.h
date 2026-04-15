#pragma once
#include <SFML/Graphics.hpp>
#include "Board.h"
#include "MoveGen.h"
#include "Engine.h"
#include "NNUETrainer.h"
#include <chrono>
#include <sstream>
#include "NNUE.h"
#include "DuckNNUE.h"
#include "AutomateNNUE.h"
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

    // ── Analysis / Game History ──────────────────────────────────────────────
    struct HistoryEntry {
        Board       board;        // board state BEFORE the move
        Move        move;         // move played from this position
        std::string moveAlg;      // algebraic notation ("Nf3", "O-O", "exd5")
        int         moveNumber;   // full move number at this position
        Color       sideToMove;   // who moved
        int         eval    = 0;  // engine eval at this position (white POV, cp)
        int         cpLoss  = -1; // centipawn loss vs best move (-1 = unknown)
    };
    std::vector<HistoryEntry> gameHistory_;  // one entry per move played
    int  viewIdx_      = -1;   // -1 = live (current game); ≥0 = viewing history
    bool analysisMode_ = false; // true when viewIdx_ >= 0

    // Returns the board to display (historical or live)
    const Board& viewBoard() const {
        if (analysisMode_ && viewIdx_ >= 0 && viewIdx_ < (int)gameHistory_.size())
            return gameHistory_[viewIdx_].board;
        return board;
    }

    void navigateHistory(int delta);  // +1 forward, -1 back
    void exitAnalysisMode();
    void drawMoveList();              // right-panel move list
    bool handleMoveListClick(int x, int y);  // returns true if click was in panel

    // Convert a move to Standard Algebraic Notation (e.g. "Nf3", "exd5", "O-O")
    static std::string moveToAlgebraic(const Board& board, const Move& move);

    // ── Analysis engine (runs on viewed position) ────────────────────────────
    Engine            analysisEngine_;          // dedicated engine for analysis
    std::thread       analysisThread_;
    std::atomic<bool> analysisDone_{false};
    std::atomic<bool> analysisStop_{false};
    int               analysisEval_  = 0;       // current eval (white POV)
    int               analysisDepth_ = 0;       // current depth
    uint64_t          analysisNodes_ = 0;       // nodes searched
    uint64_t          analysisNps_   = 0;       // nodes per second
    struct AnalysisPVLine {
        std::vector<Move> pv;
        int eval = 0;
        int depth = 0;
    };
    std::vector<AnalysisPVLine> analysisPVLines_;  // top N PV lines
    std::mutex                  analysisMtx_;
    int                         analysisViewIdx_ = -1;  // which position is being analysed

    void startAnalysisEngine();   // launch analysis on viewBoard()
    void stopAnalysisEngine();    // stop and join analysis thread
    void updateAnalysisStats();   // poll engine for latest stats

    // Opening name lookup
    struct OpeningEntry { std::string eco; std::string name; std::string moves; };
    std::vector<OpeningEntry> openingDb_;
    std::string               currentOpening_;  // name of current opening (empty if unknown)
    void loadOpeningDb();
    void updateOpeningName();

    // Debounce: delay analysis start until user stops scrolling
    sf::Clock  analysisDebounce_;
    bool       analysisPending_ = false;
    static constexpr int ANALYSIS_DEBOUNCE_MS = 250;

    // PV line expand state (which lines are expanded)
    bool pvLineExpanded_[3] = {true, false, false};

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

    // Automate Chess mode
    bool isAutomateChess_ = false;    // is Automate Chess mode active?
    // During setup: which piece type is selected in the palette (for human placement)
    PieceType automatePaletteType_ = PieceType::Pawn;
    // Bot uses NNUE eval to pick best placement from candidates
    std::unique_ptr<NNUE::Network> automatePlayNet_; // Automate Play NNUE (separate weights)

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

    // NNUE training / elo input state
    bool              nnueInputMode_      = false;
    bool              nnueEloInputMode_   = false;
    std::string       nnueInputBuffer_;
    int               nnueInputStep_      = 0;
    int               nnueConfigGames_         = 1000;
    int               nnueConfigMaxPositions_  = 5000000;
    int               nnueConfigEpochs_        = 10;
    int               nnueConfigEloGames_      = 200;
    std::mutex        nnueStatusMutex_;
    std::atomic<int64_t> nnueETAEndMs_{0};
    bool              nnueTraining_       = false;
    bool              nnueEstimating_     = false;
    std::atomic<bool> nnueCancelFlag_{false};
    std::thread       nnueThread_;

    // Bot vs NNUE mode
    bool botVsNNUE_  = false;
    bool nnueOnWhite_ = false;
    int  lastEval2_   = 0;

    // Side configuration for bot vs bot / nnue
    enum class SideConfig { Default, Normal, Swapped, Random };
    SideConfig sideConfig_ = SideConfig::Normal;

    // Elo estimation result
    NNUE::EloResult lastEloResult_;
    bool            hasEloResult_ = false;

    // Layout — base constants (used for window creation)
    static const int SQ = 80;
    static const int EVAL_BAR_W = 28;   // eval bar width
    static const int OX = 100;          // board offset (room for eval bar + coords)
    static const int OY = 62;           // increased: room for 2-line header
    static const int SETUP_PANEL_W = 140;  // extra width for setup palette
    static const int PALETTE_SQ = 48;       // palette cell size

    // Dynamic layout (updated each frame in render())
    int   dynSQ_  = SQ;   // effective square size
    int   dynOX_  = OX;   // effective board X origin
    float scale_  = 1.0f; // dynSQ_ / SQ — used to scale fonts, margins, bar widths

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

    // Automate Chess
    void enterAutomateSetup();
    void exitAutomateSetup(bool apply);
    void handleAutomateSetupClick(int x, int y);
    void drawAutomateSetupPanel();
    void drawAutomateSetupOverlay();
    bool automateSetupBotPlace();  // bot places one piece using heuristic eval

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

    // NNUE training / elo input handlers (implemented in VisualGame_Input.cpp)
    void handleTextInput(uint32_t unicode);
    void processTrainingInput();
    void processEloInput();
    void startTraining();
    void startEloEstimation();
    void updateETA(std::chrono::steady_clock::time_point startTime, int done, int total);

    void resolveSides();

    // Rendering helpers (implemented in VisualGame_Render.cpp)
    void drawSingleEvalBar(float x, float y, float w, float h, int eval,
                           const std::string& label = "", sf::Color labelColor = sf::Color::White);
    void drawDualEvalBars();
    void drawAnalysisOverlay();
};
