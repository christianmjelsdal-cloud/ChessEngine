#include "VisualGame.h"
#include <iostream>
#include <cmath>
#include <sstream>
#include <random>
#include <chrono>

// -------------------------------------------------------
// CONSTRUCTOR / DESTRUCTOR
// -------------------------------------------------------
VisualGame::VisualGame()
    : window(sf::VideoMode({static_cast<unsigned>(OX * 2 + SQ * 8 + SETUP_PANEL_W),
                            static_cast<unsigned>(OY * 2 + SQ * 8 + 80)}), "Chess Engine")
{
    window.setFramerateLimit(60);
    loadAssets();
    { MoveList ml_; MoveGen::getLegalMoves(board, ml_); legalMoves.assign(ml_.begin(), ml_.end()); }

    // Record starting position hash
    Engine::initZobrist();
    positionHistory_.push_back(Engine::computeHash(board));

    updateStatus();
}

VisualGame::~VisualGame() {
    engine_.stop();
    engine2_.stop();
    if (engineThread.joinable())
        engineThread.join();
}

// -------------------------------------------------------
// LOAD ASSETS
// -------------------------------------------------------
void VisualGame::loadAssets() {
    if (!font.openFromFile("assets/font.ttf")) {
        std::cerr << "ERROR: Could not load assets/font.ttf\n";
    }

    const char* colorPrefix[] = { "w", "b" };
    const char* pieceChar[]   = { "P", "N", "B", "R", "Q", "K" };

    for (int c = 0; c < 2; c++) {
        for (int p = 0; p < 6; p++) {
            std::string path = std::string("assets/pieces/")
                             + colorPrefix[c] + pieceChar[p] + ".png";
            if (!pieceTextures[c][p].loadFromFile(path))
                std::cerr << "ERROR: Could not load " << path << "\n";
            pieceTextures[c][p].setSmooth(true);
        }
    }

    // Duck texture
    if (!duckTexture_.loadFromFile("assets/pieces/duck.png"))
        std::cerr << "ERROR: Could not load assets/pieces/duck.png\n";
    duckTexture_.setSmooth(true);
}

// -------------------------------------------------------
// INPUT LOCKED
// -------------------------------------------------------
bool VisualGame::inputLocked() const {
    return isAnimating || isPromoting || gameOver || setupMode_;
}

// -------------------------------------------------------
// RUN (main loop)
// -------------------------------------------------------
void VisualGame::run() {
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                engine_.stop();
                engine2_.stop();
                window.close();
            }
            else if (const auto* kp = event->getIf<sf::Event::KeyPressed>()) {
                handleKeyPress(kp->code);
            }
            else if (setupMode_) {
                if (const auto* mb = event->getIf<sf::Event::MouseButtonPressed>()) {
                    if (mb->button == sf::Mouse::Button::Left)
                        handleSetupClick(mb->position.x, mb->position.y);
                }
            }
            else if (isPromoting) {
                if (const auto* mb = event->getIf<sf::Event::MouseButtonPressed>()) {
                    if (mb->button == sf::Mouse::Button::Left)
                        handlePromotionClick(mb->position.x, mb->position.y);
                }
            }
            else if (!botVsBot && !inputLocked() && !engineThinking) {
                if (const auto* mb = event->getIf<sf::Event::MouseButtonPressed>()) {
                    if (mb->button == sf::Mouse::Button::Left)
                        handleMouseDown(mb->position.x, mb->position.y);
                }
                else if (const auto* mm = event->getIf<sf::Event::MouseMoved>()) {
                    handleMouseMove(mm->position.x, mm->position.y);
                }
                else if (const auto* mr = event->getIf<sf::Event::MouseButtonReleased>()) {
                    if (mr->button == sf::Mouse::Button::Left)
                        handleMouseUp(mr->position.x, mr->position.y);
                }
            }
        }

        // Update animation
        if (isAnimating)
            updateAnimation();

        // Check if engine finished thinking
        if (engineThinking && engineDone.load())
            checkEngineResult();

        // Bot vs Bot: after engine finishes + delay, apply the pending move
        if (botVsBot && botHasPendingMove_ && !botPaused && !gameOver) {
            float elapsed = static_cast<float>(botDelayClock.getElapsedTime().asMilliseconds());
            int delayNeeded = fastMode ? 0 : botDelayMs;
            if (elapsed >= delayNeeded) {
                botHasPendingMove_ = false;
                cachedPV_.clear();
                // Apply the pending move
                lastMove = botPendingMove_;
                hasLastMove = true;
                if (fastMode) {
                    finishMove(botPendingMove_);
                } else {
                    startAnimation(botPendingMove_, botPendingPiece_);
                }
            }
        }

        render();
    }
}

// -------------------------------------------------------
// RESET GAME
// -------------------------------------------------------
void VisualGame::resetGame() {
    // Stop any running engine
    engine_.stop();
    engine2_.stop();
    if (engineThread.joinable())
        engineThread.join();

    // Reset all game state
    board = Board(); // fresh starting position
    board.isDuckChess = isDuckChess_;
    // Automate Chess: re-enter setup phase on reset
    if (isAutomateChess_) {
        enterAutomateSetup();
        return;
    }
    { MoveList ml_; MoveGen::getLegalMoves(board, ml_); legalMoves.assign(ml_.begin(), ml_.end()); }
    selectedMoves.clear();
    pieceSelected = false;
    selectedSq = { -1, -1 };
    lastMove = { {-1,-1},{-1,-1} };
    hasLastMove = false;
    gameOver = false;
    moveNumber = 1;
    isDragging = false;
    isAnimating = false;
    isPromoting = false;
    engineThinking = false;
    engineDone.store(false);
    activeEngine_ = nullptr;
    botWaiting = false;
    botHasPendingMove_ = false;
    cachedPV_.clear();
    lastEval_ = 0;
    placingDuck_ = false;

    // Reset position history
    positionHistory_.clear();
    positionHistory_.push_back(Engine::computeHash(board));

    // If in bot-vs-bot mode, restart
    if (botVsBot && !botPaused) {
        startEngineThinking();
    }

    updateStatus();
}

// -------------------------------------------------------
// SELECT PIECE
// -------------------------------------------------------
void VisualGame::selectPiece(Square sq) {
    pieceSelected = true;
    selectedSq    = sq;
    selectedMoves.clear();
    for (auto& m : legalMoves)
        if (m.from.rank == sq.rank && m.from.col == sq.col)
            selectedMoves.push_back(m);
}

// -------------------------------------------------------
// EXECUTE MOVE
// -------------------------------------------------------
void VisualGame::executeMove(const Move& move, bool animate) {
    pieceSelected = false;
    selectedMoves.clear();

    // In fast mode during bot-vs-bot, skip animation
    if (fastMode && botVsBot)
        animate = false;

    if (animate) {
        Piece p = board.getPiece(move.from);
        startAnimation(move, p);
    }
    else {
        finishMove(move);
    }
}

// -------------------------------------------------------
// ANIMATION
// -------------------------------------------------------
void VisualGame::startAnimation(const Move& move, Piece piece) {
    isAnimating  = true;
    animMove     = move;
    animPiece    = piece;
    animStartPos = squareToScreen(move.from);
    animEndPos   = squareToScreen(move.to);
    animElapsed  = 0.f;
    animClock.restart();
}

void VisualGame::updateAnimation() {
    animElapsed = animClock.getElapsedTime().asSeconds();
    if (animElapsed >= ANIM_DURATION) {
        isAnimating = false;
        finishMove(animMove);
    }
}

// -------------------------------------------------------
// FINISH MOVE
// -------------------------------------------------------
void VisualGame::finishMove(const Move& move) {
    lastMove    = move;
    hasLastMove = true;

    // Track move number (increments after black moves)
    if (board.turn == Color::Black)
        moveNumber++;

    board.applyMove(move);
    board.recomputeBitboards(); // sync bitboards after applyMove

    // In duck chess mode, handle duck placement
    if (isDuckChess_) {
        // For bot-vs-bot or engine moves: the move has duckTo set by the engine
        if (botVsBot) {
            // Engine provides duckTo for both sides
            if (move.duckTo.isValid()) {
                board.placeDuck(move.duckTo);
            }
            // Record position hash after duck placement
            positionHistory_.push_back(Engine::computeHash(board));
            { MoveList ml_; MoveGen::getLegalMoves(board, ml_); legalMoves.assign(ml_.begin(), ml_.end()); }
            updateStatus();

            if (!gameOver) {
                startEngineThinking();
            }
            return;
        }
        else if (engineEnabled && board.turn == engineColor) {
            // Human just moved (turn has flipped to engine's color) — need to place the duck
            // Don't record hash or generate moves yet; wait for duck placement
            placingDuck_ = true;
            pendingChessMove_ = move;
            updateStatus();
            return;
        }
        else if (engineEnabled && board.turn != engineColor) {
            // Engine just moved (turn has flipped to human's color) — duckTo is set
            if (move.duckTo.isValid()) {
                board.placeDuck(move.duckTo);
            }
            positionHistory_.push_back(Engine::computeHash(board));
            { MoveList ml_; MoveGen::getLegalMoves(board, ml_); legalMoves.assign(ml_.begin(), ml_.end()); }
            updateStatus();

            if (!gameOver) {
                // Now it's the human's turn — don't start engine
            }
            return;
        }
        else {
            // No engine (player vs player) — human places duck
            placingDuck_ = true;
            pendingChessMove_ = move;
            updateStatus();
            return;
        }
    }

    // Normal (non-duck) chess flow
    // Record position hash for repetition detection
    positionHistory_.push_back(Engine::computeHash(board));

    { MoveList ml_; MoveGen::getLegalMoves(board, ml_); legalMoves.assign(ml_.begin(), ml_.end()); }
    updateStatus();

    if (!gameOver) {
        if (botVsBot) {
            // In bot vs bot, start thinking immediately (delay comes after)
            startEngineThinking();
        }
        else if (engineEnabled && board.turn == engineColor) {
            startEngineThinking();
        }
    }
}

// -------------------------------------------------------
// ENGINE
// -------------------------------------------------------
void VisualGame::startEngineThinking() {
    if (gameOver || botPaused) return;

    engineThinking = true;
    engineDone.store(false);

    if (engineThread.joinable())
        engineThread.join();

    Board boardCopy = board;
    // Use engine2_ for black in bot-vs-bot so each side has its own TT/history
    Engine* eng = (botVsBot && board.turn == Color::Black) ? &engine2_ : &engine_;
    eng->setTimeLimit(botVsBot ? botThinkMs : engineTimeMs);

    // Pass position history for repetition detection
    eng->setPositionHistory(positionHistory_);

    // Track which engine is active (for live PV arrows)
    activeEngine_ = eng;

    engineThread = std::thread([this, boardCopy, eng]() mutable {
        Move best = eng->getBestMove(boardCopy);
        engineResult = best;
        engineDone.store(true);
    });

    updateStatus();
}

void VisualGame::checkEngineResult() {
    if (engineThread.joinable())
        engineThread.join();

    engineThinking = false;

    // If game was reset while engine was thinking, discard result
    if (gameOver && !botVsBot) {
        activeEngine_ = nullptr;
        return;
    }

    Move move = engineResult;

    // Validate the move is still legal (in case of reset)
    bool isLegal = false;
    for (const auto& m : legalMoves) {
        if (m == move) { isLegal = true; break; }
    }
    if (!isLegal) {
        activeEngine_ = nullptr;
        return;
    }

    Piece p = board.getPiece(move.from);

    // Save last eval for eval bar (engine already stores from White's perspective)
    if (activeEngine_) {
        lastEval_ = activeEngine_->getLiveEval();
    }

    if (botVsBot && !fastMode) {
        // Think-then-pause: engine is done, now wait for delay before applying
        botPendingMove_ = move;
        botPendingPiece_ = p;
        botHasPendingMove_ = true;
        botDelayClock.restart();
        // Cache the final PV for arrow display during the delay
        if (activeEngine_) cachedPV_ = activeEngine_->getLivePV();
        activeEngine_ = nullptr;
    } else {
        // Player vs engine, or fast mode: apply immediately
        lastMove    = move;
        hasLastMove = true;
        activeEngine_ = nullptr;

        if (fastMode && botVsBot) {
            finishMove(move);
        } else {
            startAnimation(move, p);
        }
    }
}

// -------------------------------------------------------
// UPDATE STATUS
// -------------------------------------------------------
void VisualGame::updateStatus() {
    std::string side = (board.turn == Color::White) ? "White" : "Black";

    // Automate Chess setup phase
    if (isAutomateChess_ && !board.automateSetupComplete) {
        std::string setupSide = (board.automateSetupTurn == Color::White) ? "White" : "Black";
        int ci = (int)board.automateSetupTurn;
        statusMsg = "Automate Setup — " + setupSide + " to place | Budget: " +
                    std::to_string(board.automateBudget[ci]) + "pt | Pawns: " +
                    std::to_string(board.automatePawnsPlaced[ci]) + "/6";
        return;
    }

    // Duck chess: check if a king was captured
    if (isDuckChess_) {
        if (MoveGen::isKingCaptured(board, Color::White)) {
            gameOver = true;
            statusMsg = "Black wins! White's king was captured!";
            return;
        }
        if (MoveGen::isKingCaptured(board, Color::Black)) {
            gameOver = true;
            statusMsg = "White wins! Black's king was captured!";
            return;
        }
    }

    // Check for threefold repetition
    if (!positionHistory_.empty()) {
        uint64_t currentHash = positionHistory_.back();
        int repCount = 0;
        for (const auto& h : positionHistory_) {
            if (h == currentHash) repCount++;
        }
        if (repCount >= 3) {
            gameOver = true;
            statusMsg = "Draw by threefold repetition!";
            return;
        }
    }

    // Check for 50-move rule
    if (board.halfMoveClock >= 100) {
        gameOver = true;
        statusMsg = "Draw by 50-move rule!";
        return;
    }

    if (legalMoves.empty()) {
        gameOver = true;
        if (isDuckChess_) {
            // In duck chess, no legal moves = the player loses (not stalemate)
            std::string winner = (board.turn == Color::White) ? "Black" : "White";
            statusMsg = winner + " wins! " + side + " has no legal moves!";
        } else {
            if (MoveGen::isInCheck(board, board.turn)) {
                std::string winner = (board.turn == Color::White) ? "Black" : "White";
                statusMsg = "Checkmate! " + winner + " wins!";
            }
            else {
                statusMsg = "Stalemate -- Draw!";
            }
        }
    }
    else if (placingDuck_) {
        statusMsg = side + "'s turn  --  Place the duck!";
    }
    else if (engineThinking) {
        statusMsg = side + " is thinking...";
    }
    else if (botVsBot && botPaused) {
        statusMsg = "PAUSED  |  " + side + "'s turn";
    }
    else {
        statusMsg = side + "'s turn";
        if (!isDuckChess_ && MoveGen::isInCheck(board, board.turn))
            statusMsg += "  --  CHECK!";
    }
}

// -------------------------------------------------------
// HELPER: screen <-> board coordinate conversions
// -------------------------------------------------------
Square VisualGame::screenToSquare(int x, int y) {
    int col = (x - OX) / SQ;
    int row = 7 - (y - OY) / SQ;
    if (col < 0 || col > 7 || row < 0 || row > 7)
        return {-1, -1};
    return {row, col};
}

sf::Vector2f VisualGame::squareToScreen(Square sq) {
    return {float(OX + sq.col * SQ), float(OY + (7 - sq.rank) * SQ)};
}

sf::Vector2f VisualGame::squareCenter(Square sq) {
    return {float(OX + sq.col * SQ + SQ / 2), float(OY + (7 - sq.rank) * SQ + SQ / 2)};
}

bool VisualGame::isLegalTarget(Square sq) {
    for (const auto& m : selectedMoves)
        if (m.to.rank == sq.rank && m.to.col == sq.col)
            return true;
    return false;
}

// -------------------------------------------------------
// RESOLVE SIDES: wire NNUE to the correct engine based on
// sideConfig_ and nnueOnWhite_
// -------------------------------------------------------
void VisualGame::resolveSides() {
    NNUE::Network* nnueNet = nnueNet_.get();

    if (sideConfig_ == SideConfig::Swapped) {
        nnueOnWhite_ = false;  // NNUE on Black
    } else if (sideConfig_ == SideConfig::Random) {
        nnueOnWhite_ = (rand() % 2 == 0);
    } else {
        // Default / Normal: NNUE on White
        nnueOnWhite_ = true;
    }

    if (nnueOnWhite_) {
        engine_.setNNUE(nnueNet);   // NNUE on White
        engine2_.setNNUE(nullptr);  // classical on Black
        activeEngine_ = &engine_;
    } else {
        engine_.setNNUE(nullptr);   // classical on White
        engine2_.setNNUE(nnueNet);  // NNUE on Black
        activeEngine_ = &engine2_;
    }
}

// -------------------------------------------------------
// SETUP MODE
// -------------------------------------------------------
void VisualGame::enterSetupMode() {
    setupSavedBoard_     = board;
    setupSavedDuckChess_ = isDuckChess_;
    setupMode_           = true;
    setupPaletteIdx_     = 0;
    setupStatus_         = "Setup mode: click board to place piece";
    engine_.stop();
    engine2_.stop();
    if (engineThread.joinable()) engineThread.join();
    engineThinking = false;
}

void VisualGame::exitSetupMode(bool apply) {
    setupMode_ = false;
    if (!apply) {
        board        = setupSavedBoard_;
        isDuckChess_ = setupSavedDuckChess_;
    }
    board.recomputeBitboards();
    MoveList ml; MoveGen::getLegalMoves(board, ml);
    legalMoves.assign(ml.begin(), ml.end());
    pieceSelected = false;
    selectedMoves.clear();
    gameOver = false;
    updateStatus();
}

void VisualGame::handleSetupClick(int x, int y) {
    // Check if click is in the setup panel (right of board)
    int boardRight = OX + SQ * 8;
    if (x >= boardRight) {
        // Click in palette panel — select palette item
        int relY = y - OY;
        int idx  = relY / PALETTE_SQ;
        if (idx >= 0 && idx < 13)  // 12 piece types + 1 clear
            setupPaletteIdx_ = idx;
        return;
    }

    Square sq = screenToSquare(x, y);
    if (!sq.isValid()) return;

    // Palette: 0-5 = white pieces, 6-11 = black pieces, 12 = clear
    static const PieceType pieceOrder[] = {
        PieceType::King, PieceType::Queen, PieceType::Rook,
        PieceType::Bishop, PieceType::Knight, PieceType::Pawn
    };

    if (setupPaletteIdx_ == 12) {
        board.squares[sq.rank][sq.col] = Piece{};
    } else {
        Color  c  = (setupPaletteIdx_ < 6) ? Color::White : Color::Black;
        PieceType pt = pieceOrder[setupPaletteIdx_ % 6];
        board.squares[sq.rank][sq.col] = {pt, c};
    }
    board.recomputeBitboards();
}

// =======================================================
//  AUTOMATE CHESS — Setup Phase
// =======================================================

void VisualGame::enterAutomateSetup() {
    // Save current state for cancel
    setupSavedBoard_     = board;
    setupSavedDuckChess_ = isDuckChess_;

    // Clear board and configure for Automate Chess setup
    board.clearBoard();
    board.isAutomateChess = true;
    board.automateSetupComplete = false;
    board.automateBudget[0] = board.automateBudget[1] = 35;
    board.automatePawnsPlaced[0] = board.automatePawnsPlaced[1] = 0;
    board.automateKingPlaced[0] = board.automateKingPlaced[1] = false;
    board.automateSetupTurn = Color::White;

    isAutomateChess_ = true;
    isDuckChess_ = false;
    automatePaletteType_ = PieceType::Pawn;

    engine_.stop();
    engine2_.stop();
    if (engineThread.joinable()) engineThread.join();
    engineThinking = false;
    gameOver = false;
    pieceSelected = false;
    selectedMoves.clear();

    updateStatus();
}

void VisualGame::exitAutomateSetup(bool apply) {
    if (!apply) {
        board        = setupSavedBoard_;
        isDuckChess_ = setupSavedDuckChess_;
        isAutomateChess_ = false;
    }
    board.recomputeBitboards();
    MoveList ml; MoveGen::getLegalMoves(board, ml);
    legalMoves.assign(ml.begin(), ml.end());
    pieceSelected = false;
    selectedMoves.clear();
    gameOver = false;
    updateStatus();
}

// Piece costs for display
static const char* automatePieceName(PieceType pt) {
    switch (pt) {
        case PieceType::Queen:  return "Queen (7)";
        case PieceType::Rook:   return "Rook (4)";
        case PieceType::Knight: return "Knight (3)";
        case PieceType::Bishop: return "Bishop (3)";
        case PieceType::Pawn:   return "Pawn (1)";
        case PieceType::King:   return "King (free)";
        default:                return "?";
    }
}

void VisualGame::handleAutomateSetupClick(int x, int y) {
    if (!isAutomateChess_ || board.automateSetupComplete) return;

    Color humanSide = Color::White; // human always plays White in Human vs Bot
    // In Human vs Human both sides are human; in Bot vs Bot neither is
    bool humanTurn = (board.automateSetupTurn == humanSide) ||
                     (!engineEnabled && !botVsBot); // Human vs Human

    if (!humanTurn) return; // bot's turn — handled by automateSetupBotPlace()

    // Check if click is in the palette panel (right of board)
    int boardRight = OX + SQ * 8;
    if (x >= boardRight) {
        // Palette click — select piece type
        static const PieceType palette[] = {
            PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
            PieceType::Rook, PieceType::Queen, PieceType::King
        };
        int relY = y - OY;
        int idx = relY / (PALETTE_SQ + 4);
        if (idx >= 0 && idx < 6)
            automatePaletteType_ = palette[idx];
        return;
    }

    Square sq = screenToSquare(x, y);
    if (!sq.isValid()) return;

    Color side = board.automateSetupTurn;
    PieceType pt = automatePaletteType_;

    if (!board.automateCanPlace(side, pt, sq)) {
        // Flash invalid placement
        setupStatus_ = "Invalid placement!";
        return;
    }

    board.automatePlacePiece(side, pt, sq);
    setupStatus_ = "";

    // If king placed and setup complete, transition to play
    if (board.automateSetupComplete) {
        isAutomateChess_ = true; // keep flag for play phase
        MoveList ml; MoveGen::getLegalMoves(board, ml);
        legalMoves.assign(ml.begin(), ml.end());
        updateStatus();
        // Start engine if it's the engine's turn
        if (engineEnabled && board.turn == engineColor && !gameOver)
            startEngineThinking();
        return;
    }

    // If next turn is bot's, trigger bot placement
    if (engineEnabled && board.automateSetupTurn == engineColor) {
        automateSetupBotPlace();
    }

    updateStatus();
}

// Bot places one piece using heuristic: pick the placement that maximises
// the Automate Play NNUE eval (or standard NNUE if not loaded).
bool VisualGame::automateSetupBotPlace() {
    if (!isAutomateChess_ || board.automateSetupComplete) return false;

    Color side = board.automateSetupTurn;
    int ci = (int)side;

    // Determine which piece types are available
    static const PieceType allTypes[] = {
        PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
        PieceType::Rook, PieceType::Queen, PieceType::King
    };

    // Collect all legal placements
    struct Candidate { PieceType pt; Square sq; };
    std::vector<Candidate> candidates;

    for (PieceType pt : allTypes) {
        if (Board::automatePieceCost(pt) > board.automateBudget[ci]) continue;
        if (pt == PieceType::King && board.automateKingPlaced[ci]) continue;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                Square sq{r, c};
                if (board.automateCanPlace(side, pt, sq))
                    candidates.push_back({pt, sq});
            }
        }
    }

    if (candidates.empty()) return false;

    // Score each candidate using NNUE eval after placement
    NNUE::Network* evalNet = automatePlayNet_ ? automatePlayNet_.get() : nnueNet_.get();

    int bestScore = INT_MIN;
    Candidate best = candidates[0];

    for (auto& cand : candidates) {
        Board trial = board;
        trial.automatePlacePiece(side, cand.pt, cand.sq);

        int score = 0;
        if (evalNet) {
            score = evalNet->evaluate(trial);
            // Negate for Black (eval is from White's perspective)
            if (side == Color::Black) score = -score;
        } else {
            // Simple heuristic: prefer higher-value pieces on central squares
            score = Board::automatePieceCost(cand.pt) * 10;
            int fileDist = std::abs(cand.sq.col - 3);
            score -= fileDist;
        }

        if (score > bestScore) {
            bestScore = score;
            best = cand;
        }
    }

    board.automatePlacePiece(side, best.pt, best.sq);

    if (board.automateSetupComplete) {
        MoveList ml; MoveGen::getLegalMoves(board, ml);
        legalMoves.assign(ml.begin(), ml.end());
        updateStatus();
        if (engineEnabled && board.turn == engineColor && !gameOver)
            startEngineThinking();
    } else {
        // If next turn is also bot (Bot vs Bot), schedule another placement
        if (botVsBot && board.automateSetupTurn == engineColor)
            automateSetupBotPlace();
    }

    updateStatus();
    return true;
}

void VisualGame::drawAutomateSetupPanel() {
    // Draw a palette panel to the right of the board showing piece types + costs
    float panelX = float(OX + SQ * 8 + 8);
    float panelY = float(OY);

    static const PieceType palette[] = {
        PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
        PieceType::Rook, PieceType::Queen, PieceType::King
    };
    static const char* labels[] = {
        "Pawn  1pt", "Knight 3pt", "Bishop 3pt",
        "Rook  4pt", "Queen  7pt", "King  free"
    };

    Color side = board.automateSetupTurn;
    int ci = (int)side;

    for (int i = 0; i < 6; i++) {
        PieceType pt = palette[i];
        float cellY = panelY + i * (PALETTE_SQ + 4);

        // Highlight selected
        sf::RectangleShape cell(sf::Vector2f(SETUP_PANEL_W - 8, PALETTE_SQ));
        cell.setPosition({panelX, cellY});
        bool selected = (automatePaletteType_ == pt);
        bool affordable = (Board::automatePieceCost(pt) <= board.automateBudget[ci]);
        cell.setFillColor(selected ? sf::Color(60, 100, 160) :
                          affordable ? sf::Color(40, 40, 60) : sf::Color(30, 30, 40));
        cell.setOutlineColor(selected ? sf::Color(100, 160, 255) : sf::Color(60, 60, 80));
        cell.setOutlineThickness(1);
        window.draw(cell);

        // Label
        sf::Text lbl(font, labels[i], 12);
        lbl.setFillColor(affordable ? sf::Color(200, 200, 220) : sf::Color(80, 80, 100));
        lbl.setPosition({panelX + 4, cellY + 4});
        window.draw(lbl);
    }

    // Budget display
    float budgetY = panelY + 6 * (PALETTE_SQ + 4) + 8;
    sf::Text wBudget(font,
        "White: " + std::to_string(board.automateBudget[0]) + "pt", 13);
    wBudget.setFillColor(sf::Color(220, 220, 255));
    wBudget.setPosition({panelX, budgetY});
    window.draw(wBudget);

    sf::Text bBudget(font,
        "Black: " + std::to_string(board.automateBudget[1]) + "pt", 13);
    bBudget.setFillColor(sf::Color(180, 180, 220));
    bBudget.setPosition({panelX, budgetY + 18});
    window.draw(bBudget);

    // Pawn count
    sf::Text wPawns(font,
        "W pawns: " + std::to_string(board.automatePawnsPlaced[0]) + "/6", 12);
    wPawns.setFillColor(sf::Color(180, 220, 180));
    wPawns.setPosition({panelX, budgetY + 40});
    window.draw(wPawns);

    sf::Text bPawns(font,
        "B pawns: " + std::to_string(board.automatePawnsPlaced[1]) + "/6", 12);
    bPawns.setFillColor(sf::Color(180, 220, 180));
    bPawns.setPosition({panelX, budgetY + 56});
    window.draw(bPawns);

    // Whose turn
    std::string turnStr = (board.automateSetupTurn == Color::White) ?
        "White to place" : "Black to place";
    sf::Text turnTxt(font, turnStr, 13);
    turnTxt.setFillColor(sf::Color(255, 220, 100));
    turnTxt.setPosition({panelX, budgetY + 78});
    window.draw(turnTxt);

    // Status
    if (!setupStatus_.empty()) {
        sf::Text statusTxt(font, setupStatus_, 12);
        statusTxt.setFillColor(sf::Color(255, 100, 100));
        statusTxt.setPosition({panelX, budgetY + 98});
        window.draw(statusTxt);
    }
}

void VisualGame::drawAutomateSetupOverlay() {
    // Highlight valid placement squares for the currently selected piece
    if (board.automateSetupComplete) return;

    Color side = board.automateSetupTurn;
    PieceType pt = automatePaletteType_;

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            Square sq{r, c};
            if (board.automateCanPlace(side, pt, sq)) {
                sf::RectangleShape highlight(sf::Vector2f(SQ - 2, SQ - 2));
                highlight.setPosition({float(OX + c * SQ + 1), float(OY + (7 - r) * SQ + 1)});
                highlight.setFillColor(sf::Color(100, 200, 100, 60));
                highlight.setOutlineColor(sf::Color(100, 200, 100, 120));
                highlight.setOutlineThickness(1);
                window.draw(highlight);
            }
        }
    }
}

