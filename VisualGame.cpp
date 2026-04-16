#include "VisualGame.h"
#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>
#include <random>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#endif

// -------------------------------------------------------
// CONSTRUCTOR / DESTRUCTOR
// -------------------------------------------------------
VisualGame::VisualGame()
{
    // Use desktop resolution to pick a sensible initial window size
    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    unsigned int initW = static_cast<unsigned int>(desktop.size.x * 0.75f);
    unsigned int initH = static_cast<unsigned int>(desktop.size.y * 0.75f);
    // Clamp to a minimum usable size
    initW = std::max(initW, static_cast<unsigned int>(OX * 2 + SQ * 8));
    initH = std::max(initH, static_cast<unsigned int>(OY * 2 + SQ * 8 + 80));

    window.create(sf::VideoMode({initW, initH}), "Chess Engine",
                  sf::Style::Default);
    window.setFramerateLimit(60);
    loadAssets();
    { MoveList ml_; MoveGen::getLegalMoves(board, ml_); legalMoves.assign(ml_.begin(), ml_.end()); }

    // Record starting position hash
    Engine::initZobrist();
    positionHistory_.push_back(Engine::computeHash(board));

    loadOpeningDb();
    updateStatus();
}

VisualGame::~VisualGame() {
    engine_.stop();
    engine2_.stop();
    analysisEngine_.stop();
    // Don't join analysisThread_ — it may be detached
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
            else if (const auto* re = event->getIf<sf::Event::Resized>()) {
                // Reset view to match new window size — prevents stretching on resize/maximize
                sf::FloatRect visibleArea({0.f, 0.f},
                    {float(re->size.x), float(re->size.y)});
                window.setView(sf::View(visibleArea));
            }
            else if (const auto* kp = event->getIf<sf::Event::KeyPressed>()) {
                // Ctrl+C: copy current position FEN to clipboard
                if (kp->code == sf::Keyboard::Key::C &&
                    sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl)) {
                    const Board& b = viewBoard();
                    std::string fen = b.toFEN();
                    // Use Windows clipboard
                    if (OpenClipboard(nullptr)) {
                        EmptyClipboard();
                        HGLOBAL hg = GlobalAlloc(GMEM_MOVEABLE, fen.size() + 1);
                        if (hg) {
                            memcpy(GlobalLock(hg), fen.c_str(), fen.size() + 1);
                            GlobalUnlock(hg);
                            SetClipboardData(CF_TEXT, hg);
                        }
                        CloseClipboard();
                        { std::lock_guard<std::mutex> lk(nnueStatusMutex_);
                          nnueStatus_ = "FEN copied: " + fen.substr(0, 40) + "..."; }
                    }
                }
                // Ctrl+V: paste FEN and load position
                else if (kp->code == sf::Keyboard::Key::V &&
                         sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl)) {
                    if (OpenClipboard(nullptr)) {
                        HANDLE hData = GetClipboardData(CF_TEXT);
                        if (hData) {
                            char* text = static_cast<char*>(GlobalLock(hData));
                            if (text) {
                                std::string fen(text);
                                GlobalUnlock(hData);
                                CloseClipboard();
                                // Try to load the FEN
                                Board tmp;
                                if (tmp.fromFEN(fen)) {
                                    engine_.stop(); engine2_.stop();
                                    stopAnalysisEngine();
                                    if (engineThread.joinable()) engineThread.join();
                                    board = tmp;
                                    board.hash = Engine::computeHash(board);
                                    gameHistory_.clear();
                                    viewIdx_ = -1; analysisMode_ = false;
                                    positionHistory_.clear();
                                    positionHistory_.push_back(board.hash);
                                    MoveList ml; MoveGen::getLegalMoves(board, ml);
                                    legalMoves.assign(ml.begin(), ml.end());
                                    gameOver = false; moveNumber = 1;
                                    pieceSelected = false; selectedMoves.clear();
                                    { std::lock_guard<std::mutex> lk(nnueStatusMutex_);
                                      nnueStatus_ = "FEN loaded"; }
                                    updateStatus();
                                } else {
                                    { std::lock_guard<std::mutex> lk(nnueStatusMutex_);
                                      nnueStatus_ = "Invalid FEN"; }
                                }
                            } else { CloseClipboard(); }
                        } else { CloseClipboard(); }
                    }
                }
                // Ctrl+S: export game as PGN
                else if (kp->code == sf::Keyboard::Key::S &&
                         sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl)) {
                    if (!gameHistory_.empty()) {
                        // Build PGN string
                        std::string pgn;
                        pgn += "[Event \"ChessEngine Game\"]\n";
                        pgn += "[Site \"ChessEngine\"]\n";
                        pgn += "[Date \"" + std::string(__DATE__) + "\"]\n";
                        pgn += "[White \"Engine\"]\n";
                        pgn += "[Black \"Engine\"]\n";
                        pgn += "[Result \"*\"]\n\n";

                        for (int i = 0; i < (int)gameHistory_.size(); ++i) {
                            if (gameHistory_[i].sideToMove == Color::White)
                                pgn += std::to_string(gameHistory_[i].moveNumber) + ". ";
                            pgn += gameHistory_[i].moveAlg;
                            // Eval annotation
                            if (gameHistory_[i].eval != 0) {
                                float ev = gameHistory_[i].eval / 100.f;
                                std::ostringstream oss;
                                oss << std::fixed << std::setprecision(2);
                                if (ev >= 0) oss << "+"; oss << ev;
                                pgn += " {" + oss.str() + "}";
                            }
                            pgn += " ";
                        }
                        pgn += "*\n";

                        // Save to file
                        std::string path = "assets/game_export.pgn";
                        std::ofstream f(path);
                        if (f.is_open()) {
                            f << pgn;
                            f.close();
                            { std::lock_guard<std::mutex> lk(nnueStatusMutex_);
                              nnueStatus_ = "PGN saved: " + path; }
                        } else {
                            { std::lock_guard<std::mutex> lk(nnueStatusMutex_);
                              nnueStatus_ = "PGN save failed"; }
                        }
                    }
                }
                else {
                    handleKeyPress(kp->code);
                }
            }
            else if (const auto* ws = event->getIf<sf::Event::MouseWheelScrolled>()) {
                // Scroll wheel navigates game history
                if (ws->delta < 0) navigateHistory(-1);  // scroll down = back
                else               navigateHistory(+1);  // scroll up  = forward
            }
            else if (setupMode_) {
                if (const auto* mb = event->getIf<sf::Event::MouseButtonPressed>()) {
                    if (mb->button == sf::Mouse::Button::Left) {
                        // Check move list first, then setup panel
                        if (!handleMoveListClick(mb->position.x, mb->position.y))
                            handleSetupClick(mb->position.x, mb->position.y);
                    }
                }
            }
            else if (isPromoting) {
                if (const auto* mb = event->getIf<sf::Event::MouseButtonPressed>()) {
                    if (mb->button == sf::Mouse::Button::Left) {
                        if (!handleMoveListClick(mb->position.x, mb->position.y))
                            handlePromotionClick(mb->position.x, mb->position.y);
                    }
                }
            }
            else if (!botVsBot && !inputLocked() && !engineThinking) {
                if (const auto* mb = event->getIf<sf::Event::MouseButtonPressed>()) {
                    if (mb->button == sf::Mouse::Button::Left) {
                        // Check move list panel first — it's to the right of the board
                        if (!handleMoveListClick(mb->position.x, mb->position.y))
                            handleMouseDown(mb->position.x, mb->position.y);
                    }
                }
                else if (const auto* mm = event->getIf<sf::Event::MouseMoved>()) {
                    handleMouseMove(mm->position.x, mm->position.y);
                }
                else if (const auto* mr = event->getIf<sf::Event::MouseButtonReleased>()) {
                    if (mr->button == sf::Mouse::Button::Left)
                        handleMouseUp(mr->position.x, mr->position.y);
                }
            }
            else if (const auto* mb = event->getIf<sf::Event::MouseButtonPressed>()) {
                // Bot vs bot mode — still allow move list clicks
                if (mb->button == sf::Mouse::Button::Left)
                    handleMoveListClick(mb->position.x, mb->position.y);
            }
        }

        // Update animation
        if (isAnimating)
            updateAnimation();

        // Check if engine finished thinking
        if (engineThinking && engineDone.load())
            checkEngineResult();

        // Poll analysis engine stats
        if (analysisMode_)
            updateAnalysisStats();

        // Bot vs Bot / Bot vs NNUE: after engine finishes + delay, apply the pending move
        if ((botVsBot || botVsNNUE_) && botHasPendingMove_ && !botPaused && !gameOver) {
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

    // Reset game history and analysis mode
    gameHistory_.clear();
    viewIdx_      = -1;
    analysisMode_ = false;
    stopAnalysisEngine();

    // If in bot-vs-bot or bot-vs-NNUE mode, restart
    if ((botVsBot || botVsNNUE_) && !botPaused) {
        startEngineThinking();
    }

    updateStatus();
}

// -------------------------------------------------------
// ANALYSIS NAVIGATION
// -------------------------------------------------------
void VisualGame::exitAnalysisMode() {
    analysisPending_ = false;
    stopAnalysisEngine();
    analysisMode_ = false;
    viewIdx_      = -1;
    // Restore live board's legal moves
    MoveList ml; MoveGen::getLegalMoves(board, ml);
    legalMoves.assign(ml.begin(), ml.end());
    pieceSelected = false;
    selectedMoves.clear();
    // Unpause bot if it was paused by navigation
    // Don't start thinking here if we're already in the middle of processing a move
    if ((botVsBot || botVsNNUE_) && botPaused) {
        botPaused = false;
        if (!engineThinking && !isAnimating && !botHasPendingMove_ && !gameOver)
            startEngineThinking();
    }
    updateStatus();
}

void VisualGame::navigateHistory(int delta) {
    if (gameHistory_.empty()) return;

    int newIdx;
    if (!analysisMode_) {
        // Currently live — going back enters analysis mode
        if (delta < 0) {
            newIdx = static_cast<int>(gameHistory_.size()) - 1;
            newIdx += delta;  // step back from last entry
        } else {
            return;  // already at live, can't go forward
        }
    } else {
        newIdx = viewIdx_ + delta;
    }

    if (newIdx < 0) newIdx = 0;

    // Scrolling forward past the last entry = return to live
    if (newIdx >= static_cast<int>(gameHistory_.size())) {
        exitAnalysisMode();
        // If in bot mode, unpause and resume
        if ((botVsBot || botVsNNUE_) && botPaused) {
            botPaused = false;
            if (!engineThinking && !isAnimating && !botHasPendingMove_ && !gameOver)
                startEngineThinking();
            updateStatus();
        }
        return;
    }

    // Entering analysis mode for the first time — auto-pause bot if running
    if (!analysisMode_ && (botVsBot || botVsNNUE_) && !botPaused) {
        botPaused = true;
        if (engineThinking && activeEngine_)
            activeEngine_->stop();
    }

    viewIdx_      = newIdx;
    analysisMode_ = true;
    pieceSelected = false;
    selectedMoves.clear();

    // Compute legal moves for the viewed position
    Board tmp = gameHistory_[viewIdx_].board;
    MoveList ml; MoveGen::getLegalMoves(tmp, ml);
    legalMoves.assign(ml.begin(), ml.end());

    // Stop any running analysis — position changed
    if (analysisThread_.joinable()) {
        analysisEngine_.stop();
        analysisThread_.join();
    }
    analysisDepth_ = 0;
    analysisNodes_ = 0;
    analysisPending_ = false;
    updateStatus();
}

// -------------------------------------------------------
// MOVE LIST CLICK — jump to position by clicking the panel
// -------------------------------------------------------
bool VisualGame::handleMoveListClick(int x, int y) {
    if (gameHistory_.empty()) return false;

    // Replicate the panel geometry from drawMoveList
    float panelX = float(dynOX_ + dynSQ_ * 8 + 12);
    float panelW = float(window.getSize().x) - panelX - 8.f;
    if (panelW < 80.f) return false;

    // Only handle clicks inside the panel
    if (float(x) < panelX || float(x) > panelX + panelW) return false;

    float boardH  = float(dynSQ_ * 8);
    float listH   = boardH * 0.55f;
    float panelY  = float(OY);

    // Only handle clicks in the move list area (not stats/graph below)
    if (float(y) < panelY || float(y) > panelY + listH) {
        // Check if click is in the stats area (PV expand toggles)
        if (float(y) > panelY + listH && float(y) < panelY + boardH &&
            float(x) >= panelX && float(x) <= panelX + panelW) {
            // Toggle expand for PV lines — approximate hit test
            // Each PV line is ~(sfs2+4) px tall, starting at statsAreaY + some offset
            // Simple approach: toggle line 0/1/2 based on click position
            float statsAreaY2 = panelY + listH;
            float relY = float(y) - statsAreaY2;
            unsigned int sfs2 = std::max(9u, unsigned(11 * scale_));
            float lineH2 = float(sfs2) + 6.f;
            // Skip hint/opening/stats lines (~3 lines)
            float pvStartY = float(sfs2) * 3.f + 20.f;
            if (relY > pvStartY) {
                int pvIdx = int((relY - pvStartY) / (lineH2 * 2.f));
                if (pvIdx >= 0 && pvIdx < 3)
                    pvLineExpanded_[pvIdx] = !pvLineExpanded_[pvIdx];
            }
            return true;
        }
        return false;
    }

    unsigned int fs = std::max(11u, unsigned(13 * scale_));
    float lineH     = float(fs) + 5.f;
    float headerH   = lineH + 2.f;

    // ── Nav button hit test (header row) ────────────────────────────────────
    float btnH    = headerH - 4.f;
    float btnW    = btnH * 1.1f;
    float btnY    = panelY + 2.f;
    float PAD     = 4.f;
    float btnBackX = panelX + PAD;
    float btnFwdX  = btnBackX + btnW + 4.f;

    if (float(y) >= btnY && float(y) <= btnY + btnH) {
        // Click in header row — check nav buttons
        if (float(x) >= btnBackX && float(x) <= btnBackX + btnW) {
            // Back button
            int total = int(gameHistory_.size());
            bool canGoBack = total > 0 && (analysisMode_ ? viewIdx_ > 0 : total > 0);
            if (canGoBack) navigateHistory(-1);
            return true;
        }
        if (float(x) >= btnFwdX && float(x) <= btnFwdX + btnW) {
            // Forward button
            if (analysisMode_) navigateHistory(+1);
            return true;
        }
        return true;  // click in header but not on a button — consume it
    }

    float contentH  = listH - headerH;
    int   maxVisible = std::max(1, int(contentH / lineH));

    int total        = int(gameHistory_.size());
    int highlighted  = analysisMode_ ? viewIdx_ : total - 1;
    int totalRows    = (total + 1) / 2;
    int highlightedRow = highlighted / 2;
    int scrollOffset   = std::max(0, highlightedRow - maxVisible / 2);
    scrollOffset       = std::min(scrollOffset, std::max(0, totalRows - maxVisible));

    // Which row was clicked?
    float startY = panelY + headerH;
    int clickedRow = int((float(y) - startY) / lineH) + scrollOffset;
    if (clickedRow < 0 || clickedRow >= totalRows) return true;  // in panel but no row

    // Which column — white (left half) or black (right half)?
    float SB_W      = std::max(6.f, 8.f * scale_);
    float colAreaW  = panelW - SB_W - 4.f - float(fs) * 2.2f - 4.f;
    float moveStartX = panelX + 4.f + float(fs) * 2.2f;
    float colW      = colAreaW / 2.f;

    int moveIdx;
    if (float(x) < moveStartX + colW)
        moveIdx = clickedRow * 2;       // white's move
    else
        moveIdx = clickedRow * 2 + 1;  // black's move

    if (moveIdx >= total) moveIdx = total - 1;
    if (moveIdx < 0) return true;

    // Navigate to the clicked position
    // Compute delta from current position
    int currentIdx = analysisMode_ ? viewIdx_ : total;  // total = "live" position
    int delta = moveIdx - currentIdx;
    if (delta == 0) return true;  // already there

    // Jump directly rather than stepping — set viewIdx_ directly
    if (moveIdx >= total - 1 && !analysisMode_) {
        // Clicking the last move when already live — no-op
        return true;
    }

    // Enter analysis mode at the clicked position
    if (!analysisMode_ && (botVsBot || botVsNNUE_) && !botPaused) {
        botPaused = true;
        if (engineThinking && activeEngine_) activeEngine_->stop();
    }

    viewIdx_      = moveIdx;
    analysisMode_ = true;
    pieceSelected = false;
    selectedMoves.clear();

    Board tmp = gameHistory_[viewIdx_].board;
    MoveList ml; MoveGen::getLegalMoves(tmp, ml);
    legalMoves.assign(ml.begin(), ml.end());

    if (analysisThread_.joinable()) {
        analysisEngine_.stop();
        analysisThread_.join();
    }
    analysisDepth_ = 0;
    analysisNodes_ = 0;
    analysisPending_ = false;
    updateStatus();
    return true;
}

// -------------------------------------------------------
// OPENING DATABASE
// -------------------------------------------------------
void VisualGame::loadOpeningDb() {
    static const char* files[] = {
        "assets/eco_a.tsv", "assets/eco_b.tsv", "assets/eco_c.tsv",
        "assets/eco_d.tsv", "assets/eco_e.tsv"
    };
    for (const char* path : files) {
        std::ifstream f(path);
        if (!f.is_open()) continue;
        std::string line;
        std::getline(f, line);  // skip header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            // Parse TSV: eco \t name \t pgn
            size_t t1 = line.find('\t');
            if (t1 == std::string::npos) continue;
            size_t t2 = line.find('\t', t1 + 1);
            if (t2 == std::string::npos) continue;
            std::string eco  = line.substr(0, t1);
            std::string name = line.substr(t1 + 1, t2 - t1 - 1);
            std::string pgn  = line.substr(t2 + 1);
            // Strip move numbers: "1. e4 e5 2. Nf3" -> "e4 e5 Nf3"
            std::string moves;
            std::istringstream ss(pgn);
            std::string tok;
            while (ss >> tok) {
                if (tok.back() == '.') continue;  // move number
                if (!moves.empty()) moves += ' ';
                moves += tok;
            }
            openingDb_.push_back({eco, name, moves});
        }
    }
    // Sort by move sequence length descending — longest match wins
    std::sort(openingDb_.begin(), openingDb_.end(),
              [](const OpeningEntry& a, const OpeningEntry& b){
                  return a.moves.size() > b.moves.size();
              });
}

void VisualGame::updateOpeningName() {
    if (openingDb_.empty() || gameHistory_.empty()) {
        currentOpening_.clear();
        return;
    }
    // Build move sequence from history (algebraic, space-separated)
    std::string seq;
    int limit = std::min(int(gameHistory_.size()), 30);  // only check first 30 moves
    for (int i = 0; i < limit; ++i) {
        if (!seq.empty()) seq += ' ';
        seq += gameHistory_[i].moveAlg;
    }
    // Find longest opening that is a prefix of our move sequence
    for (const auto& entry : openingDb_) {
        if (entry.moves.empty()) continue;
        // Check if entry.moves is a prefix of seq (or equal)
        if (seq.size() >= entry.moves.size() &&
            seq.substr(0, entry.moves.size()) == entry.moves &&
            (seq.size() == entry.moves.size() || seq[entry.moves.size()] == ' ')) {
            currentOpening_ = entry.eco + " " + entry.name;
            return;
        }
    }
    currentOpening_.clear();
}

// -------------------------------------------------------
// ANALYSIS ENGINE
// -------------------------------------------------------
void VisualGame::stopAnalysisEngine() {
    // Signal the engine to stop — it checks stop_ every 4096 nodes (~few ms)
    analysisEngine_.stop();
    // Join the thread to ensure the engine object is safe to reuse
    if (analysisThread_.joinable())
        analysisThread_.join();
    analysisDepth_ = 0;
    analysisNodes_ = 0;
}

void VisualGame::startAnalysisEngine() {
    // Called after debounce — safe to join here since user has stopped scrolling
    stopAnalysisEngine();
    if (!analysisMode_ || viewIdx_ < 0 || viewIdx_ >= (int)gameHistory_.size()) return;

    analysisViewIdx_ = viewIdx_;
    analysisDone_.store(false);

    analysisEngine_.setNNUE(nnueEnabled_ ? nnueNet_.get() : nullptr);
    analysisEngine_.setMultiPV(1);
    analysisEngine_.setTimeLimit(10000);  // 10s — user can navigate away

    Board boardCopy = gameHistory_[viewIdx_].board;
    boardCopy.hash  = Engine::computeHash(boardCopy);

    analysisThread_ = std::thread([this, boardCopy]() mutable {
        analysisEngine_.getBestMove(boardCopy, 64);
        analysisDone_.store(true);
    });
}

void VisualGame::updateAnalysisStats() {
    if (!analysisMode_) return;
    analysisDepth_ = analysisEngine_.getLiveDepth();
    analysisEval_  = analysisEngine_.getLiveEval();
    // Use live node count from the current search
    analysisNodes_ = analysisEngine_.getNodes();
    // Store latest eval back into the history entry so the graph updates
    if (viewIdx_ >= 0 && viewIdx_ < (int)gameHistory_.size() && analysisDepth_ > 0)
        gameHistory_[viewIdx_].eval = analysisEval_;
}
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

    // In fast mode during bot-vs-bot or bot-vs-NNUE, skip animation
    if (fastMode && (botVsBot || botVsNNUE_))
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
// MOVE TO ALGEBRAIC NOTATION
// -------------------------------------------------------
std::string VisualGame::moveToAlgebraic(const Board& board, const Move& move) {
    auto sq2s = [](Square s) -> std::string {
        return std::string(1, char('a' + s.col)) + std::string(1, char('1' + s.rank));
    };

    Piece moving   = board.getPiece(move.from);
    Piece captured = board.getPiece(move.to);
    bool  isCapture = !captured.isNone() && !captured.isDuck();

    // En passant
    bool isEP = (moving.type == PieceType::Pawn &&
                 move.from.col != move.to.col && captured.isNone());
    if (isEP) isCapture = true;

    // Castling
    if (moving.type == PieceType::King) {
        int dc = move.to.col - move.from.col;
        if (dc == 2)  return "O-O";
        if (dc == -2) return "O-O-O";
    }

    std::string result;

    // Piece letter (pawns have none)
    if (moving.type != PieceType::Pawn) {
        switch (moving.type) {
            case PieceType::Knight: result += "N"; break;
            case PieceType::Bishop: result += "B"; break;
            case PieceType::Rook:   result += "R"; break;
            case PieceType::Queen:  result += "Q"; break;
            case PieceType::King:   result += "K"; break;
            default: break;
        }
    }

    // Disambiguation: find other pieces of the same type that can reach move.to
    bool ambigFile = false, ambigRank = false;
    if (moving.type != PieceType::Pawn) {
        MoveList allMoves; MoveGen::getLegalMoves(const_cast<Board&>(board), allMoves);
        for (const auto& m : allMoves) {
            if (m == move) continue;
            Piece other = board.getPiece(m.from);
            if (other.type == moving.type && other.color == moving.color &&
                m.to.rank == move.to.rank && m.to.col == move.to.col) {
                if (m.from.col == move.from.col) ambigRank = true;
                else                              ambigFile = true;
            }
        }
    }

    // Pawn captures always include the source file
    if (moving.type == PieceType::Pawn && isCapture)
        result += char('a' + move.from.col);
    else if (ambigFile || (!ambigRank && ambigFile))
        result += char('a' + move.from.col);
    else if (ambigRank)
        result += char('1' + move.from.rank);
    else if (ambigFile && ambigRank)
        result += sq2s(move.from);

    if (isCapture) result += "x";
    result += sq2s(move.to);

    // Promotion
    if (move.promotion != PieceType::None) {
        result += "=";
        switch (move.promotion) {
            case PieceType::Queen:  result += "Q"; break;
            case PieceType::Rook:   result += "R"; break;
            case PieceType::Bishop: result += "B"; break;
            case PieceType::Knight: result += "N"; break;
            default: break;
        }
    }

    // Check / checkmate suffix — apply the move to a copy and test
    Board tmp = board;
    tmp.applyMove(move);
    MoveList responses; MoveGen::getLegalMoves(tmp, responses);
    bool inCheck = MoveGen::isInCheck(tmp, tmp.turn);
    if (inCheck) {
        result += responses.empty() ? "#" : "+";
    }

    return result;
}

// -------------------------------------------------------
// FINISH MOVE
// -------------------------------------------------------
void VisualGame::finishMove(const Move& move) {
    lastMove    = move;
    hasLastMove = true;

    // ── Record history entry (pre-move board state) ──────────────────────────
    {
        HistoryEntry entry;
        entry.board      = board;   // snapshot before the move
        entry.move       = move;
        entry.sideToMove = board.turn;
        entry.moveNumber = moveNumber;
        // Store the eval of this position (from the engine that just played)
        // lastEval_ is always white POV — correct for the graph
        entry.eval = lastEval_;
        // Algebraic notation (e.g. "Nf3", "exd5", "O-O", "e8=Q+")
        entry.moveAlg = moveToAlgebraic(board, move);
        // Only truncate future history if a HUMAN makes a move while in analysis mode
        // (bot moves always append to the live game — never branch from a viewed position)
        bool isHumanMove = !botVsBot && !botVsNNUE_ && !engineThinking;
        if (analysisMode_ && isHumanMove) {
            gameHistory_.resize(viewIdx_ + 1);
            exitAnalysisMode();
        } else if (analysisMode_ && !isHumanMove) {
            // Bot move during analysis — just exit analysis mode, keep full history
            exitAnalysisMode();
        }
        gameHistory_.push_back(std::move(entry));

        // Compute cpLoss for the previous move now that we have the eval after it
        // cpLoss[N-1] = eval[N-1] (before move N-1) - eval[N] (before move N)
        // Both from the perspective of the side that played move N-1
        int n = (int)gameHistory_.size();
        if (n >= 2) {
            int prevEval = gameHistory_[n-2].eval;  // eval before move n-2
            int currEval = gameHistory_[n-1].eval;  // eval before move n-1 = after move n-2
            Color mover  = gameHistory_[n-2].sideToMove;
            int prevPOV  = (mover == Color::White) ? prevEval : -prevEval;
            int currPOV  = (mover == Color::White) ? currEval : -currEval;
            int loss = prevPOV - currPOV;
            gameHistory_[n-2].cpLoss = std::max(0, loss);
        }
    }
    updateOpeningName();

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
        if (botVsBot || botVsNNUE_) {
            // In bot vs bot / bot vs NNUE, start thinking immediately (delay comes after for botVsBot)
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
    // Select the correct engine for the current side:
    // - botVsBot: engine_ for White, engine2_ for Black
    // - botVsNNUE_: engine_ and engine2_ assigned by resolveSides() (NNUE vs Classical)
    //   nnueOnWhite_=true  → engine_=NNUE(White), engine2_=Classical(Black)
    //   nnueOnWhite_=false → engine_=Classical(White), engine2_=NNUE(Black)
    Engine* eng;
    if (botVsBot) {
        eng = (board.turn == Color::Black) ? &engine2_ : &engine_;
    } else if (botVsNNUE_) {
        // engine2_ plays Black, engine_ plays White (regardless of NNUE assignment)
        eng = (board.turn == Color::Black) ? &engine2_ : &engine_;
    } else {
        eng = &engine_;
    }
    eng->setTimeLimit((botVsBot || botVsNNUE_) ? (fastMode ? 50 : botThinkMs) : engineTimeMs);

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

    // Save last eval for eval bar
    if (activeEngine_) {
        int liveVal = activeEngine_->getLiveEval();
        lastEval_ = liveVal;
        // For botVsNNUE_, also track engine2_ eval separately for dual eval bar
        if (botVsNNUE_ && activeEngine_ == &engine2_)
            lastEval2_ = liveVal;
        else if (botVsNNUE_)
            lastEval_ = liveVal;
    }

    if ((botVsBot || botVsNNUE_) && !fastMode) {
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

        if (fastMode && (botVsBot || botVsNNUE_)) {
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
        statusMsg = "Automate Setup - " + setupSide + " to place | Budget: " +
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
            std::string winner = (board.turn == Color::White) ? "Black" : "White";
            statusMsg = winner + " wins! " + side + " has no legal moves!";
            // Set terminal eval: winner gets +MATE, loser gets -MATE (white POV)
            if (!gameHistory_.empty())
                gameHistory_.back().eval = (board.turn == Color::White) ? -Engine::MATE_SCORE : Engine::MATE_SCORE;
        } else {
            if (MoveGen::isInCheck(board, board.turn)) {
                std::string winner = (board.turn == Color::White) ? "Black" : "White";
                statusMsg = "Checkmate! " + winner + " wins!";
                // White's turn = black delivered checkmate = black wins = negative for white
                if (!gameHistory_.empty())
                    gameHistory_.back().eval = (board.turn == Color::White) ? -Engine::MATE_SCORE : Engine::MATE_SCORE;
            }
            else {
                statusMsg = "Stalemate -- Draw!";
                if (!gameHistory_.empty())
                    gameHistory_.back().eval = 0;
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
    int col = (x - dynOX_) / dynSQ_;
    int row = 7 - (y - OY) / dynSQ_;
    if (col < 0 || col > 7 || row < 0 || row > 7)
        return {-1, -1};
    return {row, col};
}

sf::Vector2f VisualGame::squareToScreen(Square sq) {
    return {float(dynOX_ + sq.col * dynSQ_), float(OY + (7 - sq.rank) * dynSQ_)};
}

sf::Vector2f VisualGame::squareCenter(Square sq) {
    return {float(dynOX_ + sq.col * dynSQ_ + dynSQ_ / 2), float(OY + (7 - sq.rank) * dynSQ_ + dynSQ_ / 2)};
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
    int boardRight = dynOX_ + dynSQ_ * 8;
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
    int boardRight = dynOX_ + dynSQ_ * 8;
    if (x >= boardRight) {
        // Palette click — select piece type
        static const PieceType palette[] = {
            PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
            PieceType::Rook, PieceType::Queen, PieceType::King
        };
        int relY = y - OY;
        int idx = relY / int(float(PALETTE_SQ + 4) * scale_);
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
    float panelX = float(dynOX_ + dynSQ_ * 8 + 8);
    float panelY = float(OY);
    float scaledPalSQ = float(PALETTE_SQ) * scale_;
    float scaledPanW  = float(SETUP_PANEL_W) * scale_;

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
        float cellY = panelY + i * (scaledPalSQ + 4);

        sf::RectangleShape cell(sf::Vector2f(scaledPanW - 8, scaledPalSQ));
        cell.setPosition({panelX, cellY});
        bool selected = (automatePaletteType_ == pt);
        bool affordable = (Board::automatePieceCost(pt) <= board.automateBudget[ci]);
        cell.setFillColor(selected ? sf::Color(60, 100, 160) :
                          affordable ? sf::Color(40, 40, 60) : sf::Color(30, 30, 40));
        cell.setOutlineColor(selected ? sf::Color(100, 160, 255) : sf::Color(60, 60, 80));
        cell.setOutlineThickness(1);
        window.draw(cell);

        unsigned int fs = std::max(8u, unsigned(12 * scale_));
        sf::Text lbl(font, labels[i], fs);
        lbl.setFillColor(affordable ? sf::Color(200, 200, 220) : sf::Color(80, 80, 100));
        lbl.setPosition({panelX + 4, cellY + 4});
        window.draw(lbl);
    }

    float budgetY = panelY + 6 * (scaledPalSQ + 4) + 8;
    unsigned int fs13 = std::max(8u, unsigned(13 * scale_));
    unsigned int fs12 = std::max(8u, unsigned(12 * scale_));

    sf::Text wBudget(font, "White: " + std::to_string(board.automateBudget[0]) + "pt", fs13);
    wBudget.setFillColor(sf::Color(220, 220, 255));
    wBudget.setPosition({panelX, budgetY});
    window.draw(wBudget);

    sf::Text bBudget(font, "Black: " + std::to_string(board.automateBudget[1]) + "pt", fs13);
    bBudget.setFillColor(sf::Color(180, 180, 220));
    bBudget.setPosition({panelX, budgetY + 18 * scale_});
    window.draw(bBudget);

    sf::Text wPawns(font, "W pawns: " + std::to_string(board.automatePawnsPlaced[0]) + "/6", fs12);
    wPawns.setFillColor(sf::Color(180, 220, 180));
    wPawns.setPosition({panelX, budgetY + 40 * scale_});
    window.draw(wPawns);

    sf::Text bPawns(font, "B pawns: " + std::to_string(board.automatePawnsPlaced[1]) + "/6", fs12);
    bPawns.setFillColor(sf::Color(180, 220, 180));
    bPawns.setPosition({panelX, budgetY + 56 * scale_});
    window.draw(bPawns);

    std::string turnStr = (board.automateSetupTurn == Color::White) ? "White to place" : "Black to place";
    sf::Text turnTxt(font, turnStr, fs13);
    turnTxt.setFillColor(sf::Color(255, 220, 100));
    turnTxt.setPosition({panelX, budgetY + 78 * scale_});
    window.draw(turnTxt);

    if (!setupStatus_.empty()) {
        sf::Text statusTxt(font, setupStatus_, fs12);
        statusTxt.setFillColor(sf::Color(255, 100, 100));
        statusTxt.setPosition({panelX, budgetY + 98 * scale_});
        window.draw(statusTxt);
    }
}

void VisualGame::drawAutomateSetupOverlay() {
    if (board.automateSetupComplete) return;

    Color side = board.automateSetupTurn;
    PieceType pt = automatePaletteType_;

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            Square sq{r, c};
            if (board.automateCanPlace(side, pt, sq)) {
                sf::RectangleShape highlight(sf::Vector2f(float(dynSQ_ - 2), float(dynSQ_ - 2)));
                highlight.setPosition({float(dynOX_ + c * dynSQ_ + 1), float(OY + (7 - r) * dynSQ_ + 1)});
                highlight.setFillColor(sf::Color(100, 200, 100, 60));
                highlight.setOutlineColor(sf::Color(100, 200, 100, 120));
                highlight.setOutlineThickness(1);
                window.draw(highlight);
            }
        }
    }
}


