#include "VisualGame.h"
#include <iostream>
#include <cmath>
#include <sstream>
#include <random>
#include <chrono>
#include <stdexcept>

// -------------------------------------------------------
// CONSTRUCTOR / DESTRUCTOR
// -------------------------------------------------------
VisualGame::VisualGame()
    : window(sf::VideoMode({static_cast<unsigned>(OX * 2 + SQ * 8),
                            static_cast<unsigned>(OY * 2 + SQ * 8 + 80)}), "Chess Engine")
{
    window.setFramerateLimit(60);
    loadAssets();
    legalMoves = MoveGen::getLegalMoves(board);

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
    if (nnueThread_.joinable())
        nnueThread_.join();
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
    return isAnimating || isPromoting || gameOver;
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
            else if (const auto* te = event->getIf<sf::Event::TextEntered>()) {
                if (nnueInputMode_ || nnueEloInputMode_) {
                    handleTextInput(te->unicode);
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
// TEXT INPUT HANDLER (for training config)
// -------------------------------------------------------
void VisualGame::handleTextInput(uint32_t unicode) {
    if (unicode == 13 || unicode == 10) { // Enter
        if (nnueEloInputMode_)
            processEloInput();
        else
            processTrainingInput();
    }
    else if (unicode == 8) { // Backspace
        if (!nnueInputBuffer_.empty())
            nnueInputBuffer_.pop_back();
    }
    else if (unicode == 27) { // Escape
        nnueInputMode_ = false;
        nnueEloInputMode_ = false;
        nnueInputBuffer_.clear();
        nnueStatus_ = "";
    }
    else if (unicode >= '0' && unicode <= '9') {
        nnueInputBuffer_ += static_cast<char>(unicode);
    }

    // Update display
    if (nnueInputMode_) {
        if (nnueInputStep_ == 0)
            nnueStatus_ = "# games to generate (0=skip): " + nnueInputBuffer_ + "_";
        else if (nnueInputStep_ == 1)
            nnueStatus_ = "Max positions (0=all): " + nnueInputBuffer_ + "_";
        else if (nnueInputStep_ == 2)
            nnueStatus_ = "# epochs (0=default 50): " + nnueInputBuffer_ + "_";
    }
    else if (nnueEloInputMode_) {
        nnueStatus_ = "# games for ELO test (0=default 100): " + nnueInputBuffer_ + "_";
    }
}

// -------------------------------------------------------
// PROCESS TRAINING INPUT
// -------------------------------------------------------
void VisualGame::processTrainingInput() {
    int value = 0;
    if (!nnueInputBuffer_.empty()) {
        try {
            value = std::stoi(nnueInputBuffer_);
        } catch (const std::exception&) {
            value = 0;
        }
    }
    nnueInputBuffer_.clear();

    if (nnueInputStep_ == 0) {
        nnueConfigGames_ = value;
        nnueInputStep_ = 1;
        nnueStatus_ = "Max positions (0=all): _";
    }
    else if (nnueInputStep_ == 1) {
        nnueConfigMaxPositions_ = value;
        nnueInputStep_ = 2;
        nnueStatus_ = "# epochs (0=default 50): _";
    }
    else if (nnueInputStep_ == 2) {
        nnueConfigEpochs_ = (value <= 0) ? 50 : value;
        nnueInputMode_ = false;
        startTraining();
    }
}

// -------------------------------------------------------
// ETA HELPERS
// -------------------------------------------------------
void VisualGame::updateETA(std::chrono::steady_clock::time_point startTime, int done, int total) {
    if (done <= 0 || total <= 0) { nnueETAEndMs_ = 0; return; }
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    double elapsedSec = std::chrono::duration<double>(elapsed).count();
    double rate = elapsedSec / done;
    auto remainingMs = static_cast<int64_t>(rate * (total - done) * 1000.0);
    auto endMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() + remainingMs;
    nnueETAEndMs_ = endMs;
}

static std::string formatCountdown(int64_t endMs) {
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    auto remainMs = endMs - nowMs;
    if (remainMs <= 0) return " ~0s left";
    int secs = static_cast<int>(remainMs / 1000);
    if (secs < 60)
        return " ~" + std::to_string(secs) + "s left";
    int mins = secs / 60;
    secs = secs % 60;
    return " ~" + std::to_string(mins) + "m" + std::to_string(secs) + "s left";
}

// -------------------------------------------------------
// PROCESS ELO INPUT
// -------------------------------------------------------
void VisualGame::processEloInput() {
    int value = 0;
    if (!nnueInputBuffer_.empty()) {
        try {
            value = std::stoi(nnueInputBuffer_);
        } catch (const std::exception&) {
            value = 0;
        }
    }
    nnueInputBuffer_.clear();
    nnueEloInputMode_ = false;

    nnueConfigEloGames_ = (value <= 0) ? 100 : value;
    startEloEstimation();
}

// -------------------------------------------------------
// START ELO ESTIMATION
// -------------------------------------------------------
void VisualGame::startEloEstimation() {
    // Load weights if no net yet
    if (!nnueNet_) {
        auto tempNet = std::make_unique<NNUE::Network>();
        if (tempNet->loadWeights("assets/nnue_weights.bin")) {
            nnueNet_ = std::move(tempNet);
        } else {
            nnueStatus_ = "Train first (press T)!";
            return;
        }
    }

    if (nnueTraining_ || nnueEstimating_) return;

    nnueEstimating_ = true;
    nnueCancelFlag_.store(false);
    nnueStatus_ = "Estimating ELO (" + std::to_string(nnueConfigEloGames_) + " games)...";

    // Stop bot-vs-bot if active
    if (botVsBot) {
        botVsBot = false;
        engine_.stop();
        engine2_.stop();
        if (engineThread.joinable()) engineThread.join();
        engineThinking = false;
    }

    if (nnueThread_.joinable()) nnueThread_.join();

    int eloGames = nnueConfigEloGames_;
    nnueThread_ = std::thread([this, eloGames]() {
        NNUE::Trainer trainer;
        auto eloStart = std::chrono::steady_clock::now();
        lastEloResult_ = trainer.estimateElo(*nnueNet_, eloGames, 500, 2100,
            [this, eloStart](int done, int total) {
                nnueStatus_ = "ELO test: game " + std::to_string(done) + "/" + std::to_string(total);
                updateETA(eloStart, done, total);
            }, &nnueCancelFlag_);

        if (nnueCancelFlag_.load()) {
            nnueStatus_ = "ELO estimation cancelled. Partial: " +
                std::to_string(lastEloResult_.estimatedElo) +
                " (W:" + std::to_string(lastEloResult_.wins) +
                " D:" + std::to_string(lastEloResult_.draws) +
                " L:" + std::to_string(lastEloResult_.losses) + ")";
        } else {
            hasEloResult_ = true;
            nnueStatus_ = "ELO: " + std::to_string(lastEloResult_.estimatedElo) +
                " (W:" + std::to_string(lastEloResult_.wins) +
                " D:" + std::to_string(lastEloResult_.draws) +
                " L:" + std::to_string(lastEloResult_.losses) + ")";
        }
        nnueEstimating_ = false;
        nnueETAEndMs_ = 0;
    });
    updateStatus();
}

// -------------------------------------------------------
// START TRAINING (extracted from old T key handler)
// -------------------------------------------------------
void VisualGame::startTraining() {
    nnueTraining_ = true;
    nnueCancelFlag_.store(false);

    if (nnueConfigGames_ == 0)
        nnueStatus_ = "Skipping data gen, loading existing positions...";
    else
        nnueStatus_ = "Generating training data...";

    // Stop bot-vs-bot if active
    if (botVsBot) {
        botVsBot = false;
        engine_.stop();
        engine2_.stop();
        if (engineThread.joinable()) engineThread.join();
        engineThinking = false;
    }

    if (nnueThread_.joinable()) nnueThread_.join();

    nnueThread_ = std::thread([this]() {
        if (!nnueNet_)
            nnueNet_ = std::make_unique<NNUE::Network>();

        NNUE::Trainer trainer;
        NNUE::TrainingConfig config;
        config.numGames = nnueConfigGames_;
        config.thinkTimeMs = 50;          // CHANGED: 100 -> 50 for 2x faster generation
        config.epochs = nnueConfigEpochs_;
        config.batchSize = 512;
        config.learningRate = 0.001f;
        config.outputPath = "assets/nnue_weights.bin";
        config.dataPath = "assets/training_data.bin";
        config.numThreads = 0;
        config.earlyStopPatience = 15;    // In-app training uses lower patience (Python script uses 100)
        config.mirrorPositions = true;
        config.appendExistingData = true;
        config.phaseBalancedTraining = true;

        // === Phase 1: Load existing data ===
        std::vector<NNUE::TrainingPosition> existingData;
        if (config.appendExistingData) {
            existingData = trainer.loadTrainingData(config.dataPath);
            if (!existingData.empty()) {
                nnueStatus_ = "Loaded " + std::to_string(existingData.size()) + " existing positions.";
            }
        }

        if (nnueCancelFlag_.load()) { nnueTraining_ = false; nnueETAEndMs_ = 0; nnueStatus_ = "Cancelled."; return; }

        // === Phase 1.5: Limit existing positions if requested ===
        if (nnueConfigMaxPositions_ > 0 && static_cast<int>(existingData.size()) > nnueConfigMaxPositions_) {
            std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::shuffle(existingData.begin(), existingData.end(), rng);
            existingData.resize(nnueConfigMaxPositions_);
            nnueStatus_ = "Using " + std::to_string(nnueConfigMaxPositions_) + " random existing positions.";
        }

        // === Phase 2: Generate new data (skip if numGames == 0) ===
        std::vector<NNUE::TrainingPosition> newData;
        if (config.numGames > 0) {
            auto dataGenStart = std::chrono::steady_clock::now();
            newData = trainer.generateTrainingData(config,
                [this, dataGenStart](int done, int total) {
                    nnueStatus_ = "Data gen: game " + std::to_string(done) + "/" + std::to_string(total);
                    updateETA(dataGenStart, done, total);
                }, &nnueCancelFlag_);
        }

        if (nnueCancelFlag_.load()) { nnueTraining_ = false; nnueETAEndMs_ = 0; nnueStatus_ = "Cancelled."; return; }

        // === Phase 3: Mirror new data ===
        std::vector<NNUE::TrainingPosition> mirroredData;
        if (config.mirrorPositions && !newData.empty()) {
            nnueStatus_ = "Mirroring positions...";
            mirroredData = NNUE::Trainer::mirrorData(newData);
        }

        // === Phase 4: Combine all data ===
        std::vector<NNUE::TrainingPosition> allData;
        size_t totalSize = existingData.size() + newData.size() + mirroredData.size();
        allData.reserve(totalSize);

        allData.insert(allData.end(),
            std::make_move_iterator(existingData.begin()),
            std::make_move_iterator(existingData.end()));
        allData.insert(allData.end(),
            std::make_move_iterator(newData.begin()),
            std::make_move_iterator(newData.end()));
        allData.insert(allData.end(),
            std::make_move_iterator(mirroredData.begin()),
            std::make_move_iterator(mirroredData.end()));

        if (allData.empty()) {
            nnueStatus_ = "No training data available!";
            nnueTraining_ = false;
            return;
        }

        // Compute phase distribution stats
        int nOpening = 0, nMiddle = 0, nEndgame = 0;
        for (const auto& pos : allData) {
            auto phase = NNUE::Trainer::classifyPhase(pos.activeFeatures);
            switch (phase) {
                case NNUE::Trainer::GamePhase::Opening:    ++nOpening; break;
                case NNUE::Trainer::GamePhase::Middlegame: ++nMiddle; break;
                case NNUE::Trainer::GamePhase::Endgame:    ++nEndgame; break;
            }
        }
        nnueStatus_ = "Training: " + std::to_string(allData.size()) + " pos (O:"
            + std::to_string(nOpening) + " M:" + std::to_string(nMiddle)
            + " E:" + std::to_string(nEndgame) + ")";

        // === Phase 5: Save combined data for future reuse ===
        if (!newData.empty()) {
            // Only save if we generated new data (to preserve existing data when skipping)
            trainer.saveTrainingData(allData, config.dataPath);
        }

        if (nnueCancelFlag_.load()) { nnueTraining_ = false; nnueETAEndMs_ = 0; nnueStatus_ = "Cancelled."; return; }

        // === Phase 6: Train ===
        auto trainStart = std::chrono::steady_clock::now();
        int totalEpochs = config.epochs;
        trainer.train(*nnueNet_, allData, config,
            [this, trainStart, totalEpochs](int epoch, float loss) {
                nnueStatus_ = "Epoch " + std::to_string(epoch) + "/" + std::to_string(totalEpochs)
                    + " loss: " + std::to_string(loss).substr(0, 8);
                updateETA(trainStart, epoch, totalEpochs);
            }, &nnueCancelFlag_);

        if (nnueCancelFlag_.load()) {
            nnueStatus_ = "Training cancelled at current epoch. Weights saved.";
        } else {
            nnueStatus_ = "Training complete! Press N to enable NNUE.";
        }
        nnueTraining_ = false;
        nnueETAEndMs_ = 0;
    });
    updateStatus();
}

// -------------------------------------------------------
// KEY PRESS HANDLER
// -------------------------------------------------------
void VisualGame::handleKeyPress(sf::Keyboard::Key key) {
    if (key == sf::Keyboard::Key::B) {
        // Toggle bot vs bot mode
        if (!botVsBot) {
            botVsBot = true;
            botPaused = false;
            pieceSelected = false;
            selectedMoves.clear();
            isDragging = false;
            placingDuck_ = false;

            // If it's not already thinking, start
            if (!engineThinking && !isAnimating && !gameOver) {
                startEngineThinking();
            }
        } else {
            botVsBot = false;
            botWaiting = false;
            botPaused = false;
        }
        updateStatus();
    }
    else if (key == sf::Keyboard::Key::Space) {
        // Pause/resume (only in bot vs bot mode)
        if (botVsBot) {
            botPaused = !botPaused;
            if (!botPaused && !engineThinking && !isAnimating && !botWaiting && !gameOver) {
                // Resume: start thinking if idle
                startEngineThinking();
            }
            updateStatus();
        }
    }
    else if (key == sf::Keyboard::Key::A) {
        // Toggle thought arrows
        showArrows = !showArrows;
    }
    else if (key == sf::Keyboard::Key::F) {
        // Toggle fast mode (bot-vs-bot only)
        fastMode = !fastMode;
        updateStatus();
    }
    else if (key == sf::Keyboard::Key::D) {
        // Toggle duck chess mode
        isDuckChess_ = !isDuckChess_;
        resetGame();
    }
    else if (key == sf::Keyboard::Key::R) {
        // Reset game
        resetGame();
    }
    else if (key == sf::Keyboard::Key::T) {
        if (nnueTraining_) {
            // Cancel training
            nnueCancelFlag_.store(true);
            nnueStatus_ = "Cancelling training...";
        }
        else if (nnueEstimating_) {
            // Cancel ELO estimation
            nnueCancelFlag_.store(true);
            nnueStatus_ = "Cancelling ELO estimation...";
        }
        else if (nnueInputMode_) {
            // Already in input mode, cancel it
            nnueInputMode_ = false;
            nnueStatus_ = "";
        }
        else {
            // Enter training config input mode
            nnueInputMode_ = true;
            nnueInputStep_ = 0;
            nnueInputBuffer_.clear();
            nnueStatus_ = "# games to generate (0=skip): _";
        }
    }
    else if (key == sf::Keyboard::Key::E) {
        if (nnueEstimating_) {
            // Cancel ELO estimation
            nnueCancelFlag_.store(true);
            nnueStatus_ = "Cancelling ELO estimation...";
        }
        else if (nnueTraining_) {
            // Cancel training first
            nnueCancelFlag_.store(true);
            nnueStatus_ = "Cancelling training...";
        }
        else if (nnueEloInputMode_) {
            // Cancel ELO input
            nnueEloInputMode_ = false;
            nnueInputBuffer_.clear();
            nnueStatus_ = "";
        }
        else {
            // Enter ELO config input mode
            nnueEloInputMode_ = true;
            nnueInputBuffer_.clear();
            nnueStatus_ = "# games for ELO test (0=default 100): _";
        }
    }
    else if (key == sf::Keyboard::Key::N) {
        // Toggle NNUE evaluation
        if (nnueNet_ && !nnueTraining_) {
            nnueEnabled_ = !nnueEnabled_;
            NNUE::Network* net = nnueEnabled_ ? nnueNet_.get() : nullptr;
            engine_.setNNUE(net);
            engine2_.setNNUE(net);
            nnueStatus_ = nnueEnabled_ ? "NNUE eval ON" : "NNUE eval OFF (handcrafted)";
        }
        else if (!nnueNet_) {
            // Try loading saved weights
            nnueNet_ = std::make_unique<NNUE::Network>();
            if (nnueNet_->loadWeights("assets/nnue_weights.bin")) {
                nnueEnabled_ = true;
                engine_.setNNUE(nnueNet_.get());
                engine2_.setNNUE(nnueNet_.get());
                nnueStatus_ = "NNUE loaded and enabled!";
            } else {
                nnueNet_.reset();
                nnueStatus_ = "No weights found. Train first (press T)!";
            }
        }
        updateStatus();
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
    legalMoves = MoveGen::getLegalMoves(board);
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
// MOUSE DOWN
// -------------------------------------------------------
void VisualGame::handleMouseDown(int x, int y) {
    // Duck placement phase: only accept clicks on empty squares
    if (placingDuck_) {
        Square sq = screenToSquare(x, y);
        if (!sq.isValid()) return;

        // Must be an empty square (no piece, no duck)
        Piece p = board.getPiece(sq);
        if (!p.isNone()) return;

        // Place the duck
        board.placeDuck(sq);
        placingDuck_ = false;

        // Record position hash
        positionHistory_.push_back(Engine::computeHash(board));

        // Generate legal moves for the next player
        legalMoves = MoveGen::getLegalMoves(board);
        updateStatus();

        // Check for game over
        if (!gameOver) {
            if (botVsBot) {
                startEngineThinking();
            } else if (engineEnabled && board.turn == engineColor) {
                startEngineThinking();
            }
        }
        return;
    }

    Square sq = screenToSquare(x, y);
    if (!sq.isValid()) {
        pieceSelected = false;
        selectedMoves.clear();
        isDragging = false;
        return;
    }

    if (pieceSelected && isLegalTarget(sq)) {
        bool isPromotion = false;
        for (auto& m : selectedMoves) {
            if (m.to.rank == sq.rank && m.to.col == sq.col && m.promotion != PieceType::None) {
                isPromotion = true;
                break;
            }
        }

        if (isPromotion) {
            isPromoting = true;
            promoFrom = selectedSq;
            promoTo   = sq;
            isDragging = false;
            return;
        }

        for (auto& m : selectedMoves) {
            if (m.to.rank == sq.rank && m.to.col == sq.col) {
                isDragging = false;
                executeMove(m);
                return;
            }
        }
    }

    Piece p = board.getPiece(sq);
    if (!p.isNone() && !p.isDuck() && p.color == board.turn) {
        selectPiece(sq);
        isDragging = true;
        dragFrom   = sq;
        dragPos    = {float(x), float(y)};
    }
    else {
        pieceSelected = false;
        selectedMoves.clear();
        isDragging = false;
    }
}

// -------------------------------------------------------
// MOUSE MOVE
// -------------------------------------------------------
void VisualGame::handleMouseMove(int x, int y) {
    if (isDragging) {
        dragPos = {float(x), float(y)};
    }
}

// -------------------------------------------------------
// MOUSE UP
// -------------------------------------------------------
void VisualGame::handleMouseUp(int x, int y) {
    // Duck placement via drag-release
    if (placingDuck_) {
        Square sq = screenToSquare(x, y);
        if (!sq.isValid()) return;

        Piece p = board.getPiece(sq);
        if (!p.isNone()) return;

        board.placeDuck(sq);
        placingDuck_ = false;

        positionHistory_.push_back(Engine::computeHash(board));

        legalMoves = MoveGen::getLegalMoves(board);
        updateStatus();

        if (!gameOver) {
            if (botVsBot) {
                startEngineThinking();
            } else if (engineEnabled && board.turn == engineColor) {
                startEngineThinking();
            }
        }
        return;
    }

    if (!isDragging) return;

    isDragging = false;
    Square dropSq = screenToSquare(x, y);

    if (dropSq.isValid() && dropSq.rank == dragFrom.rank && dropSq.col == dragFrom.col) {
        return;
    }

    if (dropSq.isValid() && isLegalTarget(dropSq)) {
        bool isPromotion = false;
        for (auto& m : selectedMoves) {
            if (m.to.rank == dropSq.rank && m.to.col == dropSq.col && m.promotion != PieceType::None) {
                isPromotion = true;
                break;
            }
        }

        if (isPromotion) {
            isPromoting = true;
            promoFrom = selectedSq;
            promoTo   = dropSq;
            return;
        }

        for (auto& m : selectedMoves) {
            if (m.to.rank == dropSq.rank && m.to.col == dropSq.col) {
                executeMove(m, false);
                return;
            }
        }
    }
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
            legalMoves = MoveGen::getLegalMoves(board);
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
            legalMoves = MoveGen::getLegalMoves(board);
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

    legalMoves = MoveGen::getLegalMoves(board);
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
// PROMOTION DIALOG
// -------------------------------------------------------
void VisualGame::handlePromotionClick(int x, int y) {
    int promoCol = promoTo.col;
    bool isWhite = (board.turn == Color::White);
    int startRank = isWhite ? 7 : 0;
    int dir = isWhite ? -1 : 1;

    PieceType choices[] = { PieceType::Queen, PieceType::Knight, PieceType::Rook, PieceType::Bishop };

    for (int i = 0; i < 4; i++) {
        int rank = startRank + dir * i;
        float px = float(OX + promoCol * SQ);
        float py = float(OY + (7 - rank) * SQ);

        if (x >= px && x < px + SQ && y >= py && y < py + SQ) {
            Move promoMove;
            promoMove.from = promoFrom;
            promoMove.to   = promoTo;
            promoMove.promotion = choices[i];

            isPromoting = false;
            executeMove(promoMove, false);
            return;
        }
    }

    isPromoting = false;
    pieceSelected = false;
    selectedMoves.clear();
}

// -------------------------------------------------------
// UPDATE STATUS
// -------------------------------------------------------
void VisualGame::updateStatus() {
    std::string side = (board.turn == Color::White) ? "White" : "Black";

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

// =======================================================
//  R E N D E R I N G
// =======================================================

void VisualGame::render() {
    window.clear(sf::Color(40, 40, 40));
    drawBoard();
    drawHighlights();
    drawPieces();
    if (showArrows && (engineThinking || botHasPendingMove_))
        drawPVArrows();
    drawEvalBar();
    drawCoordinates();
    drawStatus();
    drawHUD();

    if (isAnimating)
        drawAnimatingPiece();
    if (isDragging)
        drawDraggedPiece();
    if (isPromoting)
        drawPromotionDialog();

    window.display();
}

// -------------------------------------------------------
// DRAW BOARD
// -------------------------------------------------------
void VisualGame::drawBoard() {
    sf::Color light(238, 216, 192);
    sf::Color dark(171, 122, 101);

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            sf::RectangleShape cell({float(SQ), float(SQ)});
            cell.setPosition({float(OX + c * SQ), float(OY + (7 - r) * SQ)});
            cell.setFillColor(((r + c) % 2 == 0) ? dark : light);
            window.draw(cell);
        }
    }
}

// -------------------------------------------------------
// DRAW HIGHLIGHTS
// -------------------------------------------------------
void VisualGame::drawHighlights() {
    if (hasLastMove) {
        sf::Color olive(186, 202, 68, 128);
        for (const auto& s : { lastMove.from, lastMove.to }) {
            sf::RectangleShape h({float(SQ), float(SQ)});
            h.setPosition({float(OX + s.col * SQ), float(OY + (7 - s.rank) * SQ)});
            h.setFillColor(olive);
            window.draw(h);
        }
    }

    // Duck placement highlights: show all empty squares
    if (placingDuck_) {
        sf::Color duckHighlight(255, 220, 50, 60); // subtle yellow
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                Piece p = board.squares[r][c];
                if (p.isNone()) {
                    sf::RectangleShape h({float(SQ), float(SQ)});
                    h.setPosition({float(OX + c * SQ), float(OY + (7 - r) * SQ)});
                    h.setFillColor(duckHighlight);
                    window.draw(h);
                }
            }
        }
        return; // Don't draw piece selection highlights while placing duck
    }

    if (!pieceSelected) return;

    sf::RectangleShape sel({float(SQ), float(SQ)});
    sel.setPosition({float(OX + selectedSq.col * SQ),
                     float(OY + (7 - selectedSq.rank) * SQ)});
    sel.setFillColor(sf::Color(255, 255, 0, 100));
    window.draw(sel);

    for (const auto& m : selectedMoves) {
        float cx = OX + m.to.col * SQ + SQ / 2.f;
        float cy = OY + (7 - m.to.rank) * SQ + SQ / 2.f;

        bool isCapture = !board.getPiece(m.to).isNone();
        if (!isCapture && board.getPiece(m.from).type == PieceType::Pawn
            && m.to.col != m.from.col)
            isCapture = true;

        if (isCapture) {
            float r = SQ / 2.f;
            sf::CircleShape ring(r);
            ring.setOrigin({r, r});
            ring.setPosition({cx, cy});
            ring.setFillColor(sf::Color::Transparent);
            ring.setOutlineThickness(-5.f);
            ring.setOutlineColor(sf::Color(0, 0, 0, 80));
            window.draw(ring);
        }
        else {
            float r = SQ / 6.f;
            sf::CircleShape dot(r);
            dot.setOrigin({r, r});
            dot.setPosition({cx, cy});
            dot.setFillColor(sf::Color(0, 0, 0, 80));
            window.draw(dot);
        }
    }
}

// -------------------------------------------------------
// DRAW PV ARROWS — engine's live thought line
// -------------------------------------------------------
void VisualGame::drawPVArrows() {
    // Get live PV from the engine, or fall back to cached PV during pending-move delay
    std::vector<Move> pv;
    if (activeEngine_)
        pv = activeEngine_->getLivePV();
    else if (botHasPendingMove_)
        pv = cachedPV_;

    if (pv.empty()) return;

    // Draw up to 5 moves of the PV
    int maxArrows = std::min((int)pv.size(), 5);
    for (int i = 0; i < maxArrows; i++) {
        const Move& m = pv[i];
        if (!m.from.isValid() || !m.to.isValid()) break;

        sf::Vector2f from = squareCenter(m.from);
        sf::Vector2f to   = squareCenter(m.to);

        // First move is semi-transparent, subsequent ones fade more
        int alpha = 140 - i * 25;
        if (alpha < 30) alpha = 30;

        // Alternate colors: blue for the side to move, red for opponent
        sf::Color color;
        if (i % 2 == 0)
            color = sf::Color(66, 135, 245, alpha);   // blue
        else
            color = sf::Color(230, 70, 70, alpha);     // red

        drawArrow(from, to, color);
    }
}

// -------------------------------------------------------
// DRAW ARROW — thick line with arrowhead
// -------------------------------------------------------
void VisualGame::drawArrow(sf::Vector2f from, sf::Vector2f to, sf::Color color) {
    sf::Vector2f dir = to - from;
    float length = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    if (length < 1.f) return;

    sf::Vector2f unit = dir / length;
    sf::Vector2f perp = {-unit.y, unit.x};

    float shaftWidth = 8.f;
    float headLength = 20.f;
    float headWidth  = 18.f;

    // Shorten shaft so arrow head sits at the target
    sf::Vector2f shaftEnd = to - unit * headLength;

    // Shaft (quad as two triangles)
    sf::ConvexShape shaft(4);
    shaft.setPoint(0, from + perp * (shaftWidth / 2.f));
    shaft.setPoint(1, from - perp * (shaftWidth / 2.f));
    shaft.setPoint(2, shaftEnd - perp * (shaftWidth / 2.f));
    shaft.setPoint(3, shaftEnd + perp * (shaftWidth / 2.f));
    shaft.setFillColor(color);
    window.draw(shaft);

    // Arrowhead (triangle)
    sf::ConvexShape head(3);
    head.setPoint(0, to);
    head.setPoint(1, shaftEnd + perp * headWidth);
    head.setPoint(2, shaftEnd - perp * headWidth);
    head.setFillColor(color);
    window.draw(head);
}

// -------------------------------------------------------
// DRAW PIECES
// -------------------------------------------------------
void VisualGame::drawPieces() {
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (isDragging && dragFrom.rank == r && dragFrom.col == c)
                continue;
            if (isAnimating && animMove.from.rank == r && animMove.from.col == c)
                continue;

            Piece p = board.squares[r][c];
            if (p.isNone()) continue;

            float px = float(OX + c * SQ);
            float py = float(OY + (7 - r) * SQ);

            // Duck piece uses special texture
            if (p.isDuck()) {
                sf::Sprite sprite(duckTexture_);
                sf::Vector2u ts = duckTexture_.getSize();
                float scale = float(SQ) / float(ts.x);
                sprite.setScale({scale, scale});
                sprite.setPosition({px, py});
                window.draw(sprite);
                continue;
            }

            int ci = (p.color == Color::White) ? 0 : 1;
            int pi = static_cast<int>(p.type) - 1;

            sf::Sprite sprite(pieceTextures[ci][pi]);
            sf::Vector2u ts = pieceTextures[ci][pi].getSize();
            float scale = float(SQ) / float(ts.x);
            sprite.setScale({scale, scale});
            sprite.setPosition({px, py});
            window.draw(sprite);
        }
    }
}

// -------------------------------------------------------
// DRAW DRAGGED PIECE
// -------------------------------------------------------
void VisualGame::drawDraggedPiece() {
    Piece p = board.getPiece(dragFrom);
    if (p.isNone()) return;

    int ci = (p.color == Color::White) ? 0 : 1;
    int pi = static_cast<int>(p.type) - 1;

    sf::Sprite sprite(pieceTextures[ci][pi]);
    sf::Vector2u ts = pieceTextures[ci][pi].getSize();
    float scale = float(SQ) / float(ts.x);
    sprite.setScale({scale, scale});
    sprite.setPosition({dragPos.x - SQ / 2.f, dragPos.y - SQ / 2.f});
    window.draw(sprite);
}

// -------------------------------------------------------
// DRAW ANIMATING PIECE
// -------------------------------------------------------
void VisualGame::drawAnimatingPiece() {
    float t = std::min(animElapsed / ANIM_DURATION, 1.f);
    t = 1.f - (1.f - t) * (1.f - t);

    float x = animStartPos.x + (animEndPos.x - animStartPos.x) * t;
    float y = animStartPos.y + (animEndPos.y - animStartPos.y) * t;

    int ci = (animPiece.color == Color::White) ? 0 : 1;
    int pi = static_cast<int>(animPiece.type) - 1;

    sf::Sprite sprite(pieceTextures[ci][pi]);
    sf::Vector2u ts = pieceTextures[ci][pi].getSize();
    float scale = float(SQ) / float(ts.x);
    sprite.setScale({scale, scale});
    sprite.setPosition({x, y});
    window.draw(sprite);
}

// -------------------------------------------------------
// DRAW PROMOTION DIALOG
// -------------------------------------------------------
void VisualGame::drawPromotionDialog() {
    sf::RectangleShape overlay({float(SQ * 8), float(SQ * 8)});
    overlay.setPosition({float(OX), float(OY)});
    overlay.setFillColor(sf::Color(0, 0, 0, 120));
    window.draw(overlay);

    bool isWhite = (board.turn == Color::White);
    int ci = isWhite ? 0 : 1;
    int startRank = isWhite ? 7 : 0;
    int dir = isWhite ? -1 : 1;

    PieceType choices[] = { PieceType::Queen, PieceType::Knight, PieceType::Rook, PieceType::Bishop };

    for (int i = 0; i < 4; i++) {
        int rank = startRank + dir * i;
        float px = float(OX + promoTo.col * SQ);
        float py = float(OY + (7 - rank) * SQ);

        sf::RectangleShape cell({float(SQ), float(SQ)});
        cell.setPosition({px, py});
        cell.setFillColor(sf::Color(240, 240, 230));
        cell.setOutlineThickness(1.f);
        cell.setOutlineColor(sf::Color(100, 100, 100));
        window.draw(cell);

        int pi = static_cast<int>(choices[i]) - 1;
        sf::Sprite sprite(pieceTextures[ci][pi]);
        sf::Vector2u ts = pieceTextures[ci][pi].getSize();
        float scale = float(SQ) / float(ts.x);
        sprite.setScale({scale, scale});
        sprite.setPosition({px, py});
        window.draw(sprite);
    }
}

// -------------------------------------------------------
// DRAW COORDINATES
// -------------------------------------------------------
void VisualGame::drawEvalBar() {
    // Eval bar sits to the left of the board
    float barX = float(OX - EVAL_BAR_W - 28);
    float barY = float(OY);
    float barH = float(SQ * 8);
    float barW = float(EVAL_BAR_W);

    // Get live eval if engine is thinking, else use last stored eval
    // Engine already stores liveEval_ from White's perspective — no negation needed
    // Only use live value once the engine has completed at least depth 1,
    // otherwise liveEval_ is 0 (reset at search start) causing a brief flash
    int eval = lastEval_;
    if (activeEngine_ && activeEngine_->getLiveDepth() > 0) {
        eval = activeEngine_->getLiveEval();
    }

    // Clamp eval to a display range: sigmoid-like mapping for smooth bar
    // Convert centipawns to a 0-1 ratio (0.5 = even, 1.0 = white winning)
    float ratio = 0.5f + 0.5f * (eval / 500.0f);   // linear scale, ±500cp = full bar
    if (ratio > 0.98f) ratio = 0.98f;
    if (ratio < 0.02f) ratio = 0.02f;

    // Background (black side - top)
    sf::RectangleShape blackSide({barW, barH});
    blackSide.setPosition({barX, barY});
    blackSide.setFillColor(sf::Color(50, 50, 50));
    window.draw(blackSide);

    // White side (bottom, grows upward)
    float whiteH = barH * ratio;
    sf::RectangleShape whiteSide({barW, whiteH});
    whiteSide.setPosition({barX, barY + barH - whiteH});
    whiteSide.setFillColor(sf::Color(235, 235, 235));
    window.draw(whiteSide);

    // Outline
    sf::RectangleShape outline({barW, barH});
    outline.setPosition({barX, barY});
    outline.setFillColor(sf::Color::Transparent);
    outline.setOutlineThickness(1.f);
    outline.setOutlineColor(sf::Color(100, 100, 100));
    window.draw(outline);

    // Eval number label — larger, centered, with background pill for readability
    float absEval = std::abs(eval / 100.0f);
    std::string evalStr;
    if (std::abs(eval) > 9000)
        evalStr = (eval > 0) ? "M" : "-M";
    else {
        std::ostringstream oss;
        oss << std::fixed;
        oss.precision(1);
        if (eval >= 0) oss << "+" << absEval;
        else oss << "-" << absEval;
        evalStr = oss.str();
    }

    sf::Text evalText(font, evalStr, 16);
    auto textBounds = evalText.getLocalBounds();
    float textW = textBounds.size.x;
    float textH = textBounds.size.y;

    // Center text horizontally in bar, place in the smaller zone for contrast
    float textX = barX + (barW - textW) / 2.f;
    float textY;
    sf::Color textColor;
    sf::Color pillColor;

    if (ratio >= 0.5f) {
        // White winning — put number near top in black zone
        textY = barY + 6;
        textColor = sf::Color(235, 235, 235);
        pillColor = sf::Color(30, 30, 30, 160);
    } else {
        // Black winning — put number near bottom in white zone
        textY = barY + barH - textH - 14;
        textColor = sf::Color(30, 30, 30);
        pillColor = sf::Color(235, 235, 235, 160);
    }

    // Draw background pill behind text for readability
    float pillPad = 3.f;
    sf::RectangleShape pill({textW + pillPad * 2, textH + pillPad * 2 + 4});
    pill.setPosition({textX - pillPad, textY - pillPad + 2});
    pill.setFillColor(pillColor);
    window.draw(pill);

    evalText.setFillColor(textColor);
    evalText.setPosition({textX, textY});
    window.draw(evalText);
}

void VisualGame::drawCoordinates() {
    for (int i = 0; i < 8; i++) {
        sf::Text rank(font, std::string(1, char('1' + i)), 14);
        rank.setFillColor(sf::Color(200, 200, 200));
        rank.setPosition({float(OX - 20), float(OY + (7 - i) * SQ + SQ / 2 - 7)});
        window.draw(rank);

        sf::Text file(font, std::string(1, char('a' + i)), 14);
        file.setFillColor(sf::Color(200, 200, 200));
        file.setPosition({float(OX + i * SQ + SQ / 2 - 4), float(OY + 8 * SQ + 5)});
        window.draw(file);
    }
}

// -------------------------------------------------------
// DRAW STATUS BAR
// -------------------------------------------------------
void VisualGame::drawStatus() {
    if (engineThinking && !engineDone.load()) {
        std::string side = (board.turn == Color::White) ? "White" : "Black";
        int liveDepth = 0;
        if (activeEngine_)
            liveDepth = activeEngine_->getLiveDepth();
        if (liveDepth > 0)
            statusMsg = side + " thinking... depth " + std::to_string(liveDepth);
        else
            statusMsg = side + " is thinking...";
    }

    // Move counter prefix
    std::string display = "Move " + std::to_string(moveNumber) + "  |  " + statusMsg;

    sf::Text status(font, display, 20);
    status.setFillColor(sf::Color::White);
    status.setPosition({float(OX), float(OY + 8 * SQ + 28)});
    window.draw(status);
}

// -------------------------------------------------------
// DRAW HUD — controls info at the top
// -------------------------------------------------------
void VisualGame::drawHUD() {
    std::string hudText;
    if (botVsBot) {
        hudText = "BOT vs BOT";
        if (isDuckChess_) hudText += "  [DUCK]";
        if (botPaused) hudText += "  [PAUSED]";
        if (fastMode) hudText += "  [FAST]";
        if (nnueEnabled_) hudText += "  [NNUE]";
        hudText += "  |  [Space] Pause  [A] Arrows  [F] Fast  [D] Duck  [R] Reset  [B] Exit";
    } else {
        hudText = "PLAYER vs ENGINE";
        if (isDuckChess_) hudText += "  [DUCK]";
        if (nnueEnabled_) hudText += "  [NNUE]";
        hudText += "  |  [B] Bot  [A] Arrows  [D] Duck  [R] Reset";
    }
    if (nnueTraining_)
        hudText += "  [T] Cancel Train";
    else if (nnueEstimating_)
        hudText += "  [E] Cancel ELO";
    else if (nnueInputMode_)
        hudText += "  [Esc] Cancel Input";
    else if (nnueEloInputMode_)
        hudText += "  [Esc] Cancel ELO Input";
    else
        hudText += "  [T] Train  [E] ELO  [N] NNUE";

    sf::Text hud(font, hudText, 13);
    hud.setFillColor(sf::Color(180, 180, 180));
    hud.setPosition({float(OX), float(OY - 25)});
    window.draw(hud);

    // Duck chess mode indicator
    if (isDuckChess_) {
        sf::Text duckLabel(font, "DUCK CHESS", 14);
        duckLabel.setFillColor(sf::Color(255, 220, 50));
        // Position it at the right side of the HUD line
        duckLabel.setPosition({float(OX + SQ * 8 - 90), float(OY - 25)});
        window.draw(duckLabel);
    }

    // NNUE status line (below board, above status)
    if (!nnueStatus_.empty()) {
        std::string displayStatus = nnueStatus_;
        auto etaEnd = nnueETAEndMs_.load();
        if (etaEnd > 0) {
            displayStatus += formatCountdown(etaEnd);
        }
        sf::Text nnueTxt(font, displayStatus, 14);
        nnueTxt.setFillColor(sf::Color(100, 200, 255));
        nnueTxt.setPosition({float(OX), float(OY + 8 * SQ + 50)});
        window.draw(nnueTxt);
    }
}

// =======================================================
//  H E L P E R S
// =======================================================

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
