// ARCHITECTURE FIX M-2: Split from VisualGame.cpp — input handling, text input,
// training orchestration, and ELO estimation logic.
//
#include "VisualGame.h"
#include "AssetPath.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <random>
#include <chrono>

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
        { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = ""; }
    }
    else if (unicode >= '0' && unicode <= '9') {
        nnueInputBuffer_ += static_cast<char>(unicode);
    }

    // Update display
    {
        std::lock_guard<std::mutex> lk(nnueStatusMutex_);
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
}

// -------------------------------------------------------
// PROCESS TRAINING INPUT
// -------------------------------------------------------
void VisualGame::processTrainingInput() {
    int value = 0;
    if (!nnueInputBuffer_.empty()) {
        try { value = std::stoi(nnueInputBuffer_); }
        catch (const std::exception&) { value = 0; }
    }
    nnueInputBuffer_.clear();

    if (nnueInputStep_ == 0) {
        nnueConfigGames_ = value;
        nnueInputStep_ = 1;
        { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Max positions (0=all): _"; }
    }
    else if (nnueInputStep_ == 1) {
        nnueConfigMaxPositions_ = value;
        nnueInputStep_ = 2;
        { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "# epochs (0=default 50): _"; }
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
// nnueETAEndMs_ is atomic; reads from render thread are safe without additional locking.
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

// -------------------------------------------------------
// PROCESS ELO INPUT
// -------------------------------------------------------
void VisualGame::processEloInput() {
    int value = 0;
    if (!nnueInputBuffer_.empty()) {
        try { value = std::stoi(nnueInputBuffer_); }
        catch (const std::exception&) { value = 0; }
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
        if (tempNet->loadWeights(assetPath("assets/nnue_weights.bin"))) {
            nnueNet_ = std::move(tempNet);
        } else {
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Train first (press T)!"; }
            return;
        }
    }

    if (nnueTraining_ || nnueEstimating_) return;

    nnueEstimating_ = true;
    nnueCancelFlag_.store(false);
    { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Estimating ELO (" + std::to_string(nnueConfigEloGames_) + " games)..."; }

    // Stop bot-vs-bot if active
    if (botVsBot || botVsNNUE_) {
        botVsBot = false;
        botVsNNUE_ = false;
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
                { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "ELO test: game " + std::to_string(done) + "/" + std::to_string(total); }
                updateETA(eloStart, done, total);
            }, &nnueCancelFlag_);

        if (nnueCancelFlag_.load()) {
            std::lock_guard<std::mutex> lk(nnueStatusMutex_);
            nnueStatus_ = "ELO estimation cancelled. Partial: " +
                std::to_string(lastEloResult_.estimatedElo) +
                " (W:" + std::to_string(lastEloResult_.wins) +
                " D:" + std::to_string(lastEloResult_.draws) +
                " L:" + std::to_string(lastEloResult_.losses) + ")";
        } else {
            hasEloResult_ = true;
            std::lock_guard<std::mutex> lk(nnueStatusMutex_);
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
        { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Skipping data gen, loading existing positions..."; }
    else
        { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Generating training data..."; }

    // Stop bot-vs-bot if active
    if (botVsBot || botVsNNUE_) {
        botVsBot = false;
        botVsNNUE_ = false;
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
        config.outputPath = assetPath("assets/nnue_weights.bin");
        config.dataPath = assetPath("assets/training_data.bin");
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
                { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Loaded " + std::to_string(existingData.size()) + " existing positions."; }
            }
        }

        if (nnueCancelFlag_.load()) { nnueTraining_ = false; nnueETAEndMs_ = 0; { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Cancelled."; } return; }

        // === Phase 1.5: Limit existing positions if requested ===
        if (nnueConfigMaxPositions_ > 0 && static_cast<int>(existingData.size()) > nnueConfigMaxPositions_) {
            std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::shuffle(existingData.begin(), existingData.end(), rng);
            existingData.resize(nnueConfigMaxPositions_);
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Using " + std::to_string(nnueConfigMaxPositions_) + " random existing positions."; }
        }

        // === Phase 2: Generate new data (skip if numGames == 0) ===
        std::vector<NNUE::TrainingPosition> newData;
        if (config.numGames > 0) {
            auto dataGenStart = std::chrono::steady_clock::now();
            newData = trainer.generateTrainingData(config,
                [this, dataGenStart](int done, int total) {
                    { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Data gen: game " + std::to_string(done) + "/" + std::to_string(total); }
                    updateETA(dataGenStart, done, total);
                }, &nnueCancelFlag_);
        }

        if (nnueCancelFlag_.load()) { nnueTraining_ = false; nnueETAEndMs_ = 0; { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Cancelled."; } return; }

        // === Phase 3: Mirror new data ===
        std::vector<NNUE::TrainingPosition> mirroredData;
        if (config.mirrorPositions && !newData.empty()) {
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Mirroring positions..."; }
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
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "No training data available!"; }
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
        { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Training: " + std::to_string(allData.size()) + " pos (O:"
            + std::to_string(nOpening) + " M:" + std::to_string(nMiddle)
            + " E:" + std::to_string(nEndgame) + ")"; }

        // === Phase 5: Save combined data for future reuse ===
        if (!newData.empty()) {
            // Only save if we generated new data (to preserve existing data when skipping)
            trainer.saveTrainingData(allData, config.dataPath);
        }

        if (nnueCancelFlag_.load()) { nnueTraining_ = false; nnueETAEndMs_ = 0; { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Cancelled."; } return; }

        // === Phase 6: Train ===
        auto trainStart = std::chrono::steady_clock::now();
        int totalEpochs = config.epochs;
        trainer.train(*nnueNet_, allData, config,
            [this, trainStart, totalEpochs](int epoch, float loss) {
                { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Epoch " + std::to_string(epoch) + "/" + std::to_string(totalEpochs)
                    + " loss: " + std::to_string(loss).substr(0, 8); }
                updateETA(trainStart, epoch, totalEpochs);
            }, &nnueCancelFlag_);

        if (nnueCancelFlag_.load()) {
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Training cancelled at current epoch. Weights saved."; }
        } else {
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Training complete! Press N to enable NNUE."; }
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
    // ── Analysis navigation: ← back, → forward, Escape exits analysis mode ──
    if (key == sf::Keyboard::Key::Left) {
        navigateHistory(-1);
        return;
    }
    if (key == sf::Keyboard::Key::Right) {
        navigateHistory(+1);
        return;
    }
    if (key == sf::Keyboard::Key::Escape && analysisMode_) {
        exitAnalysisMode();
        return;
    }

    if (key == sf::Keyboard::Key::B) {
        // Toggle bot vs bot mode
        if (!botVsBot) {
            botVsNNUE_ = false;  // cancel mixed mode if active
            botVsBot = true;
            botPaused = false;
            pieceSelected = false;
            selectedMoves.clear();
            isDragging = false;
#ifdef DUCK_CHESS
            placingDuck_ = false;
#endif

            // If it's not already thinking, start
            if (!engineThinking && !isAnimating && !gameOver) {
                startEngineThinking();
            }
        } else {
            botVsBot = false;
            engine_.stop();
            engine2_.stop();
            if (engineThread.joinable()) engineThread.join();
            engineThinking = false;
            botHasPendingMove_ = false;
            cachedPV_.clear();
            botPaused = false;
            // If engine plays White and it's White's turn, start thinking
            if (engineEnabled && board.turn == engineColor && !gameOver) {
                startEngineThinking();
            }
        }
        updateStatus();
    }
    else if (key == sf::Keyboard::Key::Space) {
        // Pause/resume (only in bot vs bot mode)
        if (botVsBot || botVsNNUE_) {
            botPaused = !botPaused;
            if (botPaused) {
                // Stop the engine on pause to save CPU
                if (engineThinking && activeEngine_)
                    activeEngine_->stop();
            }
            if (!botPaused) {
                // Unpausing — if in analysis mode, jump to latest move first
                if (analysisMode_)
                    exitAnalysisMode();  // selects latest move, stops analysis engine
                if (!engineThinking && !isAnimating && !botHasPendingMove_ && !gameOver)
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
#ifdef DUCK_CHESS
    else if (key == sf::Keyboard::Key::D) {
        // Toggle duck chess mode
        isDuckChess_ = !isDuckChess_;
        isAutomateChess_ = false;
        // Clear loaded networks — variant changed, weights are no longer valid
        nnueNet_.reset();
        duckNnueNet_.reset();
        nnueEnabled_ = false;
        engine_.setNNUE(nullptr);
        engine2_.setNNUE(nullptr);
        engine_.setDuckNNUE(nullptr);
        engine2_.setDuckNNUE(nullptr);
        // Auto-load the correct weights for the new variant
        {
            std::string wPath = isDuckChess_
                ? assetPath("assets/duck_nnue_weights.bin")
                : assetPath("assets/nnue_weights.bin");
            if (isDuckChess_) {
                duckNnueNet_ = std::make_unique<DuckNNUE::Network>();
                if (duckNnueNet_->loadWeights(wPath)) {
                    engine_.setDuckNNUE(duckNnueNet_.get());
                    engine2_.setDuckNNUE(duckNnueNet_.get());
                    { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Duck NNUE loaded (press N to enable eval)"; }
                } else {
                    duckNnueNet_.reset();
                    { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "No duck weights at: " + wPath; }
                }
            } else {
                nnueNet_ = std::make_unique<NNUE::Network>();
                if (nnueNet_->loadWeights(wPath)) {
                    { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "NNUE loaded (press N to enable eval)"; }
                } else {
                    nnueNet_.reset();
                    { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "No weights at: " + wPath; }
                }
            }
        }
        resetGame();
    }
#endif
    else if (key == sf::Keyboard::Key::M) {
        // Toggle Automate Chess mode (M for "Muster your army")
        isAutomateChess_ = !isAutomateChess_;
        if (isDuckChess_) {
            // Switching away from duck chess — clear duck networks
            duckNnueNet_.reset();
            engine_.setDuckNNUE(nullptr);
            engine2_.setDuckNNUE(nullptr);
        }
        isDuckChess_ = false;
        if (isAutomateChess_)
            enterAutomateSetup();
        else
            resetGame();
    }
    else if (key == sf::Keyboard::Key::R) {
        // Reset game
        resetGame();
    }
    else if (key == sf::Keyboard::Key::T) {
        if (nnueTraining_) {
            // Cancel training
                if (engineThread.joinable()) engineThread.join();
                engineThinking = false;
            nnueCancelFlag_.store(true);
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Cancelling training..."; }
        }
        else if (nnueEstimating_) {
            // Cancel ELO estimation
                if (engineThread.joinable()) engineThread.join();
                engineThinking = false;
            nnueCancelFlag_.store(true);
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Cancelling ELO estimation..."; }
        }
        else if (nnueInputMode_) {
            // Already in input mode, cancel it
            nnueInputMode_ = false;
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = ""; }
        }
        else {
            // Enter training config input mode
            nnueInputMode_ = true;
            nnueInputStep_ = 0;
            nnueInputBuffer_.clear();
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "# games to generate (0=skip): _"; }
        }
    }
    else if (key == sf::Keyboard::Key::E) {
        if (nnueEstimating_) {
            // Cancel ELO estimation
                if (engineThread.joinable()) engineThread.join();
                engineThinking = false;
            nnueCancelFlag_.store(true);
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Cancelling ELO estimation..."; }
        }
        else if (nnueTraining_) {
            // Cancel training first
                if (engineThread.joinable()) engineThread.join();
                engineThinking = false;
            nnueCancelFlag_.store(true);
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Cancelling training..."; }
        }
        else if (nnueEloInputMode_) {
            // Cancel ELO input
            nnueEloInputMode_ = false;
            nnueInputBuffer_.clear();
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = ""; }
        }
        else {
            // Enter ELO config input mode
            nnueEloInputMode_ = true;
            nnueInputBuffer_.clear();
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "# games for ELO test (0=default 100): _"; }
        }
    }
    else if (key == sf::Keyboard::Key::N) {
        // Toggle NNUE evaluation
        if (nnueNet_ && !nnueTraining_) {
            nnueEnabled_ = !nnueEnabled_;
            NNUE::Network* net = nnueEnabled_ ? nnueNet_.get() : nullptr;
            engine_.setNNUE(net);
            engine2_.setNNUE(net);
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = nnueEnabled_ ? "NNUE eval ON" : "NNUE eval OFF (handcrafted)"; }
        }
        else if (!nnueNet_) {
            // Try loading saved weights — use duck weights if in duck chess mode
            nnueNet_ = std::make_unique<NNUE::Network>();
            std::string weightsFile = isDuckChess_
                ? assetPath("assets/duck_nnue_weights.bin")
                : assetPath("assets/nnue_weights.bin");
            if (nnueNet_->loadWeights(weightsFile)) {
                nnueEnabled_ = true;
                engine_.setNNUE(nnueNet_.get());
                engine2_.setNNUE(nnueNet_.get());
                // For duck chess, also wire up the DuckNNUE network
                if (isDuckChess_) {
                    if (!duckNnueNet_) duckNnueNet_ = std::make_unique<DuckNNUE::Network>();
                    if (duckNnueNet_->loadWeights(weightsFile)) {
                        engine_.setDuckNNUE(duckNnueNet_.get());
                        engine2_.setDuckNNUE(duckNnueNet_.get());
                    }
                }
                { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "NNUE loaded and enabled!"; }
            } else {
                nnueNet_.reset();
                { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "No weights at: " + weightsFile; }
            }
        }
        updateStatus();
    }
    else if (key == sf::Keyboard::Key::V) {
        // Toggle Bot vs NNUE mode (White=classical, Black=NNUE)
        if (!botVsNNUE_) {
            // Ensure NNUE weights are loaded — use duck weights if in duck chess mode
            if (!nnueNet_) {
                nnueNet_ = std::make_unique<NNUE::Network>();
                std::string weightsPath = isDuckChess_
                    ? assetPath("assets/duck_nnue_weights.bin")
                    : assetPath("assets/nnue_weights.bin");
                if (!nnueNet_->loadWeights(weightsPath)) {
                    nnueNet_.reset();
                    { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "No NNUE weights found at: " + weightsPath; }
                    return;
                }
                if (isDuckChess_) {
                    if (!duckNnueNet_) duckNnueNet_ = std::make_unique<DuckNNUE::Network>();
                    duckNnueNet_->loadWeights(weightsPath);
                    engine_.setDuckNNUE(duckNnueNet_.get());
                    engine2_.setDuckNNUE(duckNnueNet_.get());
                }
            }
            // Cancel any active bot-vs-bot first
            botVsBot = false;
            botVsNNUE_ = true;
            botPaused = false;
            pieceSelected = false;
            selectedMoves.clear();
            isDragging = false;
            // Apply side selection
            resolveSides();
            if (nnueOnWhite_)
                { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Bot vs NNUE: White=NNUE  Black=Classical"; }
            else
                { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Bot vs NNUE: White=Classical  Black=NNUE"; }
            if (!engineThinking && !isAnimating && !gameOver) {
                startEngineThinking();
            }
        } else {
            // Exit Bot vs NNUE mode
            botVsNNUE_ = false;
            engine_.stop();
            engine2_.stop();
            if (engineThread.joinable()) engineThread.join();
            engineThinking = false;
            botHasPendingMove_ = false;
            cachedPV_.clear();
            botPaused = false;
            // Restore both engines to whatever nnueEnabled_ says
            NNUE::Network* net = nnueEnabled_ ? nnueNet_.get() : nullptr;
            engine_.setNNUE(net);
            engine2_.setNNUE(net);
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Bot vs NNUE mode off"; }
            // If engine plays White and it's White's turn, start thinking
            if (engineEnabled && board.turn == engineColor && !gameOver) {
                startEngineThinking();
            }
        }
        updateStatus();
    }
    else if (key == sf::Keyboard::Key::S) {
        // Cycle side configuration: Default -> Swapped -> Random -> Default
        if (sideConfig_ == SideConfig::Default)
            sideConfig_ = SideConfig::Swapped;
        else if (sideConfig_ == SideConfig::Swapped)
            sideConfig_ = SideConfig::Random;
        else
            sideConfig_ = SideConfig::Default;

        // Apply immediately
        resolveSides();

        // Build status message
        std::string cfgName;
        if (sideConfig_ == SideConfig::Default)  cfgName = "Default";
        else if (sideConfig_ == SideConfig::Swapped) cfgName = "Swapped";
        else cfgName = "Random";

        if (botVsNNUE_) {
            std::string w = nnueOnWhite_ ? "NNUE" : "Classical";
            std::string b = nnueOnWhite_ ? "Classical" : "NNUE";
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Sides: " + cfgName + " => White=" + w + "  Black=" + b; }
        } else if (!botVsBot) {
            std::string ec = (engineColor == Color::White) ? "White" : "Black";
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Sides: " + cfgName + " => Engine plays " + ec; }
        } else {
            { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Sides: " + cfgName + " (no effect in Bot vs Bot)"; }
        }
        updateStatus();
    }
    else if (key == sf::Keyboard::Key::Z) {
        // Z: toggle analysis engine on the currently viewed position
        // Works in analysis mode OR on the live position
        bool canAnalyse = analysisMode_ || !gameHistory_.empty();
        if (canAnalyse) {
            if (analysisThread_.joinable()) {
                // Analysis running — stop it
                analysisEngine_.stop();
                analysisThread_.join();
                analysisDepth_ = 0;
                analysisNodes_ = 0;
                { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Analysis stopped"; }
            } else {
                // If not in analysis mode, enter it at the last move
                if (!analysisMode_ && !gameHistory_.empty()) {
                    viewIdx_      = int(gameHistory_.size()) - 1;
                    analysisMode_ = true;
                    Board tmp = gameHistory_[viewIdx_].board;
                    MoveList ml; MoveGen::getLegalMoves(tmp, ml);
                    legalMoves.assign(ml.begin(), ml.end());
                    // Pause bot so it doesn't interfere
                    if ((botVsBot || botVsNNUE_) && !botPaused) {
                        botPaused = true;
                        if (engineThinking && activeEngine_) activeEngine_->stop();
                    }
                }
                startAnalysisEngine();
                { std::lock_guard<std::mutex> lk(nnueStatusMutex_); nnueStatus_ = "Analysis started"; }
            }
        }
    }
}

// -------------------------------------------------------
// MOUSE DOWN
// -------------------------------------------------------
void VisualGame::handleMouseDown(int x, int y) {
    // Automate Chess setup phase
    if (isAutomateChess_ && !board.automateSetupComplete) {
        handleAutomateSetupClick(x, y);
        return;
    }

#ifdef DUCK_CHESS
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
        { legalMoves.clear(); MoveList ml_; MoveGen::getLegalMoves(board, ml_); legalMoves.assign(ml_.begin(), ml_.end()); }
        updateStatus();

        // Check for game over
        if (!gameOver) {
            if (botVsBot || botVsNNUE_) {
                startEngineThinking();
            } else if (engineEnabled && board.turn == engineColor) {
                startEngineThinking();
            }
        }
        return;
    }

#endif

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
#ifdef DUCK_CHESS
    // Duck placement via drag-release
    if (placingDuck_) {
        Square sq = screenToSquare(x, y);
        if (!sq.isValid()) return;

        Piece p = board.getPiece(sq);
        if (!p.isNone()) return;

        board.placeDuck(sq);
        placingDuck_ = false;

        positionHistory_.push_back(Engine::computeHash(board));

        { legalMoves.clear(); MoveList ml_; MoveGen::getLegalMoves(board, ml_); legalMoves.assign(ml_.begin(), ml_.end()); }
        updateStatus();

        if (!gameOver) {
            if (botVsBot || botVsNNUE_) {
                startEngineThinking();
            } else if (engineEnabled && board.turn == engineColor) {
                startEngineThinking();
            }
        }
        return;
    }

#endif

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
// PROMOTION DIALOG CLICK
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
