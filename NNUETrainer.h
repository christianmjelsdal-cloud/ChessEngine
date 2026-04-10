#pragma once
#include "NNUE.h"
#include "DuckNNUE.h"
#include "Engine.h"
#include "Board.h"
#include "MoveGen.h"
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <thread>
#include <atomic>

namespace NNUE {

    struct TrainingPosition {
        std::vector<int> activeFeatures;  // active feature indices for this position
        Color sideToMove = Color::White;
        float gameResult = 0.5f;     // 1.0 = white win, 0.5 = draw, 0.0 = black win
        float searchEval = 0.0f;     // evaluation from the handcrafted engine (centipawns, white's POV)
    };

    struct TrainingConfig {
        int numGames = 2000;          // games for data generation (INCREASED: 200 -> 2000)
        int thinkTimeMs = 50;         // engine think time per move during data gen (LOWERED: 100 -> 50 for 2x speed)
        int epochs = 100;             // training epochs
        int batchSize = 512;          // mini-batch size (increased from 256)
        float learningRate = 0.001f;  // initial LR
        float lrDecay = 0.995f;       // LR decay per epoch
        float lambda = 0.5f;          // weight between eval loss and result loss
        float evalScale = 400.0f;     // sigmoid scaling factor
        std::string outputPath = "assets/nnue_weights.bin";  // where to save trained weights

        // === Speed improvements ===
        int numThreads = 0;            // 0 = auto-detect (hardware_concurrency)
        int earlyStopPatience = 15;    // stop if no improvement for this many epochs (0 = disabled)
        bool mirrorPositions = true;   // double data via horizontal mirroring
        bool appendExistingData = true; // load and append to existing training_data.bin
        std::string dataPath = "assets/training_data.bin"; // path for saved training data
        bool phaseBalancedTraining = true;  // balance training across game phases

        // === NEW: Diversified openings ===
        int randomOpeningMoves = 8;    // number of random legal moves to play at game start
                                       // diversifies starting positions to avoid repetitive games
                                       // positions from these random moves are NOT recorded as training data
        bool useOpeningBook = false;   // future: use an opening book instead of random moves

        // === Duck Chess variant ===
        bool isDuckChess = false;          // generate duck chess games instead of standard
    };

    struct EloResult {
        int wins = 0, draws = 0, losses = 0;
        double winRate = 0.0;
        int eloDifference = 0;
        int estimatedElo = 0;  // if baseline is known
    };

    class Trainer {
    public:
        Trainer();

        // Data generation via self-play (now supports parallel threads)
        std::vector<TrainingPosition> generateTrainingData(const TrainingConfig& config,
            std::function<void(int gamesPlayed, int totalGames)> progressCallback = nullptr,
            std::atomic<bool>* cancelFlag = nullptr);

        // Training loop (now with early stopping) — standard 768-feature NNUE
        void train(Network& net, const std::vector<TrainingPosition>& data,
            const TrainingConfig& config,
            std::function<void(int epoch, float loss)> progressCallback = nullptr,
            std::atomic<bool>* cancelFlag = nullptr);

        // Training loop for duck chess — 832-feature DuckNNUE
        void trainDuck(DuckNNUE::Network& net, const std::vector<TrainingPosition>& data,
            const TrainingConfig& config,
            std::function<void(int epoch, float loss)> progressCallback = nullptr,
            std::atomic<bool>* cancelFlag = nullptr);

        // Save/load training data
        void saveTrainingData(const std::vector<TrainingPosition>& data, const std::string& filename);
        std::vector<TrainingPosition> loadTrainingData(const std::string& filename);

        // Mirror positions horizontally to double training data
        static std::vector<TrainingPosition> mirrorData(const std::vector<TrainingPosition>& data);

        // Game phase classification from active features
        // Uses material-based phase: Knight/Bishop=1, Rook=2, Queen=4 (max 24)
        enum class GamePhase { Opening, Middlegame, Endgame };
        static int computeMaterialPhase(const std::vector<int>& activeFeatures);
        static GamePhase classifyPhase(const std::vector<int>& activeFeatures);

        // ELO estimation: play NNUE engine vs handcrafted engine
        EloResult estimateElo(Network& net, int numGames = 100, int thinkTimeMs = 500,
            int baselineElo = 2100,
            std::function<void(int gamesPlayed, int totalGames)> progressCallback = nullptr,
            std::atomic<bool>* cancelFlag = nullptr);

    private:
        // Adam optimizer state
        struct AdamState {
            std::vector<float> m;   // first moment
            std::vector<float> v;   // second moment
            int t = 0;              // timestep
        };

        void adamUpdate(std::vector<float>& params, std::vector<float>& grads,
                       AdamState& state, float lr, float beta1 = 0.9f, float beta2 = 0.999f);

        // Loss function (sigmoid MSE with lambda blending)
        float computeLoss(float predicted, float targetEval, float gameResult,
                         float lambda, float evalScale);

        // Sigmoid helper
        static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

        // Single-threaded data generation worker (used by parallel version)
        std::vector<TrainingPosition> generateGamesWorker(
            int numGames, int thinkTimeMs, int seedOffset,
            int randomOpeningMoves,
            std::atomic<bool>* cancelFlag = nullptr);

        // Duck chess self-play data generation worker
        std::vector<TrainingPosition> generateDuckGamesWorker(
            int numGames, int thinkTimeMs, int seedOffset,
            int randomOpeningMoves,
            std::atomic<bool>* cancelFlag = nullptr);
    };
}
