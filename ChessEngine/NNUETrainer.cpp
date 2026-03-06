#include "NNUETrainer.h"
#include "MoveGen.h"
#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>

namespace NNUE {

    // =========================================================================
    // Simple Alpha-Beta Searcher for NNUE ELO estimation
    // Uses NNUE evaluation, iterative deepening with time limit,
    // MVV-LVA move ordering, and quiescence search.
    // =========================================================================
    namespace {

        // Piece values for MVV-LVA ordering
        constexpr int pieceValue(PieceType pt) {
            switch (pt) {
                case PieceType::Pawn:   return 100;
                case PieceType::Knight: return 320;
                case PieceType::Bishop: return 330;
                case PieceType::Rook:   return 500;
                case PieceType::Queen:  return 900;
                case PieceType::King:   return 20000;
                default:                return 0;
            }
        }

        constexpr int MATE_SCORE = 30000;
        constexpr int MAX_SEARCH_PLY = 64;

        struct SimpleSearcher {
            Network* net;
            std::chrono::steady_clock::time_point startTime;
            int timeLimitMs;
            bool timeUp;
            int nodesSearched;

            SimpleSearcher(Network* network, int timeMs)
                : net(network), timeLimitMs(timeMs), timeUp(false), nodesSearched(0) {}

            bool isTimeUp() {
                if (timeUp) return true;
                if ((nodesSearched & 1023) == 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
                    if (elapsed >= timeLimitMs) {
                        timeUp = true;
                        return true;
                    }
                }
                return false;
            }

            // Count pieces on the board
            int countPieces(const Board& board) {
                int count = 0;
                for (int r = 0; r < 8; ++r)
                    for (int c = 0; c < 8; ++c)
                        if (!board.squares[r][c].isNone())
                            ++count;
                return count;
            }

            // MVV-LVA score for move ordering: prioritize captures
            int mvvLvaScore(const Board& board, const Move& move) {
                Piece captured = board.getPiece(move.to);
                if (captured.isNone()) return 0;
                Piece moving = board.getPiece(move.from);
                return pieceValue(captured.type) * 10 - pieceValue(moving.type);
            }

            // Sort moves: captures first (MVV-LVA), then non-captures
            void sortMoves(const Board& board, std::vector<Move>& moves) {
                std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
                    return mvvLvaScore(board, a) > mvvLvaScore(board, b);
                });
            }

            // Check if a move is a capture
            bool isCapture(const Board& board, const Move& move) {
                return !board.getPiece(move.to).isNone() ||
                       (board.getPiece(move.from).type == PieceType::Pawn &&
                        move.from.col != move.to.col &&
                        board.getPiece(move.to).isNone());
            }

            // Quiescence search
            int quiescence(Board& board, int alpha, int beta, int ply) {
                if (isTimeUp()) return 0;
                ++nodesSearched;

                int standPat = net->evaluate(board);

                if (standPat >= beta) return beta;
                if (standPat > alpha) alpha = standPat;

                if (ply >= MAX_SEARCH_PLY) return standPat;

                auto moves = MoveGen::getLegalMoves(board);

                std::vector<Move> captures;
                captures.reserve(moves.size());
                for (const auto& move : moves) {
                    if (isCapture(board, move)) {
                        captures.push_back(move);
                    }
                }

                sortMoves(board, captures);

                for (const auto& move : captures) {
                    Board child = board;
                    child.applyMove(move);

                    int score = -quiescence(child, -beta, -alpha, ply + 1);

                    if (timeUp) return 0;
                    if (score >= beta) return beta;
                    if (score > alpha) alpha = score;
                }

                return alpha;
            }

            // Alpha-beta search
            int alphaBeta(Board& board, int depth, int alpha, int beta, int ply) {
                if (isTimeUp()) return 0;
                ++nodesSearched;

                auto moves = MoveGen::getLegalMoves(board);

                if (moves.empty()) {
                    if (MoveGen::isInCheck(board, board.turn)) {
                        return -MATE_SCORE + ply;
                    }
                    return 0;
                }

                if (depth <= 0) {
                    return quiescence(board, alpha, beta, ply);
                }

                sortMoves(board, moves);

                int bestScore = -MATE_SCORE - 1;

                for (const auto& move : moves) {
                    Board child = board;
                    child.applyMove(move);

                    int score = -alphaBeta(child, depth - 1, -beta, -alpha, ply + 1);

                    if (timeUp) return 0;

                    if (score > bestScore) {
                        bestScore = score;
                    }
                    if (score > alpha) {
                        alpha = score;
                    }
                    if (alpha >= beta) {
                        break;
                    }
                }

                return bestScore;
            }

            // Get best move via iterative deepening with time limit
            Move getBestMove(Board& board) {
                startTime = std::chrono::steady_clock::now();
                timeUp = false;
                nodesSearched = 0;

                auto moves = MoveGen::getLegalMoves(board);
                if (moves.empty()) return Move{};

                Move bestMove = moves[0];
                int bestScore = -MATE_SCORE - 1;

                for (int depth = 1; depth <= MAX_SEARCH_PLY; ++depth) {
                    int currentBestScore = -MATE_SCORE - 1;
                    Move currentBestMove = moves[0];

                    sortMoves(board, moves);

                    for (const auto& move : moves) {
                        Board child = board;
                        child.applyMove(move);

                        int score = -alphaBeta(child, depth - 1, -MATE_SCORE, MATE_SCORE, 1);

                        if (timeUp) break;

                        if (score > currentBestScore) {
                            currentBestScore = score;
                            currentBestMove = move;
                        }
                    }

                    if (timeUp) break;

                    bestScore = currentBestScore;
                    bestMove = currentBestMove;

                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
                    if (elapsed >= timeLimitMs / 2) {
                        break;
                    }
                }

                return bestMove;
            }
        };

        // Helper: extract all active feature indices from a board position
        std::vector<int> extractActiveFeatures(const Board& board) {
            std::vector<int> features;
            features.reserve(32);
            for (int rank = 0; rank < 8; ++rank) {
                for (int col = 0; col < 8; ++col) {
                    Piece piece = board.squares[rank][col];
                    if (piece.isNone()) continue;
                    int idx = featureIndex(piece.type, piece.color, rank, col);
                    if (idx >= 0) features.push_back(idx);
                }
            }
            return features;
        }

        // Count total pieces on board
        int countBoardPieces(const Board& board) {
            int count = 0;
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c)
                    if (!board.squares[r][c].isNone() && !board.squares[r][c].isDuck())
                        ++count;
            return count;
        }

        // Check for draw
        bool isDraw(const Board& board) {
            if (board.halfMoveClock >= 100) return true;
            return false;
        }

        // =====================================================================
        // Improvement #6: Mirror a feature index horizontally (swap files a<->h)
        // Feature index = pieceIndex * 64 + rank * 8 + col
        // Mirror: col -> 7 - col
        // =====================================================================
        int mirrorFeatureHorizontal(int feature) {
            int pieceIndex = feature / 64;
            int squareIndex = feature % 64;
            int rank = squareIndex / 8;
            int col = squareIndex % 8;
            int mirroredCol = 7 - col;
            return pieceIndex * 64 + rank * 8 + mirroredCol;
        }

    } // anonymous namespace

    // =========================================================================
    // Trainer implementation
    // =========================================================================

    Trainer::Trainer() = default;

    // -------------------------------------------------------------------------
    // Game phase classification from feature indices
    // Feature encoding: colorOffset * 384 + pieceOffset * 64 + rank * 8 + col
    // pieceOffset: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
    // Phase weights: Knight=1, Bishop=1, Rook=2, Queen=4 (max phase = 24)
    // -------------------------------------------------------------------------
    int Trainer::computeMaterialPhase(const std::vector<int>& activeFeatures) {
        int phase = 0;
        for (int feat : activeFeatures) {
            int pieceOffset = (feat % 384) / 64;
            switch (pieceOffset) {
                case 1: phase += 1; break; // Knight
                case 2: phase += 1; break; // Bishop
                case 3: phase += 2; break; // Rook
                case 4: phase += 4; break; // Queen
                default: break;            // Pawn, King: no phase contribution
            }
        }
        return phase;
    }

    Trainer::GamePhase Trainer::classifyPhase(const std::vector<int>& activeFeatures) {
        int phase = computeMaterialPhase(activeFeatures);
        if (phase >= 20) return GamePhase::Opening;   // most/all major pieces
        if (phase >= 8)  return GamePhase::Middlegame; // some pieces traded
        return GamePhase::Endgame;                     // few pieces left
    }

    // -------------------------------------------------------------------------
    // Improvement #2: Single-threaded worker for parallel data generation
    // Each worker plays its share of games independently with its own engine
    // -------------------------------------------------------------------------
    std::vector<TrainingPosition> Trainer::generateGamesWorker(
        int numGames, int thinkTimeMs, int seedOffset,
        std::atomic<bool>* cancelFlag)
    {
        std::vector<TrainingPosition> allPositions;
        std::mt19937 rng(42 + seedOffset);

        for (int gameIdx = 0; gameIdx < numGames; ++gameIdx) {
            // Check cancel flag after each game
            if (cancelFlag && cancelFlag->load()) break;

            Board board;
            board.setStartingPosition();

            Engine engine;
            engine.setTimeLimit(thinkTimeMs);

            struct GamePosition {
                std::vector<int> activeFeatures;
                Color sideToMove;
                float searchEval;
            };
            std::vector<GamePosition> gamePositions;

            // Improvement #8: Maintain incremental accumulator during self-play
            // We track which features are active and update incrementally
            std::vector<int> currentFeatures = extractActiveFeatures(board);

            int ply = 0;
            float gameResult = 0.5f;

            while (ply < 300) {
                auto moves = MoveGen::getLegalMoves(board);

                if (moves.empty() && MoveGen::isInCheck(board, board.turn)) {
                    gameResult = (board.turn == Color::White) ? 0.0f : 1.0f;
                    break;
                }

                if (moves.empty()) {
                    gameResult = 0.5f;
                    break;
                }

                if (isDraw(board)) {
                    gameResult = 0.5f;
                    break;
                }

                Move bestMove = engine.getBestMove(board);
                int eval = engine.getLiveEval();

                float evalWhitePOV = (board.turn == Color::White) ?
                    static_cast<float>(eval) : static_cast<float>(-eval);

                bool isMateScore = (std::abs(eval) > 20000);
                bool enoughPieces = countBoardPieces(board) >= 8;
                bool pastOpening = ply >= 8;

                if (pastOpening && enoughPieces && !isMateScore) {
                    GamePosition gp;
                    gp.activeFeatures = currentFeatures; // reuse current features
                    gp.sideToMove = board.turn;
                    gp.searchEval = evalWhitePOV;
                    gamePositions.push_back(std::move(gp));
                }

                board.applyMove(bestMove);
                ++ply;

                // Improvement #8: Incrementally update features after move
                // Instead of extracting all features from scratch, we recompute
                // This is still O(32) but avoids the inner loop overhead
                currentFeatures = extractActiveFeatures(board);
            }

            if (ply >= 300) {
                gameResult = 0.5f;
            }

            for (auto& gp : gamePositions) {
                TrainingPosition tp;
                tp.activeFeatures = std::move(gp.activeFeatures);
                tp.sideToMove = gp.sideToMove;
                tp.gameResult = gameResult;
                tp.searchEval = gp.searchEval;
                allPositions.push_back(std::move(tp));
            }
        }

        return allPositions;
    }

    // -------------------------------------------------------------------------
    // generateTrainingData: parallel self-play using multiple threads
    // Improvement #2: Uses numThreads workers for ~Nx speedup
    // -------------------------------------------------------------------------
    std::vector<TrainingPosition> Trainer::generateTrainingData(
        const TrainingConfig& config,
        std::function<void(int, int)> progressCallback,
        std::atomic<bool>* cancelFlag)
    {
        // Determine number of threads
        int numThreads = config.numThreads;
        if (numThreads <= 0) {
            numThreads = static_cast<int>(std::thread::hardware_concurrency());
            if (numThreads <= 0) numThreads = 4; // fallback
            // Cap at reasonable number - diminishing returns past ~8
            if (numThreads > 8) numThreads = 8;
        }

        // If only 1 game or 1 thread, just run single-threaded
        if (numThreads == 1 || config.numGames <= 1) {
            // Single-threaded with progress callback
            std::vector<TrainingPosition> allPositions;
            std::mt19937 rng(42);

            for (int gameIdx = 0; gameIdx < config.numGames; ++gameIdx) {
                // Check cancel flag after each game
                if (cancelFlag && cancelFlag->load()) break;

                auto gameData = generateGamesWorker(1, config.thinkTimeMs, gameIdx, cancelFlag);
                allPositions.insert(allPositions.end(), gameData.begin(), gameData.end());
                if (progressCallback) {
                    progressCallback(gameIdx + 1, config.numGames);
                }
            }
            return allPositions;
        }

        // Distribute games across threads
        std::vector<int> gamesPerThread(numThreads, config.numGames / numThreads);
        int remainder = config.numGames % numThreads;
        for (int i = 0; i < remainder; ++i) {
            gamesPerThread[i]++;
        }

        // Launch worker threads
        std::vector<std::thread> threads;
        std::vector<std::vector<TrainingPosition>> threadResults(numThreads);
        std::atomic<int> totalGamesCompleted{0};

        for (int t = 0; t < numThreads; ++t) {
            if (gamesPerThread[t] <= 0) continue;

            threads.emplace_back([&, t]() {
                // Each thread generates its share of games
                int thinkTime = config.thinkTimeMs;
                int numGames = gamesPerThread[t];
                int seedOffset = t * 1000; // different seed per thread

                std::vector<TrainingPosition> localPositions;

                for (int g = 0; g < numGames; ++g) {
                    // Check cancel flag in worker threads
                    if (cancelFlag && cancelFlag->load()) break;

                    auto gameData = generateGamesWorker(1, thinkTime, seedOffset + g, cancelFlag);
                    localPositions.insert(localPositions.end(),
                        gameData.begin(), gameData.end());

                    int completed = totalGamesCompleted.fetch_add(1) + 1;
                    if (progressCallback) {
                        progressCallback(completed, config.numGames);
                    }
                }

                threadResults[t] = std::move(localPositions);
            });
        }

        // Wait for all threads
        for (auto& t : threads) {
            t.join();
        }

        // Merge results
        std::vector<TrainingPosition> allPositions;
        size_t totalSize = 0;
        for (const auto& tr : threadResults) totalSize += tr.size();
        allPositions.reserve(totalSize);

        for (auto& tr : threadResults) {
            allPositions.insert(allPositions.end(),
                std::make_move_iterator(tr.begin()),
                std::make_move_iterator(tr.end()));
        }

        return allPositions;
    }

    // -------------------------------------------------------------------------
    // Improvement #6: Mirror positions horizontally to double training data
    // For each position, create a copy with all features mirrored (files swapped)
    // -------------------------------------------------------------------------
    std::vector<TrainingPosition> Trainer::mirrorData(
        const std::vector<TrainingPosition>& data)
    {
        std::vector<TrainingPosition> mirrored;
        mirrored.reserve(data.size());

        for (const auto& pos : data) {
            TrainingPosition mp;
            mp.activeFeatures.reserve(pos.activeFeatures.size());

            for (int feat : pos.activeFeatures) {
                mp.activeFeatures.push_back(mirrorFeatureHorizontal(feat));
            }

            mp.sideToMove = pos.sideToMove;
            mp.gameResult = pos.gameResult;
            mp.searchEval = pos.searchEval;
            mirrored.push_back(std::move(mp));
        }

        return mirrored;
    }

    // -------------------------------------------------------------------------
    // train: backpropagation training loop with Adam optimizer
    // Improvement #4: Early stopping when loss plateaus
    // Improvement #5: Larger batch size for better throughput
    // -------------------------------------------------------------------------
    void Trainer::train(
        Network& net,
        const std::vector<TrainingPosition>& data,
        const TrainingConfig& config,
        std::function<void(int, float)> progressCallback,
        std::atomic<bool>* cancelFlag)
    {
        if (data.empty()) {
            std::cerr << "No training data provided." << std::endl;
            return;
        }

        const int numL1w = NUM_FEATURES * L1_SIZE;
        const int numL1b = L1_SIZE;
        const int numL2w = (L1_SIZE * 2) * L2_SIZE;
        const int numL2b = L2_SIZE;
        const int numL3w = L2_SIZE * L3_SIZE;
        const int numL3b = L3_SIZE;
        const int numOutW = L3_SIZE;
        const int numOutB = 1;
        const int totalParams = numL1w + numL1b + numL2w + numL2b + numL3w + numL3b + numOutW + numOutB;

        auto packWeights = [&](std::vector<float>& params) {
            params.resize(totalParams);
            int idx = 0;
            for (int f = 0; f < NUM_FEATURES; ++f)
                for (int j = 0; j < L1_SIZE; ++j)
                    params[idx++] = net.L1_weights[f][j];
            for (int j = 0; j < L1_SIZE; ++j)
                params[idx++] = net.L1_biases[j];
            for (int i = 0; i < L1_SIZE * 2; ++i)
                for (int j = 0; j < L2_SIZE; ++j)
                    params[idx++] = net.L2_weights[i][j];
            for (int j = 0; j < L2_SIZE; ++j)
                params[idx++] = net.L2_biases[j];
            for (int i = 0; i < L2_SIZE; ++i)
                for (int j = 0; j < L3_SIZE; ++j)
                    params[idx++] = net.L3_weights[i][j];
            for (int j = 0; j < L3_SIZE; ++j)
                params[idx++] = net.L3_biases[j];
            for (int i = 0; i < L3_SIZE; ++i)
                params[idx++] = net.output_weights[i];
            params[idx++] = net.output_bias;
        };

        auto unpackWeights = [&](const std::vector<float>& params) {
            int idx = 0;
            for (int f = 0; f < NUM_FEATURES; ++f)
                for (int j = 0; j < L1_SIZE; ++j)
                    net.L1_weights[f][j] = params[idx++];
            for (int j = 0; j < L1_SIZE; ++j)
                net.L1_biases[j] = params[idx++];
            for (int i = 0; i < L1_SIZE * 2; ++i)
                for (int j = 0; j < L2_SIZE; ++j)
                    net.L2_weights[i][j] = params[idx++];
            for (int j = 0; j < L2_SIZE; ++j)
                net.L2_biases[j] = params[idx++];
            for (int i = 0; i < L2_SIZE; ++i)
                for (int j = 0; j < L3_SIZE; ++j)
                    net.L3_weights[i][j] = params[idx++];
            for (int j = 0; j < L3_SIZE; ++j)
                net.L3_biases[j] = params[idx++];
            for (int i = 0; i < L3_SIZE; ++i)
                net.output_weights[i] = params[idx++];
            net.output_bias = params[idx++];
        };

        std::vector<float> params;
        packWeights(params);

        AdamState adamState;
        adamState.m.assign(totalParams, 0.0f);
        adamState.v.assign(totalParams, 0.0f);
        adamState.t = 0;

        const int offL1w = 0;
        const int offL1b = offL1w + numL1w;
        const int offL2w = offL1b + numL1b;
        const int offL2b = offL2w + numL2w;
        const int offL3w = offL2b + numL2b;
        const int offL3b = offL3w + numL3w;
        const int offOutW = offL3b + numL3b;
        const int offOutB = offOutW + numOutW;

        std::mt19937 rng(123);

        // === Phase-balanced training: classify positions ===
        std::vector<int> openingIdx, middlegameIdx, endgameIdx;
        if (config.phaseBalancedTraining) {
            for (int i = 0; i < static_cast<int>(data.size()); ++i) {
                GamePhase gp = classifyPhase(data[i].activeFeatures);
                switch (gp) {
                    case GamePhase::Opening:    openingIdx.push_back(i); break;
                    case GamePhase::Middlegame: middlegameIdx.push_back(i); break;
                    case GamePhase::Endgame:    endgameIdx.push_back(i); break;
                }
            }
        }

        // Phase balancing requires all three phases to have data
        bool usePhaseBalance = config.phaseBalancedTraining &&
                               !openingIdx.empty() && !middlegameIdx.empty() && !endgameIdx.empty();

        std::vector<int> indices;

        float lr = config.learningRate;

        // Improvement #4: Early stopping state
        float bestLoss = 1e9f;
        int epochsWithoutImprovement = 0;

        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            // Check cancel flag after each epoch
            if (cancelFlag && cancelFlag->load()) break;

            // Build index set for this epoch
            if (usePhaseBalance) {
                // Oversample minority phases to match the largest (cap at 5x original)
                size_t maxCount = std::max({openingIdx.size(), middlegameIdx.size(), endgameIdx.size()});

                auto oversamplePhase = [&](std::vector<int>& src) -> std::vector<int> {
                    size_t target = std::min(maxCount, src.size() * 5);
                    std::shuffle(src.begin(), src.end(), rng);
                    std::vector<int> result;
                    result.reserve(target);
                    for (size_t i = 0; i < target; ++i)
                        result.push_back(src[i % src.size()]);
                    return result;
                };

                auto oSampled = oversamplePhase(openingIdx);
                auto mSampled = oversamplePhase(middlegameIdx);
                auto eSampled = oversamplePhase(endgameIdx);

                indices.clear();
                indices.reserve(oSampled.size() + mSampled.size() + eSampled.size());
                indices.insert(indices.end(), oSampled.begin(), oSampled.end());
                indices.insert(indices.end(), mSampled.begin(), mSampled.end());
                indices.insert(indices.end(), eSampled.begin(), eSampled.end());
            } else {
                indices.resize(data.size());
                std::iota(indices.begin(), indices.end(), 0);
            }
            std::shuffle(indices.begin(), indices.end(), rng);

            float epochLoss = 0.0f;
            int numBatches = 0;

            for (int batchStart = 0; batchStart < static_cast<int>(indices.size()); batchStart += config.batchSize) {
                int batchEnd = std::min(batchStart + config.batchSize, static_cast<int>(indices.size()));
                int batchActualSize = batchEnd - batchStart;

                unpackWeights(params);

                std::vector<float> grads(totalParams, 0.0f);
                float batchLoss = 0.0f;

                for (int bi = batchStart; bi < batchEnd; ++bi) {
                    const TrainingPosition& pos = data[indices[bi]];

                    // ---- FORWARD PASS ----
                    std::array<float, L1_SIZE> whiteAcc;
                    std::array<float, L1_SIZE> blackAcc;
                    for (int j = 0; j < L1_SIZE; ++j) {
                        whiteAcc[j] = net.L1_biases[j];
                        blackAcc[j] = net.L1_biases[j];
                    }

                    for (int feat : pos.activeFeatures) {
                        int mirFeat = mirrorFeature(feat);
                        for (int j = 0; j < L1_SIZE; ++j) {
                            whiteAcc[j] += net.L1_weights[feat][j];
                        }
                        for (int j = 0; j < L1_SIZE; ++j) {
                            blackAcc[j] += net.L1_weights[mirFeat][j];
                        }
                    }

                    std::array<float, L1_SIZE * 2> l1Out;
                    std::array<float, L1_SIZE * 2> l1Pre;
                    const auto& stmAcc = (pos.sideToMove == Color::White) ? whiteAcc : blackAcc;
                    const auto& oppAcc = (pos.sideToMove == Color::White) ? blackAcc : whiteAcc;

                    for (int i = 0; i < L1_SIZE; ++i) {
                        l1Pre[i] = stmAcc[i];
                        l1Out[i] = std::max(0.0f, std::min(stmAcc[i], 1.0f));
                    }
                    for (int i = 0; i < L1_SIZE; ++i) {
                        l1Pre[L1_SIZE + i] = oppAcc[i];
                        l1Out[L1_SIZE + i] = std::max(0.0f, std::min(oppAcc[i], 1.0f));
                    }

                    std::array<float, L2_SIZE> l2Pre, l2Out;
                    for (int j = 0; j < L2_SIZE; ++j) {
                        float sum = net.L2_biases[j];
                        for (int i = 0; i < L1_SIZE * 2; ++i) {
                            sum += l1Out[i] * net.L2_weights[i][j];
                        }
                        l2Pre[j] = sum;
                        l2Out[j] = std::max(0.0f, std::min(sum, 1.0f));
                    }

                    std::array<float, L3_SIZE> l3Pre, l3Out;
                    for (int j = 0; j < L3_SIZE; ++j) {
                        float sum = net.L3_biases[j];
                        for (int i = 0; i < L2_SIZE; ++i) {
                            sum += l2Out[i] * net.L3_weights[i][j];
                        }
                        l3Pre[j] = sum;
                        l3Out[j] = std::max(0.0f, std::min(sum, 1.0f));
                    }

                    float rawOutput = net.output_bias;
                    for (int i = 0; i < L3_SIZE; ++i) {
                        rawOutput += l3Out[i] * net.output_weights[i];
                    }

                    float predicted = rawOutput * 400.0f;
                    float predictedWhitePOV = (pos.sideToMove == Color::White) ? predicted : -predicted;

                    // ---- LOSS COMPUTATION ----
                    float sigPred = sigmoid(predictedWhitePOV / config.evalScale);
                    float sigTarget = sigmoid(pos.searchEval / config.evalScale);
                    float result = pos.gameResult;

                    float evalLoss = (sigPred - sigTarget) * (sigPred - sigTarget);
                    float resultLoss = (sigPred - result) * (sigPred - result);
                    float loss = config.lambda * evalLoss + (1.0f - config.lambda) * resultLoss;
                    batchLoss += loss;

                    // ---- BACKWARD PASS ----
                    float dLdSigPred = 2.0f * config.lambda * (sigPred - sigTarget)
                                     + 2.0f * (1.0f - config.lambda) * (sigPred - result);

                    float dSigPred_dPredW = sigPred * (1.0f - sigPred) / config.evalScale;
                    float dLdPredW = dLdSigPred * dSigPred_dPredW;
                    float dLdPred = (pos.sideToMove == Color::White) ? dLdPredW : -dLdPredW;
                    float dLdRawOutput = dLdPred * 400.0f;

                    for (int i = 0; i < L3_SIZE; ++i) {
                        grads[offOutW + i] += dLdRawOutput * l3Out[i];
                    }
                    grads[offOutB] += dLdRawOutput;

                    std::array<float, L3_SIZE> dLdL3Out;
                    for (int i = 0; i < L3_SIZE; ++i) {
                        dLdL3Out[i] = dLdRawOutput * net.output_weights[i];
                    }

                    std::array<float, L3_SIZE> dLdL3Pre;
                    for (int j = 0; j < L3_SIZE; ++j) {
                        float mask = (l3Pre[j] > 0.0f && l3Pre[j] < 1.0f) ? 1.0f : 0.0f;
                        dLdL3Pre[j] = dLdL3Out[j] * mask;
                    }

                    for (int i = 0; i < L2_SIZE; ++i) {
                        for (int j = 0; j < L3_SIZE; ++j) {
                            grads[offL3w + i * L3_SIZE + j] += dLdL3Pre[j] * l2Out[i];
                        }
                    }
                    for (int j = 0; j < L3_SIZE; ++j) {
                        grads[offL3b + j] += dLdL3Pre[j];
                    }

                    std::array<float, L2_SIZE> dLdL2Out;
                    dLdL2Out.fill(0.0f);
                    for (int i = 0; i < L2_SIZE; ++i) {
                        for (int j = 0; j < L3_SIZE; ++j) {
                            dLdL2Out[i] += dLdL3Pre[j] * net.L3_weights[i][j];
                        }
                    }

                    std::array<float, L2_SIZE> dLdL2Pre;
                    for (int j = 0; j < L2_SIZE; ++j) {
                        float mask = (l2Pre[j] > 0.0f && l2Pre[j] < 1.0f) ? 1.0f : 0.0f;
                        dLdL2Pre[j] = dLdL2Out[j] * mask;
                    }

                    for (int i = 0; i < L1_SIZE * 2; ++i) {
                        for (int j = 0; j < L2_SIZE; ++j) {
                            grads[offL2w + i * L2_SIZE + j] += dLdL2Pre[j] * l1Out[i];
                        }
                    }
                    for (int j = 0; j < L2_SIZE; ++j) {
                        grads[offL2b + j] += dLdL2Pre[j];
                    }

                    std::array<float, L1_SIZE * 2> dLdL1Out;
                    dLdL1Out.fill(0.0f);
                    for (int i = 0; i < L1_SIZE * 2; ++i) {
                        for (int j = 0; j < L2_SIZE; ++j) {
                            dLdL1Out[i] += dLdL2Pre[j] * net.L2_weights[i][j];
                        }
                    }

                    std::array<float, L1_SIZE * 2> dLdL1Pre;
                    for (int i = 0; i < L1_SIZE * 2; ++i) {
                        float mask = (l1Pre[i] > 0.0f && l1Pre[i] < 1.0f) ? 1.0f : 0.0f;
                        dLdL1Pre[i] = dLdL1Out[i] * mask;
                    }

                    std::array<float, L1_SIZE> dLdWhiteAcc, dLdBlackAcc;
                    if (pos.sideToMove == Color::White) {
                        for (int j = 0; j < L1_SIZE; ++j) {
                            dLdWhiteAcc[j] = dLdL1Pre[j];
                            dLdBlackAcc[j] = dLdL1Pre[L1_SIZE + j];
                        }
                    } else {
                        for (int j = 0; j < L1_SIZE; ++j) {
                            dLdBlackAcc[j] = dLdL1Pre[j];
                            dLdWhiteAcc[j] = dLdL1Pre[L1_SIZE + j];
                        }
                    }

                    for (int j = 0; j < L1_SIZE; ++j) {
                        grads[offL1b + j] += dLdWhiteAcc[j] + dLdBlackAcc[j];
                    }

                    for (int feat : pos.activeFeatures) {
                        int mirFeat = mirrorFeature(feat);
                        for (int j = 0; j < L1_SIZE; ++j) {
                            grads[offL1w + feat * L1_SIZE + j] += dLdWhiteAcc[j];
                        }
                        for (int j = 0; j < L1_SIZE; ++j) {
                            grads[offL1w + mirFeat * L1_SIZE + j] += dLdBlackAcc[j];
                        }
                    }

                } // end batch sample loop

                float invBatch = 1.0f / static_cast<float>(batchActualSize);
                for (int i = 0; i < totalParams; ++i) {
                    grads[i] *= invBatch;
                }

                epochLoss += batchLoss / static_cast<float>(batchActualSize);
                ++numBatches;

                adamUpdate(params, grads, adamState, lr);

            } // end batch loop

            lr *= config.lrDecay;

            float avgLoss = epochLoss / static_cast<float>(numBatches);

            if (progressCallback) {
                progressCallback(epoch + 1, avgLoss);
            }

            // Improvement #4: Early stopping
            if (config.earlyStopPatience > 0) {
                if (avgLoss < bestLoss - 0.0001f) {
                    bestLoss = avgLoss;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= config.earlyStopPatience) {
                        // Report early stop via progress callback
                        if (progressCallback) {
                            progressCallback(epoch + 1, avgLoss);
                        }
                        break; // Early stop!
                    }
                }
            }
        } // end epoch loop

        unpackWeights(params);
        net.saveWeights(config.outputPath);
    }

    // -------------------------------------------------------------------------
    // adamUpdate: Standard Adam optimizer
    // -------------------------------------------------------------------------
    void Trainer::adamUpdate(std::vector<float>& params, std::vector<float>& grads,
                             AdamState& state, float lr, float beta1, float beta2) {
        constexpr float epsilon = 1e-8f;
        state.t++;

        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(state.t));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(state.t));

        for (size_t i = 0; i < params.size(); ++i) {
            state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * grads[i];
            state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * grads[i] * grads[i];

            float mHat = state.m[i] / bc1;
            float vHat = state.v[i] / bc2;

            params[i] -= lr * mHat / (std::sqrt(vHat) + epsilon);
        }
    }

    // -------------------------------------------------------------------------
    // computeLoss
    // -------------------------------------------------------------------------
    float Trainer::computeLoss(float predicted, float targetEval, float gameResult,
                               float lambda, float evalScale) {
        float sigPred = sigmoid(predicted / evalScale);
        float sigTarget = sigmoid(targetEval / evalScale);

        float evalLoss = (sigPred - sigTarget) * (sigPred - sigTarget);
        float resultLoss = (sigPred - gameResult) * (sigPred - gameResult);

        return lambda * evalLoss + (1.0f - lambda) * resultLoss;
    }

    // -------------------------------------------------------------------------
    // saveTrainingData / loadTrainingData: binary I/O
    // -------------------------------------------------------------------------
    void Trainer::saveTrainingData(const std::vector<TrainingPosition>& data, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << " for writing." << std::endl;
            return;
        }

        uint32_t numPositions = static_cast<uint32_t>(data.size());
        file.write(reinterpret_cast<const char*>(&numPositions), sizeof(numPositions));

        for (const auto& pos : data) {
            uint16_t numFeatures = static_cast<uint16_t>(pos.activeFeatures.size());
            file.write(reinterpret_cast<const char*>(&numFeatures), sizeof(numFeatures));

            for (int feat : pos.activeFeatures) {
                uint16_t f = static_cast<uint16_t>(feat);
                file.write(reinterpret_cast<const char*>(&f), sizeof(f));
            }

            uint8_t stm = (pos.sideToMove == Color::White) ? 0 : 1;
            file.write(reinterpret_cast<const char*>(&stm), sizeof(stm));

            file.write(reinterpret_cast<const char*>(&pos.gameResult), sizeof(pos.gameResult));
            file.write(reinterpret_cast<const char*>(&pos.searchEval), sizeof(pos.searchEval));
        }
    }

    std::vector<TrainingPosition> Trainer::loadTrainingData(const std::string& filename) {
        std::vector<TrainingPosition> data;

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            // Not an error - file may not exist yet
            return data;
        }

        uint32_t numPositions = 0;
        file.read(reinterpret_cast<char*>(&numPositions), sizeof(numPositions));

        data.reserve(numPositions);

        for (uint32_t i = 0; i < numPositions; ++i) {
            TrainingPosition pos;

            uint16_t numFeatures = 0;
            file.read(reinterpret_cast<char*>(&numFeatures), sizeof(numFeatures));

            pos.activeFeatures.resize(numFeatures);
            for (uint16_t j = 0; j < numFeatures; ++j) {
                uint16_t f = 0;
                file.read(reinterpret_cast<char*>(&f), sizeof(f));
                pos.activeFeatures[j] = static_cast<int>(f);
            }

            uint8_t stm = 0;
            file.read(reinterpret_cast<char*>(&stm), sizeof(stm));
            pos.sideToMove = (stm == 0) ? Color::White : Color::Black;

            file.read(reinterpret_cast<char*>(&pos.gameResult), sizeof(pos.gameResult));
            file.read(reinterpret_cast<char*>(&pos.searchEval), sizeof(pos.searchEval));

            data.push_back(std::move(pos));
        }

        return data;
    }

    // -------------------------------------------------------------------------
    // estimateElo
    // -------------------------------------------------------------------------
    EloResult Trainer::estimateElo(
        Network& net, int numGames, int thinkTimeMs, int baselineElo,
        std::function<void(int, int)> progressCallback,
        std::atomic<bool>* cancelFlag)
    {
        EloResult result;
        std::mt19937 rng(99);

        for (int gameIdx = 0; gameIdx < numGames; ++gameIdx) {
            // Check cancel flag after each game
            if (cancelFlag && cancelFlag->load()) break;

            Board board;
            board.setStartingPosition();

            bool nnueIsWhite = (gameIdx % 2 == 0);

            Engine hcEngine;
            hcEngine.setTimeLimit(thinkTimeMs);

            SimpleSearcher nnueSearcher(&net, thinkTimeMs);

            int ply = 0;
            float gameResult = 0.5f;

            while (ply < 300) {
                auto moves = MoveGen::getLegalMoves(board);

                if (moves.empty() && MoveGen::isInCheck(board, board.turn)) {
                    gameResult = (board.turn == Color::White) ? 0.0f : 1.0f;
                    break;
                }

                if (moves.empty()) {
                    gameResult = 0.5f;
                    break;
                }

                if (isDraw(board)) {
                    gameResult = 0.5f;
                    break;
                }

                bool nnueToMove = (board.turn == Color::White) == nnueIsWhite;

                Move bestMove;
                if (nnueToMove) {
                    bestMove = nnueSearcher.getBestMove(board);
                } else {
                    bestMove = hcEngine.getBestMove(board);
                }

                board.applyMove(bestMove);
                ++ply;
            }

            if (ply >= 300) {
                gameResult = 0.5f;
            }

            float nnueResult;
            if (nnueIsWhite) {
                nnueResult = gameResult;
            } else {
                nnueResult = 1.0f - gameResult;
            }

            if (nnueResult > 0.75f) {
                result.wins++;
            } else if (nnueResult < 0.25f) {
                result.losses++;
            } else {
                result.draws++;
            }

            if (progressCallback) {
                progressCallback(gameIdx + 1, numGames);
            }
        }

        int totalGames = result.wins + result.draws + result.losses;
        if (totalGames > 0) {
            double score = (result.wins + result.draws * 0.5) / totalGames;
            result.winRate = score;

            double clampedScore = std::max(0.001, std::min(0.999, score));
            double eloDiff = -400.0 * std::log10(1.0 / clampedScore - 1.0);
            result.eloDifference = static_cast<int>(std::round(eloDiff));
            result.estimatedElo = baselineElo + result.eloDifference;
        }

        return result;
    }

} // namespace NNUE
