#include "NNUETrainer.h"
#include "DuckNNUE.h"
#include "MoveGen.h"
#include "TrainAVX.h"
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
            void sortMoves(const Board& board, MoveList& moves) {
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

                MoveList moves; MoveGen::getLegalMoves(board, moves);

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
                    child.recomputeBitboards();

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

                MoveList moves; MoveGen::getLegalMoves(board, moves);

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
                    child.recomputeBitboards();

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

                MoveList moves; MoveGen::getLegalMoves(board, moves);
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
                        child.recomputeBitboards();

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
                    if (piece.isNone() || piece.isDuck()) continue;
                    int idx = featureIndex(piece.type, piece.color, rank, col);
                    if (idx >= 0) features.push_back(idx);
                }
            }
            return features;
        }

        // Helper: extract active features for duck chess (832 features)
        // Includes standard piece features (0-767) + duck square feature (768-831)
        std::vector<int> extractDuckActiveFeatures(const Board& board) {
            std::vector<int> features;
            features.reserve(33);  // up to 32 pieces + 1 duck
            for (int rank = 0; rank < 8; ++rank) {
                for (int col = 0; col < 8; ++col) {
                    Piece piece = board.squares[rank][col];
                    if (piece.isNone() || piece.isDuck()) continue;
                    int idx = featureIndex(piece.type, piece.color, rank, col);
                    if (idx >= 0) features.push_back(idx);
                }
            }
            // Add duck square feature
            if (board.isDuckChess && board.duckSquare.isValid()) {
                features.push_back(DuckNNUE::duckFeatureIndex(
                    board.duckSquare.rank, board.duckSquare.col));
            }
            return features;
        }

        // Mirror a feature index horizontally (swap files a<->h)
        int mirrorFeatureHorizontal(int feature) {
            int pieceIndex  = feature / 64;
            int squareIdx   = feature % 64;
            int rank2       = squareIdx / 8;
            int col2        = squareIdx % 8;
            return pieceIndex * 64 + rank2 * 8 + (7 - col2);
        }

        // Mirror a duck feature horizontally (swap files a<->h)
        int mirrorDuckFeatureHorizontal(int feature) {
            if (feature < DuckNNUE::DUCK_FEATURE_OFFSET) {
                return mirrorFeatureHorizontal(feature);  // standard piece feature
            }
            // Duck feature: mirror the col
            int sq = feature - DuckNNUE::DUCK_FEATURE_OFFSET;
            int rank = sq / 8;
            int col = sq % 8;
            int mirroredCol = 7 - col;
            return DuckNNUE::DUCK_FEATURE_OFFSET + rank * 8 + mirroredCol;
        }

        // Color-swap vertical mirror for duck chess training data.
        // Flips the board vertically (rank 0 ↔ rank 7) and swaps piece colors,
        // producing a position where the original black side now moves first.
        // This doubles the dataset and balances white/black perspective coverage.
        //
        // Feature encoding (768): pieceIndex * 64 + rank * 8 + col
        //   White pieces: pieceIndex 0-5  (Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5)
        //   Black pieces: pieceIndex 6-11
        // Duck features (768-831): DUCK_FEATURE_OFFSET + rank * 8 + col
        //
        // Transform: rank → (7-rank), pieceIndex → (pieceIndex XOR 6), duck rank → (7-rank)
        // Also: sideToMove flipped, searchEval negated, gameResult flipped (1↔0, 0.5 stays)
        TrainingPosition colorSwapMirror(const TrainingPosition& pos) {
            TrainingPosition mp;
            mp.activeFeatures.reserve(pos.activeFeatures.size());
            for (int feat : pos.activeFeatures) {
                if (feat < DuckNNUE::DUCK_FEATURE_OFFSET) {
                    // Standard piece feature: flip rank, swap color
                    int pieceIdx  = feat / 64;
                    int squareIdx = feat % 64;
                    int rank2     = squareIdx / 8;
                    int col2      = squareIdx % 8;
                    int newPiece  = pieceIdx ^ 6;          // swap White↔Black (0-5 ↔ 6-11)
                    int newRank   = 7 - rank2;             // flip rank
                    mp.activeFeatures.push_back(newPiece * 64 + newRank * 8 + col2);
                } else {
                    // Duck feature: flip rank only (duck has no color)
                    int sq      = feat - DuckNNUE::DUCK_FEATURE_OFFSET;
                    int rank2   = sq / 8;
                    int col2    = sq % 8;
                    int newRank = 7 - rank2;
                    mp.activeFeatures.push_back(DuckNNUE::DUCK_FEATURE_OFFSET + newRank * 8 + col2);
                }
            }
            mp.sideToMove = (pos.sideToMove == Color::White) ? Color::Black : Color::White;
            mp.searchEval = -pos.searchEval;   // negate: was white POV, now black POV
            // Flip game result: white win (1.0) → black win (0.0), draw (0.5) stays
            mp.gameResult = 1.0f - pos.gameResult;
            return mp;
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
        // Mirror a feature index horizontally (swap files a<->h)
        // Feature index = pieceIndex * 64 + rank * 8 + col
        // =====================================================================
        // NEW: SCReLU helper and its derivative for backpropagation
        // SCReLU(x) = clamp(x, 0, 1)^2
        // d/dx SCReLU(x) = 2 * clamp(x, 0, 1)  (when 0 < x < 1, else 0)
        // =====================================================================
        inline float screlu(float x) {
            float clamped = std::max(0.0f, std::min(x, 1.0f));
            return clamped * clamped;
        }

        inline float screlu_derivative(float pre_activation) {
            if (pre_activation <= 0.0f || pre_activation >= 1.0f) return 0.0f;
            return 2.0f * pre_activation;  // 2 * clamp(x, 0, 1), but x is already in (0,1)
        }

    } // anonymous namespace

    // =========================================================================
    // Trainer implementation
    // =========================================================================

    Trainer::Trainer() = default;

    // -------------------------------------------------------------------------
    // Game phase classification from feature indices
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
    // NEW: Diversified self-play worker with random opening moves
    // Each game starts with N random legal moves to create diverse positions
    // Only positions AFTER the random opening are recorded as training data
    // -------------------------------------------------------------------------
    std::vector<TrainingPosition> Trainer::generateGamesWorker(
        int numGames, int thinkTimeMs, int seedOffset,
        int randomOpeningMoves,
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

            // ===== NEW: Play random opening moves to diversify starting positions =====
            int openingMovesPlayed = 0;
            for (int i = 0; i < randomOpeningMoves; ++i) {
                MoveList moves; MoveGen::getLegalMoves(board, moves);
                if (moves.empty()) break;

                // Pick a random legal move
                std::uniform_int_distribution<int> moveDist(0, static_cast<int>(moves.size()) - 1);
                Move randomMove = moves[moveDist(rng)];
                board.applyMove(randomMove);
                board.recomputeBitboards();
                ++openingMovesPlayed;

                // Check if game ended during random opening
                MoveList nextMoves; MoveGen::getLegalMoves(board, nextMoves);
                if (nextMoves.empty()) break;
            }

            // If game ended during random opening, skip this game
            {
                MoveList moves; MoveGen::getLegalMoves(board, moves);
                if (moves.empty()) continue;
            }
            // ===== END diversified opening =====

            struct GamePosition {
                std::vector<int> activeFeatures;
                Color sideToMove;
                float searchEval;
            };
            std::vector<GamePosition> gamePositions;

            std::vector<int> currentFeatures = extractActiveFeatures(board);

            int ply = openingMovesPlayed;  // Count from actual game start
            float gameResult = 0.5f;

            while (ply < 300) {
                MoveList moves; MoveGen::getLegalMoves(board, moves);

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
                // CHANGED: record from ply 0 of engine play (opening randomization already diversified)
                bool pastOpening = (ply >= openingMovesPlayed + 2);

                if (pastOpening && enoughPieces && !isMateScore) {
                    GamePosition gp;
                    gp.activeFeatures = currentFeatures;
                    gp.sideToMove = board.turn;
                    gp.searchEval = evalWhitePOV;
                    gamePositions.push_back(std::move(gp));
                }

                board.applyMove(bestMove);
                board.recomputeBitboards();
                ++ply;

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
    // Duck chess self-play worker: generates duck chess training data
    // Uses 832-feature encoding (768 piece + 64 duck square)
    // -------------------------------------------------------------------------
    std::vector<TrainingPosition> Trainer::generateDuckGamesWorker(
        int numGames, int thinkTimeMs, int seedOffset,
        int randomOpeningMoves,
        std::atomic<bool>* cancelFlag)
    {
        std::vector<TrainingPosition> allPositions;
        std::mt19937 rng(42 + seedOffset);

        for (int gameIdx = 0; gameIdx < numGames; ++gameIdx) {
            if (cancelFlag && cancelFlag->load()) break;

            Board board;
            board.setStartingPosition();
            board.isDuckChess = true;

            Engine engine;
            engine.setTimeLimit(thinkTimeMs);

            // Play random opening moves to diversify starting positions
            int openingMovesPlayed = 0;
            for (int i = 0; i < randomOpeningMoves; ++i) {
                MoveList moves; MoveGen::getLegalMoves(board, moves);
                if (moves.empty()) break;

                std::uniform_int_distribution<int> moveDist(0, static_cast<int>(moves.size()) - 1);
                Move randomMove = moves[moveDist(rng)];
                board.applyMove(randomMove);
                board.recomputeBitboards();

                // In duck chess, place the duck randomly after each move
                if (board.isDuckChess) {
#ifdef DUCK_CHESS
                    SquareList duckPlacements; MoveGen::getDuckPlacements(board, duckPlacements);
                    if (!duckPlacements.empty()) {
                        std::uniform_int_distribution<int> duckDist(0, static_cast<int>(duckPlacements.size()) - 1);
                        board.placeDuck(duckPlacements[duckDist(rng)]);
                    }
#endif
                }

                ++openingMovesPlayed;

                MoveList nextMoves; MoveGen::getLegalMoves(board, nextMoves);
                if (nextMoves.empty()) break;
            }

            // If game ended during random opening, skip
            {
                MoveList moves; MoveGen::getLegalMoves(board, moves);
                if (moves.empty()) continue;
            }

            struct GamePosition {
                std::vector<int> activeFeatures;
                Color sideToMove;
                float searchEval;
            };
            std::vector<GamePosition> gamePositions;

            std::vector<int> currentFeatures = extractDuckActiveFeatures(board);

            int ply = openingMovesPlayed;
            float gameResult = 0.5f;

            while (ply < 300) {
                MoveList moves; MoveGen::getLegalMoves(board, moves);

                if (moves.empty()) {
                    // In duck chess, no legal moves = loss (no stalemate)
                    gameResult = (board.turn == Color::White) ? 0.0f : 1.0f;
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
                bool enoughPieces = countBoardPieces(board) >= 6;  // slightly lower threshold for duck chess
                bool pastOpening = (ply >= openingMovesPlayed + 2);

                if (pastOpening && enoughPieces && !isMateScore) {
                    GamePosition gp;
                    gp.activeFeatures = currentFeatures;
                    gp.sideToMove = board.turn;
                    gp.searchEval = evalWhitePOV;
                    gamePositions.push_back(std::move(gp));
                }

                board.applyMove(bestMove);
                board.recomputeBitboards();

                // In duck chess, the engine handles duck placement via searchDuck
                // The board state after applyMove + duck placement is recorded next iteration

                ++ply;

                currentFeatures = extractDuckActiveFeatures(board);
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
    // Routes to standard or duck chess worker based on config.isDuckChess
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
            if (numThreads > 16) numThreads = 16; // INCREASED: allow up to 16 threads
        }

        // If only 1 game or 1 thread, just run single-threaded
        if (numThreads == 1 || config.numGames <= 1) {
            std::vector<TrainingPosition> allPositions;
            std::mt19937 rng(42);

            for (int gameIdx = 0; gameIdx < config.numGames; ++gameIdx) {
                if (cancelFlag && cancelFlag->load()) break;

                auto gameData = config.isDuckChess
                    ? generateDuckGamesWorker(1, config.thinkTimeMs, gameIdx,
                        config.randomOpeningMoves, cancelFlag)
                    : generateGamesWorker(1, config.thinkTimeMs, gameIdx,
                        config.randomOpeningMoves, cancelFlag);
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
                bool isDuck = config.isDuckChess;
                int thinkTime = config.thinkTimeMs;
                int numGames = gamesPerThread[t];
                int seedOffset = t * 1000;

                std::vector<TrainingPosition> localPositions;

                for (int g = 0; g < numGames; ++g) {
                    if (cancelFlag && cancelFlag->load()) break;

                    auto gameData = isDuck
                        ? generateDuckGamesWorker(1, thinkTime, seedOffset + g,
                            config.randomOpeningMoves, cancelFlag)
                        : generateGamesWorker(1, thinkTime, seedOffset + g,
                            config.randomOpeningMoves, cancelFlag);
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
    // Mirror positions horizontally to double training data
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
    // colorSwapMirrorData: vertical mirror + color swap for duck chess
    // Doubles the dataset and balances white/black perspective coverage.
    // -------------------------------------------------------------------------
    std::vector<TrainingPosition> Trainer::colorSwapMirrorData(
        const std::vector<TrainingPosition>& data)
    {
        std::vector<TrainingPosition> result;
        result.reserve(data.size() * 2);
        result.insert(result.end(), data.begin(), data.end());
        for (const auto& pos : data) {
            result.push_back(colorSwapMirror(pos));
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // train: backpropagation training loop with Adam optimizer
    // CHANGED: SCReLU activation in forward/backward pass
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
                    params[idx++] = (*net.L1_weights)[f][j];
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
                    (*net.L1_weights)[f][j] = params[idx++];
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

        bool usePhaseBalance = config.phaseBalancedTraining &&
                               !openingIdx.empty() && !middlegameIdx.empty() && !endgameIdx.empty();

        std::vector<int> indices;

        float lr = config.learningRate;

        // Early stopping state
        float bestLoss = 1e9f;
        int epochsWithoutImprovement = 0;

        // Pre-allocate gradient buffer once outside the epoch loop — zeroed at the start of each batch
        std::vector<float> grads(totalParams, 0.0f);
        // Unpack weights into net struct before first batch
        unpackWeights(params);

        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            if (cancelFlag && cancelFlag->load()) break;

            // Build index set for this epoch
            if (usePhaseBalance) {
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

                std::fill(grads.begin(), grads.end(), 0.0f);
                float batchLoss = 0.0f;

                for (int bi = batchStart; bi < batchEnd; ++bi) {
                    const TrainingPosition& pos = data[indices[bi]];

                    // ---- FORWARD PASS (with SCReLU) ----
                    std::array<float, L1_SIZE> whiteAcc;
                    std::array<float, L1_SIZE> blackAcc;
                    for (int j = 0; j < L1_SIZE; ++j) {
                        whiteAcc[j] = net.L1_biases[j];
                        blackAcc[j] = net.L1_biases[j];
                    }

                    for (int feat : pos.activeFeatures) {
                        int mirFeat = mirrorFeature(feat);
                        for (int j = 0; j < L1_SIZE; ++j) {
                            whiteAcc[j] += (*net.L1_weights)[feat][j];
                        }
                        for (int j = 0; j < L1_SIZE; ++j) {
                            blackAcc[j] += (*net.L1_weights)[mirFeat][j];
                        }
                    }

                    // CHANGED: SCReLU activation for L1
                    std::array<float, L1_SIZE * 2> l1Out;
                    std::array<float, L1_SIZE * 2> l1Pre;  // pre-activation values needed for backward pass
                    const auto& stmAcc = (pos.sideToMove == Color::White) ? whiteAcc : blackAcc;
                    const auto& oppAcc = (pos.sideToMove == Color::White) ? blackAcc : whiteAcc;

                    for (int i = 0; i < L1_SIZE; ++i) {
                        l1Pre[i] = stmAcc[i];
                        l1Out[i] = screlu(stmAcc[i]);
                    }
                    for (int i = 0; i < L1_SIZE; ++i) {
                        l1Pre[L1_SIZE + i] = oppAcc[i];
                        l1Out[L1_SIZE + i] = screlu(oppAcc[i]);
                    }

                    // CHANGED: SCReLU activation for L2
                    std::array<float, L2_SIZE> l2Pre, l2Out;
                    for (int j = 0; j < L2_SIZE; ++j) {
                        float sum = net.L2_biases[j];
                        for (int i = 0; i < L1_SIZE * 2; ++i) {
                            sum += l1Out[i] * net.L2_weights[i][j];
                        }
                        l2Pre[j] = sum;
                        l2Out[j] = screlu(sum);
                    }

                    // CHANGED: SCReLU activation for L3
                    std::array<float, L3_SIZE> l3Pre, l3Out;
                    for (int j = 0; j < L3_SIZE; ++j) {
                        float sum = net.L3_biases[j];
                        for (int i = 0; i < L2_SIZE; ++i) {
                            sum += l2Out[i] * net.L3_weights[i][j];
                        }
                        l3Pre[j] = sum;
                        l3Out[j] = screlu(sum);
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

                    // ---- BACKWARD PASS (with SCReLU derivatives) ----
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

                    // CHANGED: SCReLU derivative for L3
                    std::array<float, L3_SIZE> dLdL3Pre;
                    for (int j = 0; j < L3_SIZE; ++j) {
                        dLdL3Pre[j] = dLdL3Out[j] * screlu_derivative(l3Pre[j]);
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

                    // CHANGED: SCReLU derivative for L2
                    std::array<float, L2_SIZE> dLdL2Pre;
                    for (int j = 0; j < L2_SIZE; ++j) {
                        dLdL2Pre[j] = dLdL2Out[j] * screlu_derivative(l2Pre[j]);
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

                    // CHANGED: SCReLU derivative for L1
                    std::array<float, L1_SIZE * 2> dLdL1Pre;
                    for (int i = 0; i < L1_SIZE * 2; ++i) {
                        dLdL1Pre[i] = dLdL1Out[i] * screlu_derivative(l1Pre[i]);
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
                // Unpack updated weights for the next batch's forward pass
                unpackWeights(params);

                // Check cancel flag every 50 batches for responsive cancellation
                if (cancelFlag && (numBatches % 50 == 0) && cancelFlag->load()) break;

            } // end batch loop

            lr *= config.lrDecay;

            float avgLoss = epochLoss / static_cast<float>(numBatches);

            if (progressCallback) {
                progressCallback(epoch + 1, avgLoss);
            }

            // Early stopping
            if (config.earlyStopPatience > 0) {
                if (avgLoss < bestLoss - 0.0001f) {
                    bestLoss = avgLoss;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= config.earlyStopPatience) {
                        if (progressCallback) {
                            progressCallback(epoch + 1, avgLoss);
                        }
                        break;
                    }
                }
            }
        } // end epoch loop

        unpackWeights(params);
        net.saveWeights(config.outputPath);
    }

    // -------------------------------------------------------------------------
    // trainDuck: backpropagation training loop for 832-feature DuckNNUE
    // Same algorithm as train() but uses DuckNNUE::NUM_FEATURES and duck mirror
    // -------------------------------------------------------------------------
    void Trainer::trainDuck(
        DuckNNUE::Network& net,
        const std::vector<TrainingPosition>& data,
        const TrainingConfig& config,
        std::function<void(int, float)> progressCallback,
        std::atomic<bool>* cancelFlag,
        std::function<void(int, int, float)> batchCallback)
    {
        if (data.empty()) {
            std::cerr << "No duck training data provided." << std::endl;
            return;
        }

        // Max positions cap: truncate data if needed (shuffle first for random sampling)
        std::vector<TrainingPosition> cappedData;
        const std::vector<TrainingPosition>* trainingData = &data;
        if (config.maxPositions > 0 && (int)data.size() > config.maxPositions) {
            cappedData = data;
            std::mt19937 capRng(42);
            std::shuffle(cappedData.begin(), cappedData.end(), capRng);
            cappedData.resize(static_cast<size_t>(config.maxPositions));
            trainingData = &cappedData;
            std::cerr << "[TrainDuck] Capped to " << config.maxPositions
                      << " positions (from " << data.size() << ")\n";
        }

        constexpr int DNUM = DuckNNUE::NUM_FEATURES;  // 832

        const int numL1w = DNUM * L1_SIZE;
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
            float* dst = params.data();
            std::memcpy(dst, net.L1_weights.data(), DNUM * L1_SIZE * sizeof(float));
            dst += DNUM * L1_SIZE;
            std::memcpy(dst, net.L1_biases.data(), L1_SIZE * sizeof(float));
            dst += L1_SIZE;
            std::memcpy(dst, net.L2_weights.data(), L1_SIZE * 2 * L2_SIZE * sizeof(float));
            dst += L1_SIZE * 2 * L2_SIZE;
            std::memcpy(dst, net.L2_biases.data(), L2_SIZE * sizeof(float));
            dst += L2_SIZE;
            std::memcpy(dst, net.L3_weights.data(), L2_SIZE * L3_SIZE * sizeof(float));
            dst += L2_SIZE * L3_SIZE;
            std::memcpy(dst, net.L3_biases.data(), L3_SIZE * sizeof(float));
            dst += L3_SIZE;
            std::memcpy(dst, net.output_weights.data(), L3_SIZE * sizeof(float));
            dst += L3_SIZE;
            *dst = net.output_bias;
        };

        auto unpackWeights = [&](const std::vector<float>& params) {
            // Use memcpy for contiguous array blocks — ~8x faster than element loops
            const float* src = params.data();
            // L1 weights: DNUM * L1_SIZE floats (contiguous: array<array<float,512>,832>)
            std::memcpy(net.L1_weights.data(), src, DNUM * L1_SIZE * sizeof(float));
            src += DNUM * L1_SIZE;
            std::memcpy(net.L1_biases.data(), src, L1_SIZE * sizeof(float));
            src += L1_SIZE;
            // L2 weights: (L1_SIZE*2) * L2_SIZE floats
            std::memcpy(net.L2_weights.data(), src, L1_SIZE * 2 * L2_SIZE * sizeof(float));
            src += L1_SIZE * 2 * L2_SIZE;
            std::memcpy(net.L2_biases.data(), src, L2_SIZE * sizeof(float));
            src += L2_SIZE;
            // L3 weights: L2_SIZE * L3_SIZE floats
            std::memcpy(net.L3_weights.data(), src, L2_SIZE * L3_SIZE * sizeof(float));
            src += L2_SIZE * L3_SIZE;
            std::memcpy(net.L3_biases.data(), src, L3_SIZE * sizeof(float));
            src += L3_SIZE;
            std::memcpy(net.output_weights.data(), src, L3_SIZE * sizeof(float));
            src += L3_SIZE;
            net.output_bias = *src;
            // Keep transposed weight caches in sync — used by AVX2 forward pass
            net.transposeWeights();
        };

        // Duck-aware mirror feature for perspective flip
        auto mirrorDuckFeature = [](int feat) -> int {
            return DuckNNUE::mirrorDuckFeature(feat);
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

        // Phase-balanced training: classify positions
        std::vector<int> openingIdx, middlegameIdx, endgameIdx;
        if (config.phaseBalancedTraining) {
            for (int i = 0; i < static_cast<int>(trainingData->size()); ++i) {
                GamePhase gp = classifyPhase((*trainingData)[i].activeFeatures);
                switch (gp) {
                    case GamePhase::Opening:    openingIdx.push_back(i); break;
                    case GamePhase::Middlegame: middlegameIdx.push_back(i); break;
                    case GamePhase::Endgame:    endgameIdx.push_back(i); break;
                }
            }
        }

        bool usePhaseBalance = config.phaseBalancedTraining &&
                               !openingIdx.empty() && !middlegameIdx.empty() && !endgameIdx.empty();

        std::vector<int> indices;
        float lr = config.learningRate;

        float bestLoss = 1e9f;
        int epochsWithoutImprovement = 0;

        // SWA: running average of weights starting from swaStart epoch
        std::vector<float> swaParams(totalParams, 0.0f);
        int swaCount = 0;

        // Total batch count for warmup/cosine scheduling
        int totalBatchesSoFar = 0;

        // Determine thread count once — hardware concurrency doesn't change between epochs
        int numTrainThreads = std::max(1, (int)std::thread::hardware_concurrency());
        numTrainThreads = std::min(numTrainThreads, 16);

        // Pre-allocate gradient buffers and loss accumulators outside epoch/batch loops
        std::vector<std::vector<float>> threadGrads(numTrainThreads, std::vector<float>(totalParams, 0.0f));
        std::vector<float> threadLoss(numTrainThreads, 0.0f);

        // ── Persistent thread pool ────────────────────────────────────────────
        // Threads are created once and reused for all batches in all epochs.
        // Each batch posts work via tStart_/tEnd_ and wakes workers via a
        // condition variable. Workers signal done via a barrier counter.
        // This eliminates ~178K thread create/destroy cycles per gen.
        struct BatchWork { int tStart = 0, tEnd = 0; };
        std::vector<BatchWork> threadWork(numTrainThreads);
        std::mutex              poolMtx;
        std::condition_variable poolWake;   // wakes workers when a batch is ready
        std::condition_variable poolDone;   // wakes the main thread when all workers done
        int poolPending  = 0;   // workers that haven't finished yet
        int poolEpoch    = 0;   // incremented each batch to distinguish "no work" from stale
        int poolCurEpoch = 0;   // workers read this to detect new work
        bool poolStop    = false;
        int poolNumChunks = 0;

        auto workerBodyFn = [&](int t) {
            int lastSeen = 0;
            for (;;) {
                // Wait for new batch work
                std::unique_lock<std::mutex> lk(poolMtx);
                poolWake.wait(lk, [&]{ return poolStop || poolCurEpoch != lastSeen; });
                if (poolStop) return;
                lastSeen = poolCurEpoch;
                int tStart = threadWork[t].tStart;
                int tEnd   = threadWork[t].tEnd;
                bool active = (t < poolNumChunks);
                lk.unlock();

                if (active) {
                    auto& tGrads = threadGrads[t];
                    float tLoss = 0.0f;

                    for (int bi = tStart; bi < tEnd; ++bi) {
                        const TrainingPosition& pos = (*trainingData)[indices[bi]];

                        // ---- FORWARD PASS (AVX2) ----
                        alignas(32) float whiteAcc[L1_SIZE];
                        alignas(32) float blackAcc[L1_SIZE];
                        std::memcpy(whiteAcc, net.L1_biases.data(), L1_SIZE * sizeof(float));
                        std::memcpy(blackAcc, net.L1_biases.data(), L1_SIZE * sizeof(float));

                        for (int feat : pos.activeFeatures) {
                            int mirFeat = mirrorDuckFeature(feat);
                            TrainAVX::avx_add_row<L1_SIZE>(whiteAcc, net.L1_weights[feat].data());
                            TrainAVX::avx_add_row<L1_SIZE>(blackAcc, net.L1_weights[mirFeat].data());
                        }

                        alignas(32) float l1Pre[L1_SIZE * 2];
                        alignas(32) float l1Out[L1_SIZE * 2];
                        const float* stmAcc = (pos.sideToMove == Color::White) ? whiteAcc : blackAcc;
                        const float* oppAcc = (pos.sideToMove == Color::White) ? blackAcc : whiteAcc;
                        TrainAVX::avx_screlu<L1_SIZE>(stmAcc, l1Pre,           l1Out);
                        TrainAVX::avx_screlu<L1_SIZE>(oppAcc, l1Pre + L1_SIZE, l1Out + L1_SIZE);

                        alignas(32) float l2Pre[L2_SIZE], l2Out[L2_SIZE];
                        TrainAVX::avx_gemv_T<L2_SIZE, L1_SIZE * 2>(
                            &net.L2_weights_T[0][0], net.L2_biases.data(), l1Out, l2Pre);
                        TrainAVX::avx_screlu<L2_SIZE>(l2Pre, l2Pre, l2Out);

                        alignas(32) float l3Pre[L3_SIZE], l3Out[L3_SIZE];
                        TrainAVX::avx_gemv_T<L3_SIZE, L2_SIZE>(
                            &net.L3_weights_T[0][0], net.L3_biases.data(), l2Out, l3Pre);
                        TrainAVX::avx_screlu<L3_SIZE>(l3Pre, l3Pre, l3Out);

                        float rawOutput = TrainAVX::avx_dot<L3_SIZE>(
                            net.output_weights.data(), l3Out) + net.output_bias;
                        float predicted = rawOutput * 400.0f;
                        float predictedWhitePOV = (pos.sideToMove == Color::White) ? predicted : -predicted;

                        // ---- LOSS ----
                        float sigPred   = sigmoid(predictedWhitePOV / config.evalScale);
                        float sigTarget = sigmoid(pos.searchEval / config.evalScale);
                        float result = pos.gameResult;
                        if (config.labelSmoothing > 0.0f)
                            result = result * (1.0f - config.labelSmoothing) + 0.5f * config.labelSmoothing;
                        float evalLoss   = (sigPred - sigTarget) * (sigPred - sigTarget);
                        float resultLoss = (sigPred - result)    * (sigPred - result);
                        float posLoss = config.lambda * evalLoss + (1.0f - config.lambda) * resultLoss;
                        float boost = 1.0f;
                        if (config.mateBoost > 1.0f && std::abs(pos.searchEval) > 300.0f) {
                            boost = 1.0f + (config.mateBoost - 1.0f) *
                                    std::min(1.0f, (std::abs(pos.searchEval) - 300.0f) / 700.0f);
                            posLoss *= boost;
                        }
                        tLoss += posLoss;

                        // ---- BACKWARD PASS (AVX2) ----
                        float dLdSigPred = (2.0f * config.lambda * (sigPred - sigTarget)
                                         + 2.0f * (1.0f - config.lambda) * (sigPred - result)) * boost;
                        float dSigPred_dPredW = sigPred * (1.0f - sigPred) / config.evalScale;
                        float dLdPredW  = dLdSigPred * dSigPred_dPredW;
                        float dLdPred   = (pos.sideToMove == Color::White) ? dLdPredW : -dLdPredW;
                        float dLdRawOut = dLdPred * 400.0f;

                        TrainAVX::avx_axpy<L3_SIZE>(tGrads.data() + offOutW, l3Out, dLdRawOut);
                        tGrads[offOutB] += dLdRawOut;

                        alignas(32) float dLdL3Out[L3_SIZE];
                        for (int i = 0; i < L3_SIZE; ++i)
                            dLdL3Out[i] = dLdRawOut * net.output_weights[i];

                        alignas(32) float dLdL3Pre[L3_SIZE];
                        TrainAVX::avx_screlu_deriv_mul<L3_SIZE>(dLdL3Out, l3Pre, dLdL3Pre);

                        TrainAVX::avx_outer_add<L2_SIZE, L3_SIZE>(tGrads.data() + offL3w, l2Out, dLdL3Pre);
                        TrainAVX::avx_axpy<L3_SIZE>(tGrads.data() + offL3b, dLdL3Pre, 1.0f);

                        alignas(32) float dLdL2Out[L2_SIZE] = {};
                        TrainAVX::avx_matvec_T_add<L2_SIZE, L3_SIZE>(dLdL2Out, &net.L3_weights[0][0], dLdL3Pre);

                        alignas(32) float dLdL2Pre[L2_SIZE];
                        TrainAVX::avx_screlu_deriv_mul<L2_SIZE>(dLdL2Out, l2Pre, dLdL2Pre);

                        TrainAVX::avx_outer_add<L1_SIZE * 2, L2_SIZE>(tGrads.data() + offL2w, l1Out, dLdL2Pre);
                        TrainAVX::avx_axpy<L2_SIZE>(tGrads.data() + offL2b, dLdL2Pre, 1.0f);

                        alignas(32) float dLdL1Out[L1_SIZE * 2] = {};
                        TrainAVX::avx_matvec_T_add<L1_SIZE * 2, L2_SIZE>(dLdL1Out, &net.L2_weights[0][0], dLdL2Pre);

                        alignas(32) float dLdL1Pre[L1_SIZE * 2];
                        TrainAVX::avx_screlu_deriv_mul<L1_SIZE * 2>(dLdL1Out, l1Pre, dLdL1Pre);

                        const float* dLdWhiteAcc = (pos.sideToMove == Color::White)
                            ? dLdL1Pre : dLdL1Pre + L1_SIZE;
                        const float* dLdBlackAcc = (pos.sideToMove == Color::White)
                            ? dLdL1Pre + L1_SIZE : dLdL1Pre;

                        for (int j = 0; j < L1_SIZE; j += 8) {
                            __m256 w = _mm256_add_ps(
                                _mm256_loadu_ps(dLdWhiteAcc + j),
                                _mm256_loadu_ps(dLdBlackAcc + j));
                            __m256 gg = _mm256_loadu_ps(tGrads.data() + offL1b + j);
                            _mm256_storeu_ps(tGrads.data() + offL1b + j, _mm256_add_ps(gg, w));
                        }

                        for (int feat : pos.activeFeatures) {
                            int mirFeat = mirrorDuckFeature(feat);
                            TrainAVX::avx_axpy<L1_SIZE>(
                                tGrads.data() + offL1w + feat    * L1_SIZE, dLdWhiteAcc, 1.0f);
                            TrainAVX::avx_axpy<L1_SIZE>(
                                tGrads.data() + offL1w + mirFeat * L1_SIZE, dLdBlackAcc, 1.0f);
                        }
                    } // end sample loop
                    threadLoss[t] = tLoss;
                }

                // Signal done
                {
                    std::lock_guard<std::mutex> lk2(poolMtx);
                    --poolPending;
                    if (poolPending == 0) poolDone.notify_one();
                }
            }
        };

        // Launch persistent worker threads
        std::vector<std::thread> poolThreads;
        poolThreads.reserve(numTrainThreads);
        for (int t = 0; t < numTrainThreads; ++t)
            poolThreads.emplace_back(workerBodyFn, t);

        // Helper: dispatch a batch to the pool and wait for completion
        auto dispatchBatch = [&](int batchStart, int batchEnd) {
            int batchActualSize = batchEnd - batchStart;
            int chunkSize = std::max(1, batchActualSize / numTrainThreads);
            int numChunks = std::min(numTrainThreads,
                            (batchActualSize + chunkSize - 1) / chunkSize);

            // Zero the buffers for active chunks
            for (int t = 0; t < numChunks; ++t) {
                std::fill(threadGrads[t].begin(), threadGrads[t].end(), 0.0f);
                threadLoss[t] = 0.0f;
                threadWork[t].tStart = batchStart + t * chunkSize;
                threadWork[t].tEnd   = std::min(threadWork[t].tStart + chunkSize, batchEnd);
            }
            {
                std::lock_guard<std::mutex> lk(poolMtx);
                poolNumChunks = numChunks;
                poolPending   = numChunks;
                ++poolCurEpoch;
            }
            poolWake.notify_all();
            // Wait for all chunks to finish
            {
                std::unique_lock<std::mutex> lk(poolMtx);
                poolDone.wait(lk, [&]{ return poolPending == 0; });
            }
        };

        // Unpack initial weights before first batch
        unpackWeights(params);

        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            if (cancelFlag && cancelFlag->load()) break;

            if (usePhaseBalance) {
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
                indices.resize(trainingData->size());
                std::iota(indices.begin(), indices.end(), 0);
            }
            std::shuffle(indices.begin(), indices.end(), rng);

            float epochLoss = 0.0f;
            int numBatches = 0;

            for (int batchStart = 0; batchStart < static_cast<int>(indices.size()); batchStart += config.batchSize) {
                int batchEnd = std::min(batchStart + config.batchSize, static_cast<int>(indices.size()));
                int batchActualSize = batchEnd - batchStart;

                // Dispatch batch to persistent thread pool (no thread creation overhead)
                dispatchBatch(batchStart, batchEnd);
                int numChunks = poolNumChunks;  // set by dispatchBatch

                // Gradient reduction: sum all thread buffers into thread 0 using AVX2
                float batchLoss = 0.0f;
                for (int t = 0; t < numChunks; ++t) {
                    batchLoss += threadLoss[t];
                    if (t > 0)
                        TrainAVX::avx_reduce_add(threadGrads[0].data(), threadGrads[t].data(), totalParams);
                }
                std::vector<float>& grads = threadGrads[0];

                TrainAVX::avx_scale(grads.data(), 1.0f / static_cast<float>(batchActualSize), totalParams);

                epochLoss += batchLoss / static_cast<float>(batchActualSize);
                ++numBatches;
                totalBatchesSoFar++;

                // Gradient accumulation: only update every gradAccum batches
                int effectiveGradAccum = std::max(1, config.gradAccum);
                if (numBatches % effectiveGradAccum == 0 || batchEnd >= (int)indices.size()) {
                    // LR warmup: linear ramp over warmupSteps batches
                    float effectiveLr = lr;
                    if (config.warmupSteps > 0 && totalBatchesSoFar <= config.warmupSteps)
                        effectiveLr = lr * float(totalBatchesSoFar) / float(config.warmupSteps);

                    // AVX2 Adam update — replaces scalar adamUpdate() call
                    adamState.t++;
                    float bc1 = 1.0f - std::pow(0.9f,   static_cast<float>(adamState.t));
                    float bc2 = 1.0f - std::pow(0.999f, static_cast<float>(adamState.t));
                    TrainAVX::avx_adam_update(
                        params.data(), grads.data(),
                        adamState.m.data(), adamState.v.data(),
                        totalParams,
                        effectiveLr, 0.9f, 0.999f,
                        1.0f / bc1, 1.0f / bc2,
                        1e-8f, config.weightDecay);

                    TrainAVX::avx_fill_zero(grads.data(), totalParams);
                    // Unpack updated weights into net struct — only after an actual Adam step
                    unpackWeights(params);
                }

                if (cancelFlag && (numBatches % 50 == 0) && cancelFlag->load()) break;

                // Batch progress callback
                if (batchCallback) {
                    int totalBatches = ((int)indices.size() + config.batchSize - 1) / config.batchSize;
                    float curBatchLoss = batchLoss / static_cast<float>(batchActualSize);
                    batchCallback(numBatches, totalBatches, curBatchLoss);
                }

            } // end batch loop

            // LR schedule: cosine annealing or exponential decay
            if (config.cosineLr) {
                int T = (config.cosineT0 > 0) ? config.cosineT0 : config.epochs;
                int epochInCycle = epoch % T;
                lr = config.learningRate * 0.5f * (1.0f + std::cos(3.14159f * epochInCycle / T));
                lr = std::max(lr, 1e-6f);
            } else {
                lr *= config.lrDecay;
            }

            float avgLoss = epochLoss / static_cast<float>(numBatches);

            // SWA: accumulate weight average after swaStart epoch
            if (config.swa && epoch + 1 >= config.swaStart) {
                unpackWeights(params);
                // Running average: swaParams = (swaParams * n + params) / (n+1)
                swaCount++;
                for (int i = 0; i < totalParams; ++i)
                    swaParams[i] += (params[i] - swaParams[i]) / float(swaCount);
            }

            // Unpack weights before callback so caller can compute val loss
            unpackWeights(params);

            if (progressCallback) {
                progressCallback(epoch + 1, avgLoss);
            }

            // Early stopping
            if (config.earlyStopPatience > 0) {
                if (avgLoss < bestLoss - 0.0001f) {
                    bestLoss = avgLoss;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= config.earlyStopPatience) {
                        if (progressCallback) {
                            progressCallback(epoch + 1, avgLoss);
                        }
                        break;
                    }
                }
            }
        } // end epoch loop

        // Shut down the persistent thread pool
        {
            std::lock_guard<std::mutex> lk(poolMtx);
            poolStop = true;
        }
        poolWake.notify_all();
        for (auto& th : poolThreads) th.join();

        // Apply SWA weights if accumulated
        if (config.swa && swaCount > 0) {
            for (int i = 0; i < totalParams; ++i)
                params[i] = swaParams[i];
        }

        unpackWeights(params);
        net.saveWeights(config.outputPath);
    }

    // -------------------------------------------------------------------------
    // adamUpdate: AdamW optimizer (Adam + decoupled weight decay)
    // -------------------------------------------------------------------------
    void Trainer::adamUpdate(std::vector<float>& params, std::vector<float>& grads,
                             AdamState& state, float lr, float beta1, float beta2,
                             float weightDecay) {
        constexpr float epsilon = 1e-8f;
        state.t++;

        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(state.t));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(state.t));

        for (size_t i = 0; i < params.size(); ++i) {
            state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * grads[i];
            state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * grads[i] * grads[i];

            float mHat = state.m[i] / bc1;
            float vHat = state.v[i] / bc2;

            // AdamW: apply weight decay directly to params (decoupled from gradient)
            params[i] -= lr * mHat / (std::sqrt(vHat) + epsilon)
                       + lr * weightDecay * params[i];
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
            return data;
        }

        // Detect format: SelfPlayGen writes "NNUE" magic + version + count
        // Legacy format: raw uint32_t count with no header
        char magic[4] = {};
        file.read(magic, 4);
        if (!file) return data;

        uint32_t numPositions = 0;
        if (magic[0]=='N' && magic[1]=='N' && magic[2]=='U' && magic[3]=='E') {
            // SelfPlayGen format: skip 1-byte version, then read count
            uint8_t version = 0;
            file.read(reinterpret_cast<char*>(&version), 1);
            file.read(reinterpret_cast<char*>(&numPositions), sizeof(numPositions));
        } else {
            // Legacy format: magic bytes are actually the first 4 bytes of count
            std::memcpy(&numPositions, magic, 4);
        }

        data.reserve(numPositions);

        for (uint32_t i = 0; i < numPositions; ++i) {
            TrainingPosition pos;

            uint16_t numFeatures = 0;
            file.read(reinterpret_cast<char*>(&numFeatures), sizeof(numFeatures));
            if (!file || numFeatures > 900) break;  // sanity check (max 832 duck features)

            pos.activeFeatures.resize(numFeatures);
            for (uint16_t j = 0; j < numFeatures; ++j) {
                uint16_t f = 0;
                file.read(reinterpret_cast<char*>(&f), sizeof(f));
                pos.activeFeatures[j] = static_cast<int>(f);
            }

            uint8_t stm = 0;
            file.read(reinterpret_cast<char*>(&stm), sizeof(stm));
            pos.sideToMove = (stm == 0) ? Color::White : Color::Black;

            // SelfPlayGen format: result then eval
            // Legacy format: gameResult then searchEval (same layout)
            file.read(reinterpret_cast<char*>(&pos.gameResult), sizeof(pos.gameResult));
            file.read(reinterpret_cast<char*>(&pos.searchEval), sizeof(pos.searchEval));

            if (!file) break;
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
                MoveList moves; MoveGen::getLegalMoves(board, moves);

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
                board.recomputeBitboards();
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


