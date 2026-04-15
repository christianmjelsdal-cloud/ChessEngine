// ARCHITECTURE FIX M-2: Split from VisualGame.cpp — all rendering and drawing
// functions (board, pieces, highlights, eval bars, HUD, status, arrows, etc.)
//
#include "VisualGame.h"
#include "Syzygy.h"
#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <chrono>

// -------------------------------------------------------
// formatCountdown — file-local helper for ETA display in HUD
// -------------------------------------------------------
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

// =======================================================
//  R E N D E R I N G
// =======================================================

void VisualGame::render() {
    // ── Dynamic layout: scale board to fit window ──────────────────────────
    {
        auto sz = window.getSize();
        const int leftMargin = OX;
        const int rightMargin = OX;
        const int topMargin  = OY;
        const int botMargin  = OY + 60;
        int availW = static_cast<int>(sz.x) - leftMargin - rightMargin;
        int availH = static_cast<int>(sz.y) - topMargin - botMargin;
        if (isAutomateChess_ && !board.automateSetupComplete)
            availW -= SETUP_PANEL_W;
        int sq = std::min(availW / 8, availH / 8);
        dynSQ_ = std::max(sq, 40);
        dynOX_ = leftMargin;
        // Scale factor relative to base square size (80px)
        scale_  = float(dynSQ_) / float(SQ);
    }
    // Derived scaled values used throughout draw functions
    const int  sBW  = std::max(4, int(EVAL_BAR_W * scale_));  // scaled eval bar width
    const int  sGap = std::max(4, int(28 * scale_));           // gap between bar and board
    const int  sOY  = int(OY * scale_);                        // scaled top margin (for header text)
    const auto sfs  = [&](int base) -> unsigned int {          // scaled font size
        return static_cast<unsigned int>(std::max(8, int(base * scale_)));
    };

    window.clear(sf::Color(40, 40, 40));
    drawBoard();
    drawHighlights();
    // Automate Chess setup overlay (valid placement squares)
    if (isAutomateChess_ && !board.automateSetupComplete)
        drawAutomateSetupOverlay();
    drawPieces();
    if (showArrows && (!analysisMode_ ? (engineThinking || botHasPendingMove_) : analysisDepth_ > 0))
        drawPVArrows();
    if (botVsNNUE_)
        drawDualEvalBars();
    else
        drawEvalBar();
    drawCoordinates();
    drawStatus();
    drawHUD();
    // Analysis panel (right side) — always drawn when history exists
    if (!gameHistory_.empty())
        drawMoveList();
    if (analysisMode_)
        drawAnalysisOverlay();
    // Automate Chess setup panel (right side)
    if (isAutomateChess_ && !board.automateSetupComplete)
        drawAutomateSetupPanel();

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
            sf::RectangleShape cell({float(dynSQ_), float(dynSQ_)});
            cell.setPosition({float(dynOX_ + c * dynSQ_), float(OY + (7 - r) * dynSQ_)});
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
            sf::RectangleShape h({float(dynSQ_), float(dynSQ_)});
            h.setPosition({float(dynOX_ + s.col * dynSQ_), float(OY + (7 - s.rank) * dynSQ_)});
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
                    sf::RectangleShape h({float(dynSQ_), float(dynSQ_)});
                    h.setPosition({float(dynOX_ + c * dynSQ_), float(OY + (7 - r) * dynSQ_)});
                    h.setFillColor(duckHighlight);
                    window.draw(h);
                }
            }
        }
        return; // Don't draw piece selection highlights while placing duck
    }

    if (!pieceSelected) return;

    sf::RectangleShape sel({float(dynSQ_), float(dynSQ_)});
    sel.setPosition({float(dynOX_ + selectedSq.col * dynSQ_),
                     float(OY + (7 - selectedSq.rank) * dynSQ_)});
    sel.setFillColor(sf::Color(255, 255, 0, 100));
    window.draw(sel);

    for (const auto& m : selectedMoves) {
        float cx = dynOX_ + m.to.col * dynSQ_ + dynSQ_ / 2.f;
        float cy = OY + (7 - m.to.rank) * dynSQ_ + dynSQ_ / 2.f;

        bool isCapture = !board.getPiece(m.to).isNone();
        if (!isCapture && board.getPiece(m.from).type == PieceType::Pawn
            && m.to.col != m.from.col)
            isCapture = true;

        if (isCapture) {
            float r = dynSQ_ / 2.f;
            sf::CircleShape ring(r);
            ring.setOrigin({r, r});
            ring.setPosition({cx, cy});
            ring.setFillColor(sf::Color::Transparent);
            ring.setOutlineThickness(-5.f);
            ring.setOutlineColor(sf::Color(0, 0, 0, 80));
            window.draw(ring);
        }
        else {
            float r = dynSQ_ / 6.f;
            sf::CircleShape dot(r);
            dot.setOrigin({r, r});
            dot.setPosition({cx, cy});
            dot.setFillColor(sf::Color(0, 0, 0, 80));
            window.draw(dot);
        }
    }
}

// -------------------------------------------------------
// DRAW PV ARROWS — engine's live thought line + analysis arrows
// -------------------------------------------------------
void VisualGame::drawPVArrows() {
    // ── Analysis mode: show the analysis engine's best move arrow
    if (analysisMode_) {
        auto topMoves = analysisEngine_.getTopRootMoves(1);
        if (!topMoves.empty() && topMoves[0].move.from.isValid()) {
            drawArrow(squareCenter(topMoves[0].move.from), squareCenter(topMoves[0].move.to),
                      sf::Color(80, 220, 80, 200));
        }

        // Threat arrow: show opponent's best response in red
        if (viewIdx_ >= 0 && viewIdx_ < (int)gameHistory_.size()) {
            const Board& vb = viewBoard();
            // Quick 1-ply: find the best capture/check for the opponent
            Board oppBoard = vb;
            oppBoard.turn = (vb.turn == Color::White) ? Color::Black : Color::White;
            MoveList oppMoves; MoveGen::getLegalMoves(oppBoard, oppMoves);
            // Find highest-value capture as the "threat"
            Move threatMove{};
            int bestVal = -1;
            for (const auto& m : oppMoves) {
                Piece cap = oppBoard.getPiece(m.to);
                if (!cap.isNone() && !cap.isDuck()) {
                    int val = 0;
                    switch (cap.type) {
                        case PieceType::Queen:  val = 9; break;
                        case PieceType::Rook:   val = 5; break;
                        case PieceType::Bishop:
                        case PieceType::Knight: val = 3; break;
                        case PieceType::Pawn:   val = 1; break;
                        default: break;
                    }
                    if (val > bestVal) { bestVal = val; threatMove = m; }
                }
            }
            if (threatMove.from.isValid() && bestVal > 0) {
                drawArrow(squareCenter(threatMove.from), squareCenter(threatMove.to),
                          sf::Color(220, 60, 60, 140));
            }
        }
        return;
    }

    // Live game: show engine's PV
    std::vector<Move> pv;
    if (activeEngine_)
        pv = activeEngine_->getLivePV();
    else if (botHasPendingMove_)
        pv = cachedPV_;

    if (pv.empty()) return;

    int maxArrows = std::min((int)pv.size(), 5);
    for (int i = 0; i < maxArrows; i++) {
        const Move& m = pv[i];
        if (!m.from.isValid() || !m.to.isValid()) break;

        sf::Vector2f from = squareCenter(m.from);
        sf::Vector2f to   = squareCenter(m.to);

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
    const Board& b = viewBoard();  // use historical board in analysis mode
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (!analysisMode_ && isDragging && dragFrom.rank == r && dragFrom.col == c)
                continue;
            if (!analysisMode_ && isAnimating && animMove.from.rank == r && animMove.from.col == c)
                continue;

            Piece p = b.squares[r][c];
            if (p.isNone()) continue;

            float px = float(dynOX_ + c * dynSQ_);
            float py = float(OY + (7 - r) * dynSQ_);

#ifdef DUCK_CHESS
            if (p.isDuck()) {
                sf::Sprite sprite(duckTexture_);
                sf::Vector2u ts = duckTexture_.getSize();
                float scale = float(dynSQ_) / float(std::max(ts.x, ts.y));
                sprite.setScale({scale, scale});
                sprite.setPosition({px, py});
                window.draw(sprite);
                continue;
            }
#endif

            int ci = (p.color == Color::White) ? 0 : 1;
            int pi = static_cast<int>(p.type) - 1;
            if (pi < 0 || pi >= 6) continue;

            sf::Sprite sprite(pieceTextures[ci][pi]);
            sf::Vector2u ts = pieceTextures[ci][pi].getSize();
            float scale = float(dynSQ_) / float(std::max(ts.x, ts.y));
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
    if (pi < 0 || pi >= 6) return;  // FIX 10.3: bounds check on piece type index

    sf::Sprite sprite(pieceTextures[ci][pi]);
    sf::Vector2u ts = pieceTextures[ci][pi].getSize();
    float scale = float(dynSQ_) / float(std::max(ts.x, ts.y));
    sprite.setScale({scale, scale});
    sprite.setPosition({dragPos.x - dynSQ_ / 2.f, dragPos.y - dynSQ_ / 2.f});
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
    if (pi < 0 || pi >= 6) return;  // FIX 10.2: bounds check on piece type index

    sf::Sprite sprite(pieceTextures[ci][pi]);
    sf::Vector2u ts = pieceTextures[ci][pi].getSize();
    float scale = float(dynSQ_) / float(std::max(ts.x, ts.y));
    sprite.setScale({scale, scale});
    sprite.setPosition({x, y});
    window.draw(sprite);
}

// -------------------------------------------------------
// DRAW PROMOTION DIALOG
// -------------------------------------------------------
void VisualGame::drawPromotionDialog() {
    sf::RectangleShape overlay({float(dynSQ_ * 8), float(dynSQ_ * 8)});
    overlay.setPosition({float(dynOX_), float(OY)});
    overlay.setFillColor(sf::Color(0, 0, 0, 120));
    window.draw(overlay);

    bool isWhite = (board.turn == Color::White);
    int ci = isWhite ? 0 : 1;
    int startRank = isWhite ? 7 : 0;
    int dir = isWhite ? -1 : 1;

    PieceType choices[] = { PieceType::Queen, PieceType::Knight, PieceType::Rook, PieceType::Bishop };

    for (int i = 0; i < 4; i++) {
        int rank = startRank + dir * i;
        float px = float(dynOX_ + promoTo.col * dynSQ_);
        float py = float(OY + (7 - rank) * dynSQ_);

        sf::RectangleShape cell({float(dynSQ_), float(dynSQ_)});
        cell.setPosition({px, py});
        cell.setFillColor(sf::Color(240, 240, 230));
        cell.setOutlineThickness(1.f);
        cell.setOutlineColor(sf::Color(100, 100, 100));
        window.draw(cell);

        int pi = static_cast<int>(choices[i]) - 1;
        sf::Sprite sprite(pieceTextures[ci][pi]);
        sf::Vector2u ts = pieceTextures[ci][pi].getSize();
        float scale = float(dynSQ_) / float(std::max(ts.x, ts.y));  // FIX 10.6: uniform scaling
        sprite.setScale({scale, scale});
        sprite.setPosition({px, py});
        window.draw(sprite);
    }
}

// -------------------------------------------------------
// DRAW EVAL BAR
// -------------------------------------------------------
void VisualGame::drawEvalBar() {
    // Eval bar sits to the left of the board
    float barX = float(dynOX_ - EVAL_BAR_W - 28);
    float barY = float(OY);
    float barH = float(dynSQ_ * 8);
    float barW = float(EVAL_BAR_W);

    // Get eval: analysis engine when in analysis mode, live engine otherwise
    int eval = lastEval_;
    if (analysisMode_) {
        eval = analysisDepth_ > 0 ? analysisEval_ : 0;
    } else if (activeEngine_ && activeEngine_->getLiveDepth() > 0) {
        eval = activeEngine_->getLiveEval();
    }

    // Detect mate scores: MATE_SCORE (100000) - ply
    bool isMate = std::abs(eval) > (Engine::MATE_SCORE - 500);
    int mateMoves = 0;
    if (isMate) {
        int ply = Engine::MATE_SCORE - std::abs(eval);
        mateMoves = (ply + 1) / 2;  // convert ply to full moves
        if (mateMoves < 1) mateMoves = 1;
    }

    // Clamp eval to a display range: sigmoid-like mapping for smooth bar
    // Convert centipawns to a 0-1 ratio (0.5 = even, 1.0 = white winning)
    float ratio;
    if (isMate) {
        ratio = (eval > 0) ? 0.98f : 0.02f;  // slam bar to the winning side
    } else {
        ratio = 0.5f + 0.5f * (eval / 500.0f);   // linear scale, ±500cp = full bar
        if (ratio > 0.98f) ratio = 0.98f;
        if (ratio < 0.02f) ratio = 0.02f;
    }

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
    std::string evalStr;
    if (isMate) {
        // Show "M3" for white delivering mate in 3, "-M3" for black
        evalStr = (eval > 0 ? "M" : "-M") + std::to_string(mateMoves);
    } else {
        float absEval = std::abs(eval / 100.0f);
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

// -------------------------------------------------------
// SINGLE EVAL BAR — reusable for dual bar mode
// -------------------------------------------------------
void VisualGame::drawSingleEvalBar(float barX, float barY, float barW, float barH,
                                    int eval, const std::string& label, sf::Color labelColor) {
    // Detect mate scores
    bool isMate = std::abs(eval) > (Engine::MATE_SCORE - 500);
    int mateMoves = 0;
    if (isMate) {
        int ply = Engine::MATE_SCORE - std::abs(eval);
        mateMoves = (ply + 1) / 2;
        if (mateMoves < 1) mateMoves = 1;
    }

    // Clamp eval to display range
    float ratio;
    if (isMate) {
        ratio = (eval > 0) ? 0.98f : 0.02f;
    } else {
        ratio = 0.5f + 0.5f * (eval / 500.0f);
        if (ratio > 0.98f) ratio = 0.98f;
        if (ratio < 0.02f) ratio = 0.02f;
    }

    // Background (black side)
    sf::RectangleShape blackSide({barW, barH});
    blackSide.setPosition({barX, barY});
    blackSide.setFillColor(sf::Color(50, 50, 50));
    window.draw(blackSide);

    // White side
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

    // Eval number
    std::string evalStr;
    if (isMate) {
        evalStr = (eval > 0 ? "M" : "-M") + std::to_string(mateMoves);
    } else {
        float absEval = std::abs(eval / 100.0f);
        std::ostringstream oss;
        oss << std::fixed;
        oss.precision(1);
        if (eval >= 0) oss << "+" << absEval;
        else oss << "-" << absEval;
        evalStr = oss.str();
    }

    sf::Text evalText(font, evalStr, 14);
    auto textBounds = evalText.getLocalBounds();
    float textW = textBounds.size.x;
    float textH = textBounds.size.y;
    float textX = barX + (barW - textW) / 2.f;
    float textY;
    sf::Color textColor;
    sf::Color pillColor;

    if (ratio >= 0.5f) {
        textY = barY + 6;
        textColor = sf::Color(235, 235, 235);
        pillColor = sf::Color(30, 30, 30, 160);
    } else {
        textY = barY + barH - textH - 14;
        textColor = sf::Color(30, 30, 30);
        pillColor = sf::Color(235, 235, 235, 160);
    }

    float pillPad = 2.f;
    sf::RectangleShape pill({textW + pillPad * 2, textH + pillPad * 2 + 4});
    pill.setPosition({textX - pillPad, textY - pillPad + 2});
    pill.setFillColor(pillColor);
    window.draw(pill);

    evalText.setFillColor(textColor);
    evalText.setPosition({textX, textY});
    window.draw(evalText);

    // Label at top
    if (!label.empty()) {
        sf::Text lbl(font, label, 10);
        auto lblBounds = lbl.getLocalBounds();
        float lblX = barX + (barW - lblBounds.size.x) / 2.f;
        lbl.setPosition({lblX, barY - 14});
        lbl.setFillColor(labelColor);
        window.draw(lbl);
    }
}

// -------------------------------------------------------
// DUAL EVAL BARS — shown in Bot vs NNUE mode
// -------------------------------------------------------
void VisualGame::drawDualEvalBars() {
    float barH = float(dynSQ_ * 8);
    float barW = float(EVAL_BAR_W);
    float barY = float(OY);

    // Left bar: White's engine eval
    float leftX = float(dynOX_ - EVAL_BAR_W - 28);
    // Right bar: Black's engine eval
    float rightX = float(dynOX_ + dynSQ_ * 8 + 28);

    // Determine which eval belongs to which side
    // engine_ = classical side, engine2_ = NNUE side
    int whiteEval, blackEval;
    std::string whiteLabel, blackLabel;

    if (nnueOnWhite_) {
        // engine2_ (NNUE) plays White, engine_ (Classical) plays Black
        whiteEval = lastEval2_;
        blackEval = lastEval_;
        whiteLabel = "NNUE";
        blackLabel = "Classical";
    } else {
        // engine_ (Classical) plays White, engine2_ (NNUE) plays Black
        whiteEval = lastEval_;
        blackEval = lastEval2_;
        whiteLabel = "Classical";
        blackLabel = "NNUE";
    }

    // Live eval update for the currently thinking engine
    if (activeEngine_ && activeEngine_->getLiveDepth() > 0) {
        int liveVal = activeEngine_->getLiveEval();
        if (activeEngine_ == &engine2_) {
            if (nnueOnWhite_) whiteEval = liveVal;
            else blackEval = liveVal;
        } else {
            if (nnueOnWhite_) blackEval = liveVal;
            else whiteEval = liveVal;
        }
    }

    // Color-code labels: blue for Classical, green for NNUE
    sf::Color classicalColor(120, 160, 255);  // soft blue
    sf::Color nnueColor(100, 220, 120);       // soft green

    sf::Color wLabelColor = (whiteLabel == "NNUE") ? nnueColor : classicalColor;
    sf::Color bLabelColor = (blackLabel == "NNUE") ? nnueColor : classicalColor;

    drawSingleEvalBar(leftX,  barY, barW, barH, whiteEval, whiteLabel, wLabelColor);
    drawSingleEvalBar(rightX, barY, barW, barH, blackEval, blackLabel, bLabelColor);
}

void VisualGame::drawCoordinates() {
    for (int i = 0; i < 8; i++) {
        sf::Text rank(font, std::string(1, char('1' + i)), 14);
        rank.setFillColor(sf::Color(200, 200, 200));
        rank.setPosition({float(dynOX_ - 20), float(OY + (7 - i) * dynSQ_ + dynSQ_ / 2 - 7)});
        window.draw(rank);

        sf::Text file(font, std::string(1, char('a' + i)), 14);
        file.setFillColor(sf::Color(200, 200, 200));
        file.setPosition({float(dynOX_ + i * dynSQ_ + dynSQ_ / 2 - 4), float(OY + 8 * dynSQ_ + 5)});
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
    status.setPosition({float(dynOX_), float(OY + 8 * dynSQ_ + 28)});
    window.draw(status);
}

// -------------------------------------------------------
// DRAW HUD — controls info at the top
// -------------------------------------------------------
void VisualGame::drawHUD() {
    // ── Line 1: variant label (only when a variant is active) ──────────────
    if (isDuckChess_) {
        sf::Text varLabel(font, "DUCK CHESS", 15);
        varLabel.setFillColor(sf::Color(255, 220, 50));
        varLabel.setPosition({float(dynOX_), float(OY - 50)});
        window.draw(varLabel);
    } else if (isAutomateChess_) {
        std::string lbl = board.automateSetupComplete ? "AUTOMATE CHESS" : "AUTOMATE SETUP";
        sf::Text varLabel(font, lbl, 15);
        varLabel.setFillColor(sf::Color(100, 220, 255));
        varLabel.setPosition({float(dynOX_), float(OY - 50)});
        window.draw(varLabel);
    }

    // ── Line 2: mode + hotkeys ──────────────────────────────────────────────
    std::string hudText;
    std::string sideTag;
    if (sideConfig_ == SideConfig::Swapped) sideTag = " [SWAPPED]";
    else if (sideConfig_ == SideConfig::Random) sideTag = " [RANDOM]";

    if (botVsNNUE_) {
        hudText = "BOT vs NNUE";
        if (botPaused) hudText += "  [PAUSED]";
        if (fastMode)  hudText += "  [FAST]";
        hudText += sideTag;
        std::string w = nnueOnWhite_ ? "NNUE" : "Classical";
        std::string b = nnueOnWhite_ ? "Classical" : "NNUE";
        hudText += "  |  W=" + w + "  B=" + b;
        hudText += "  |  [Space] Pause  [S] Sides  [A] Arrows  [F] Fast  [R] Reset  [V] Exit";
    } else if (botVsBot) {
        hudText = "BOT vs BOT";
        if (botPaused)    hudText += "  [PAUSED]";
        if (fastMode)     hudText += "  [FAST]";
        if (nnueEnabled_) hudText += "  [NNUE]";
        hudText += "  |  [Space] Pause  [A] Arrows  [F] Fast  [D] Duck  [M] Automate  [R] Reset  [B] Exit";
    } else {
        hudText = "PLAYER vs ENGINE";
        if (nnueEnabled_) hudText += "  [NNUE]";
        std::string ec = (engineColor == Color::White) ? "White" : "Black";
        hudText += sideTag + "  |  Engine=" + ec;
        hudText += "  |  [B] Bot  [S] Sides  [A] Arrows  [D] Duck  [M] Automate  [R] Reset  [V] Bot vs NNUE";
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
    if (!gameHistory_.empty())
        hudText += "  [Z] Analyse  [Ctrl+S] PGN";

    // Scale font size down if the window is wide (maximized) to keep text on one line
    unsigned int winW = window.getSize().x;
    unsigned int hudFontSize = (winW > 1200) ? 12 : 13;

    sf::Text hud(font, hudText, hudFontSize);
    hud.setFillColor(sf::Color(180, 180, 180));
    hud.setPosition({float(dynOX_), float(OY - 26)});
    window.draw(hud);

    // ── NNUE status line (below board) ──────────────────────────────────────
    std::string displayStatus;
    {
        std::lock_guard<std::mutex> lk(nnueStatusMutex_);
        displayStatus = nnueStatus_;
    }
    if (!displayStatus.empty()) {
        auto etaEnd = nnueETAEndMs_.load();
        if (etaEnd > 0) displayStatus += formatCountdown(etaEnd);
        sf::Text nnueTxt(font, displayStatus, 14);
        nnueTxt.setFillColor(sf::Color(100, 200, 255));
        nnueTxt.setPosition({float(dynOX_), float(OY + 8 * dynSQ_ + 50)});
        window.draw(nnueTxt);
    }
}


// -------------------------------------------------------
// -------------------------------------------------------
// DRAW MOVE LIST — analysis panel (right of board)
// -------------------------------------------------------
void VisualGame::drawMoveList() {
    if (gameHistory_.empty()) return;

    auto sz = window.getSize();
    const float SB_W   = std::max(6.f, 8.f * scale_);   // scrollbar width
    const float PAD    = 4.f;
    float panelX = float(dynOX_ + dynSQ_ * 8 + 12);
    float panelY = float(OY);
    float panelW = float(sz.x) - panelX - 8.f;
    if (panelW < 80.f) return;

    // Split panel: move list on top, analysis stats + graph below
    // Move list gets ~55% of board height, rest for stats/graph
    float boardH     = float(dynSQ_ * 8);
    float listH      = boardH * 0.55f;   // move list area height
    float statsAreaY = panelY + listH;
    float statsAreaH = boardH - listH;

    unsigned int fs  = std::max(11u, unsigned(13 * scale_));
    float lineH      = float(fs) + 5.f;
    float headerH    = lineH + 2.f;
    float contentH   = listH - headerH;          // drawable rows area
    int   maxVisible = std::max(1, int(contentH / lineH));

    int total        = int(gameHistory_.size());
    int highlighted  = analysisMode_ ? viewIdx_ : total - 1;
    int totalRows    = (total + 1) / 2;

    // Auto-scroll: keep highlighted row centred
    int highlightedRow = highlighted / 2;
    int scrollOffset   = std::max(0, highlightedRow - maxVisible / 2);
    scrollOffset       = std::min(scrollOffset, std::max(0, totalRows - maxVisible));

    // ── Panel background ────────────────────────────────────────────────────
    sf::RectangleShape bg({panelW, boardH});
    bg.setPosition({panelX, panelY});
    bg.setFillColor(sf::Color(28, 28, 36));
    bg.setOutlineColor(sf::Color(60, 60, 80));
    bg.setOutlineThickness(1.f);
    window.draw(bg);

    // Divider between list and stats
    sf::RectangleShape divider({panelW, 1.f});
    divider.setPosition({panelX, statsAreaY});
    divider.setFillColor(sf::Color(60, 60, 80));
    window.draw(divider);

    // ── Header + nav buttons ────────────────────────────────────────────────
    int   currentPos  = analysisMode_ ? viewIdx_ : total - 1;  // -1 = live (past end)
    bool  canGoBack   = total > 0 && (analysisMode_ ? viewIdx_ > 0 : total > 0);
    bool  canGoFwd    = analysisMode_;  // can go forward only when in analysis mode

    // Nav button geometry (stored for click handling)
    float btnH   = headerH - 4.f;
    float btnW   = btnH * 1.1f;
    float btnY   = panelY + 2.f;
    float btnBackX = panelX + PAD;
    float btnFwdX  = btnBackX + btnW + 4.f;

    // Back button  <
    {
        sf::Color col = canGoBack ? sf::Color(180, 180, 220) : sf::Color(60, 60, 80);
        sf::ConvexShape tri(3);
        float cx = btnBackX + btnW / 2.f, cy = btnY + btnH / 2.f;
        tri.setPoint(0, {cx + btnW * 0.35f, cy - btnH * 0.4f});
        tri.setPoint(1, {cx + btnW * 0.35f, cy + btnH * 0.4f});
        tri.setPoint(2, {cx - btnW * 0.35f, cy});
        tri.setFillColor(col);
        window.draw(tri);
    }
    // Forward button  >
    {
        sf::Color col = canGoFwd ? sf::Color(180, 180, 220) : sf::Color(60, 60, 80);
        sf::ConvexShape tri(3);
        float cx = btnFwdX + btnW / 2.f, cy = btnY + btnH / 2.f;
        tri.setPoint(0, {cx - btnW * 0.35f, cy - btnH * 0.4f});
        tri.setPoint(1, {cx - btnW * 0.35f, cy + btnH * 0.4f});
        tri.setPoint(2, {cx + btnW * 0.35f, cy});
        tri.setFillColor(col);
        window.draw(tri);
    }

    // "MOVES" label centred in remaining header space
    sf::Text header(font, "MOVES", std::max(10u, unsigned(11 * scale_)));
    header.setFillColor(sf::Color(120, 120, 160));
    float headerTextX = btnFwdX + btnW + 6.f;
    header.setPosition({headerTextX, panelY + 3});
    window.draw(header);

    // ── Scrollbar ───────────────────────────────────────────────────────────
    float sbX      = panelX + panelW - SB_W - 2.f;
    float sbTrackY = panelY + headerH;
    float sbTrackH = contentH;

    // Track
    sf::RectangleShape sbTrack({SB_W, sbTrackH});
    sbTrack.setPosition({sbX, sbTrackY});
    sbTrack.setFillColor(sf::Color(40, 40, 55));
    window.draw(sbTrack);

    // Thumb
    if (totalRows > maxVisible) {
        float thumbRatio = float(maxVisible) / float(totalRows);
        float thumbH     = std::max(16.f, sbTrackH * thumbRatio);
        float thumbY     = sbTrackY + (sbTrackH - thumbH) *
                           float(scrollOffset) / float(totalRows - maxVisible);
        sf::RectangleShape sbThumb({SB_W, thumbH});
        sbThumb.setPosition({sbX, thumbY});
        sbThumb.setFillColor(sf::Color(90, 90, 130));
        sbThumb.setOutlineColor(sf::Color(120, 120, 180));
        sbThumb.setOutlineThickness(1.f);
        window.draw(sbThumb);
    }

    // ── Clip move list to panel using SFML viewport ─────────────────────────
    sf::View fullView = window.getView();
    {
        float wx = float(sz.x), wy = float(sz.y);
        float clipX = panelX;
        float clipY = panelY + headerH;
        float clipW = panelW - SB_W - 4.f;
        float clipH = contentH;

        sf::View clipView(sf::FloatRect({clipX, clipY}, {clipW, clipH}));
        clipView.setViewport(sf::FloatRect(
            {clipX / wx, clipY / wy},
            {clipW / wx, clipH / wy}));
        window.setView(clipView);

        float startY  = clipY;
        float colW    = (clipW - float(fs) * 2.2f - PAD) / 2.f;
        float moveStartX = clipX + PAD + float(fs) * 2.2f;

        for (int row = scrollOffset; row < totalRows && (row - scrollOffset) < maxVisible + 1; ++row) {
            float rowY = startY + float(row - scrollOffset) * lineH;

            int wIdx = row * 2;
            int bIdx = row * 2 + 1;

            // Move number
            sf::Text numTxt(font, std::to_string(row + 1) + ".", fs);
            numTxt.setFillColor(sf::Color(100, 100, 130));
            numTxt.setPosition({clipX + PAD, rowY});
            window.draw(numTxt);

            auto cpColor = [](int loss, bool isBlack) -> sf::Color {
                if (loss < 0)    return isBlack ? sf::Color(180,180,190) : sf::Color(210,210,220);
                if (loss <= 5)   return sf::Color(100, 220, 100);
                if (loss <= 20)  return sf::Color(180, 220, 100);
                if (loss <= 50)  return sf::Color(220, 200, 80);
                if (loss <= 100) return sf::Color(220, 140, 60);
                return sf::Color(220, 70, 70);
            };

            // White move
            if (wIdx < total) {
                bool hi = (wIdx == highlighted);
                if (hi) {
                    sf::RectangleShape hiBg({colW - 2.f, lineH - 1.f});
                    hiBg.setPosition({moveStartX, rowY});
                    hiBg.setFillColor(sf::Color(70, 110, 180, 200));
                    window.draw(hiBg);
                }
                sf::Text t(font, gameHistory_[wIdx].moveAlg, fs);
                t.setFillColor(hi ? sf::Color::White : cpColor(gameHistory_[wIdx].cpLoss, false));
                t.setPosition({moveStartX + 3, rowY});
                window.draw(t);
            }

            // Black move
            if (bIdx < total) {
                bool hi = (bIdx == highlighted);
                float bX = moveStartX + colW;
                if (hi) {
                    sf::RectangleShape hiBg({colW - 2.f, lineH - 1.f});
                    hiBg.setPosition({bX, rowY});
                    hiBg.setFillColor(sf::Color(70, 110, 180, 200));
                    window.draw(hiBg);
                }
                sf::Text t(font, gameHistory_[bIdx].moveAlg, fs);
                t.setFillColor(hi ? sf::Color::White : cpColor(gameHistory_[bIdx].cpLoss, true));
                t.setPosition({bX + 3, rowY});
                window.draw(t);
            }
        }

        window.setView(fullView);  // restore full view
    }

    // ── Analysis stats + PV + eval graph (lower portion) ────────────────────
    float hintY = statsAreaY + 4.f;
    unsigned int hfs = std::max(9u, unsigned(10 * scale_));

    // Opening name (show when available)
    if (!currentOpening_.empty()) {
        sf::Text openingTxt(font, currentOpening_, hfs);
        openingTxt.setFillColor(sf::Color(180, 160, 100));
        openingTxt.setPosition({panelX + PAD, hintY});
        window.draw(openingTxt);
        hintY += float(hfs) + 3.f;
    }

    sf::Text hint(font, analysisMode_ ? "<- -> scroll  |  [Z] Analyse  |  Esc: live" : "<- or scroll: browse", hfs);
    hint.setFillColor(sf::Color(80, 80, 110));
    hint.setPosition({panelX + PAD, hintY});
    window.draw(hint);

    if (analysisMode_) {
        float statsY = hintY + float(hfs) + 5.f;
        unsigned int sfs2 = std::max(9u, unsigned(11 * scale_));

        // Tablebase probe (show when ≤5 pieces)
        const Board& vb = viewBoard();
        int pc = Syzygy::pieceCount(vb);
        if (Syzygy::maxPieces() > 0 && pc <= Syzygy::maxPieces()) {
            int tbScore = 0;
            std::string tbStr;
            if (Syzygy::probeWDL(vb, tbScore)) {
                if      (tbScore > Syzygy::TB_CURSED_WIN)   tbStr = "TB: White wins";
                else if (tbScore > 0)                        tbStr = "TB: Cursed win";
                else if (tbScore == 0)                       tbStr = "TB: Draw";
                else if (tbScore >= Syzygy::TB_BLESSED_LOSS) tbStr = "TB: Blessed loss";
                else                                         tbStr = "TB: Black wins";
                // Flip for side to move
                if (vb.turn == Color::Black) {
                    if      (tbScore > Syzygy::TB_CURSED_WIN)   tbStr = "TB: Black wins";
                    else if (tbScore < Syzygy::TB_BLESSED_LOSS)  tbStr = "TB: White wins";
                }
            } else {
                tbStr = "TB: not available";
            }
            sf::Text tbTxt(font, tbStr, sfs2);
            tbTxt.setFillColor(tbScore > 0 ? sf::Color(100, 220, 100) :
                               tbScore < 0 ? sf::Color(220, 100, 100) :
                                             sf::Color(200, 200, 100));
            tbTxt.setPosition({panelX + PAD, statsY});
            window.draw(tbTxt);
            statsY += float(sfs2) + 4.f;
        }

        // Stats line
        std::string statsStr = "depth " + std::to_string(analysisDepth_);
        if (analysisNodes_ > 0)
            statsStr += "  " + std::to_string(analysisNodes_ / 1000) + "K nodes";
        float evalF = analysisEval_ / 100.f;
        std::ostringstream evalOss;
        evalOss << std::fixed << std::setprecision(2);
        if (analysisEval_ >= 0) evalOss << "+" << evalF; else evalOss << evalF;
        statsStr += "  " + evalOss.str();

        sf::Text statsTxt(font, statsStr, sfs2);
        statsTxt.setFillColor(sf::Color(140, 200, 140));
        statsTxt.setPosition({panelX + PAD, statsY});
        window.draw(statsTxt);

        statsY += float(sfs2) + 4.f;

        // ── PV lines (Lichess-style: eval pill + moves + expand toggle) ──────
        auto topMoves = analysisEngine_.getTopRootMoves(3);
        if (!topMoves.empty()) {
            const Board* viewBrd = (viewIdx_ >= 0 && viewIdx_ < (int)gameHistory_.size())
                                   ? &gameHistory_[viewIdx_].board : nullptr;

            // Helper: move to algebraic from a board state
            auto toAlg = [&](const Board& b, const Move& m) -> std::string {
                return VisualGame::moveToAlgebraic(b, m);
            };

            for (int k = 0; k < (int)topMoves.size(); ++k) {
                const auto& rm = topMoves[k];
                if (!rm.move.from.isValid()) continue;

                // Eval string
                float ev = rm.score / 100.f;
                std::ostringstream oss; oss << std::fixed << std::setprecision(2);
                if (rm.score >= 0) oss << "+"; oss << ev;
                std::string evalStr = oss.str();

                // Eval pill background
                float pillW = float(sfs2) * 3.2f;
                float pillH = float(sfs2) + 4.f;
                sf::RectangleShape pill({pillW, pillH});
                pill.setPosition({panelX + PAD, statsY});
                pill.setFillColor(sf::Color(50, 55, 70));
                pill.setOutlineColor(sf::Color(90, 95, 120));
                pill.setOutlineThickness(1.f);
                window.draw(pill);

                sf::Text evalTxt(font, evalStr, sfs2);
                evalTxt.setFillColor(rm.score >= 0 ? sf::Color(200, 230, 200) : sf::Color(230, 180, 180));
                evalTxt.setPosition({panelX + PAD + 3.f, statsY + 2.f});
                window.draw(evalTxt);

                // Build move continuation string from PV
                std::string pvStr;
                Board tmpBoard;
                if (viewBrd) tmpBoard = *viewBrd;
                bool boardValid = viewBrd != nullptr;

                for (int mi = 0; mi < (int)rm.pv.size() && mi < (pvLineExpanded_[k] ? 12 : 6); ++mi) {
                    const Move& m = rm.pv[mi];
                    if (!m.from.isValid()) break;
                    if (!pvStr.empty()) pvStr += ' ';
                    if (boardValid) {
                        pvStr += toAlg(tmpBoard, m);
                        tmpBoard.applyMove(m);
                    } else {
                        pvStr += std::string(1, char('a'+m.from.col)) + std::string(1, char('1'+m.from.rank))
                               + std::string(1, char('a'+m.to.col))   + std::string(1, char('1'+m.to.rank));
                    }
                }
                bool hasContinuation = rm.pv.size() > 6;

                // Moves text
                float movesX = panelX + PAD + pillW + 6.f;
                float movesW = panelW - pillW - 6.f - (hasContinuation ? float(sfs2) * 1.5f : 0.f) - PAD;

                sf::Text movesTxt(font, pvStr, sfs2);
                movesTxt.setFillColor(sf::Color(200, 200, 215));
                movesTxt.setPosition({movesX, statsY + 2.f});
                window.draw(movesTxt);

                // Expand toggle (^ or v) if there's more continuation
                if (hasContinuation) {
                    std::string tog = pvLineExpanded_[k] ? "^" : "v";
                    sf::Text togTxt(font, tog, sfs2);
                    togTxt.setFillColor(sf::Color(120, 120, 160));
                    togTxt.setPosition({panelX + panelW - float(sfs2) * 1.5f - PAD, statsY + 2.f});
                    window.draw(togTxt);
                }

                statsY += pillH + 2.f;

                // Second line if expanded
                if (pvLineExpanded_[k] && rm.pv.size() > 6) {
                    std::string pvStr2;
                    Board tmpBoard2;
                    if (viewBrd) tmpBoard2 = *viewBrd;
                    bool bv2 = viewBrd != nullptr;
                    for (int mi = 6; mi < (int)rm.pv.size() && mi < 12; ++mi) {
                        const Move& m = rm.pv[mi];
                        if (!m.from.isValid()) break;
                        if (!pvStr2.empty()) pvStr2 += ' ';
                        if (bv2) {
                            // Replay first 6 moves to get correct board state
                            if (mi == 6) {
                                for (int r = 0; r < 6 && r < (int)rm.pv.size(); ++r)
                                    if (rm.pv[r].from.isValid()) tmpBoard2.applyMove(rm.pv[r]);
                            }
                            pvStr2 += toAlg(tmpBoard2, m);
                            tmpBoard2.applyMove(m);
                        }
                    }
                    if (!pvStr2.empty()) {
                        sf::Text cont(font, pvStr2, sfs2);
                        cont.setFillColor(sf::Color(160, 160, 180));
                        cont.setPosition({panelX + PAD + pillW + 6.f, statsY + 1.f});
                        window.draw(cont);
                        statsY += float(sfs2) + 3.f;
                    }
                }

                // Separator line between PV entries
                if (k < (int)topMoves.size() - 1) {
                    sf::RectangleShape sep({panelW - 8.f, 1.f});
                    sep.setPosition({panelX + PAD, statsY + 1.f});
                    sep.setFillColor(sf::Color(45, 45, 60));
                    window.draw(sep);
                    statsY += 4.f;
                }
            }
            statsY += 4.f;
        }

        // Eval graph
        float graphH = statsAreaY + statsAreaH - statsY - 6.f;
        float graphW = panelW - 8.f;
        if (graphH > 20.f && graphW > 20.f && !gameHistory_.empty()) {
            sf::RectangleShape graphBg({graphW, graphH});
            graphBg.setPosition({panelX + PAD, statsY});
            graphBg.setFillColor(sf::Color(20, 20, 30));
            graphBg.setOutlineColor(sf::Color(50, 50, 70));
            graphBg.setOutlineThickness(1.f);
            window.draw(graphBg);

            float midY  = statsY + graphH / 2.f;
            float maxEv = 500.f;
            int   n     = int(gameHistory_.size());
            float stepX = graphW / std::max(1, n);

            // ── Filled areas (white advantage = light, black = dark) ─────────
            // Draw vertical bars from midY to the graph line for each segment
            for (int gi = 0; gi < n; ++gi) {
                float ev  = std::max(-maxEv, std::min(maxEv, float(gameHistory_[gi].eval)));
                float gx  = panelX + PAD + gi * stepX;
                float gy  = midY - (ev / maxEv) * (graphH / 2.f - 2.f);
                float barH = std::abs(gy - midY);
                if (barH < 0.5f) continue;
                float barY = (gy < midY) ? gy : midY;
                sf::RectangleShape bar({std::max(1.f, stepX), barH});
                bar.setPosition({gx, barY});
                // White advantage (ev > 0, gy < midY) = light fill
                // Black advantage (ev < 0, gy > midY) = dark fill
                if (ev > 0)
                    bar.setFillColor(sf::Color(200, 200, 210, 60));
                else
                    bar.setFillColor(sf::Color(30, 30, 40, 80));
                window.draw(bar);
            }

            // ── Zero line ────────────────────────────────────────────────────
            sf::RectangleShape zeroLine({graphW, 1.f});
            zeroLine.setPosition({panelX + PAD, midY});
            zeroLine.setFillColor(sf::Color(80, 80, 100));
            window.draw(zeroLine);

            // ── Graph line ───────────────────────────────────────────────────
            for (int gi = 0; gi < n - 1; ++gi) {
                float ev  = std::max(-maxEv, std::min(maxEv, float(gameHistory_[gi].eval)));
                float ev2 = std::max(-maxEv, std::min(maxEv, float(gameHistory_[gi+1].eval)));
                float gx  = panelX + PAD + gi * stepX;
                float gy  = midY - (ev  / maxEv) * (graphH / 2.f - 2.f);
                float gx2 = panelX + PAD + (gi+1) * stepX;
                float gy2 = midY - (ev2 / maxEv) * (graphH / 2.f - 2.f);
                float dx = gx2-gx, dy = gy2-gy, len = std::sqrt(dx*dx+dy*dy);
                if (len > 0.5f) {
                    sf::RectangleShape seg({len, 1.5f});
                    seg.setOrigin({0, 0.75f});
                    seg.setPosition({gx, gy});
                    seg.setRotation(sf::degrees(std::atan2(dy,dx)*180.f/3.14159f));
                    seg.setFillColor(sf::Color(140, 160, 220, 220));
                    window.draw(seg);
                }
            }

            // ── Current position marker + eval label ─────────────────────────
            int markerIdx = analysisMode_ ? viewIdx_ : n - 1;
            if (markerIdx >= 0 && markerIdx < n) {
                float ev = std::max(-maxEv, std::min(maxEv, float(gameHistory_[markerIdx].eval)));
                float gx = panelX + PAD + markerIdx * stepX;
                float gy = midY - (ev / maxEv) * (graphH / 2.f - 2.f);

                // Dot
                sf::CircleShape marker(3.5f);
                marker.setOrigin({3.5f, 3.5f});
                marker.setPosition({gx, gy});
                marker.setFillColor(sf::Color(255, 200, 50));
                window.draw(marker);

                // Eval label — format as "+2.4" or "M" for mate
                std::string evalLabel;
                int rawEval = gameHistory_[markerIdx].eval;
                bool isMate = std::abs(rawEval) >= Engine::MATE_SCORE - 500;
                if (isMate) {
                    evalLabel = (rawEval > 0) ? "+M" : "-M";
                } else {
                    float evF = rawEval / 100.f;
                    std::ostringstream oss; oss << std::fixed << std::setprecision(1);
                    if (rawEval >= 0) oss << "+"; oss << evF;
                    evalLabel = oss.str();
                }

                unsigned int lfs = std::max(8u, unsigned(10 * scale_));
                sf::Text evalLbl(font, evalLabel, lfs);
                evalLbl.setFillColor(sf::Color(255, 220, 80));

                // Position label above or below the dot to avoid clipping
                float lblX = gx + 5.f;
                float lblY = (gy - graphH * 0.25f > statsY) ? gy - float(lfs) - 4.f
                                                              : gy + 5.f;
                // Clamp horizontally so it doesn't overflow the panel
                float lblW = evalLbl.getLocalBounds().size.x;
                if (lblX + lblW > panelX + panelW - 4.f)
                    lblX = gx - lblW - 5.f;

                evalLbl.setPosition({lblX, lblY});
                window.draw(evalLbl);
            }
        }
    }
}


// DRAW ANALYSIS OVERLAY — dim indicator when viewing history
// -------------------------------------------------------
void VisualGame::drawAnalysisOverlay() {
    if (!analysisMode_ || viewIdx_ < 0 || viewIdx_ >= (int)gameHistory_.size()) return;

    // Highlight the move that was played from this position
    const HistoryEntry& entry = gameHistory_[viewIdx_];
    if (entry.move.from.isValid() && entry.move.to.isValid()) {
        sf::Color fromCol(255, 200, 50, 80);
        sf::Color toCol(255, 200, 50, 120);
        sf::RectangleShape fromSq({float(dynSQ_), float(dynSQ_)});
        fromSq.setPosition({float(dynOX_ + entry.move.from.col * dynSQ_),
                             float(OY + (7 - entry.move.from.rank) * dynSQ_)});
        fromSq.setFillColor(fromCol);
        window.draw(fromSq);
        sf::RectangleShape toSq({float(dynSQ_), float(dynSQ_)});
        toSq.setPosition({float(dynOX_ + entry.move.to.col * dynSQ_),
                           float(OY + (7 - entry.move.to.rank) * dynSQ_)});
        toSq.setFillColor(toCol);
        window.draw(toSq);
    }

    // "ANALYSIS" banner at top of board
    unsigned int bfs = std::max(10u, unsigned(12 * scale_));
    int moveN = entry.moveNumber;
    std::string side = (entry.sideToMove == Color::White) ? "White" : "Black";
    std::string label = "ANALYSIS  Move " + std::to_string(moveN) + " - " + side;
    sf::Text banner(font, label, bfs);
    banner.setFillColor(sf::Color(255, 200, 50));
    banner.setPosition({float(dynOX_), float(OY + dynSQ_ * 8 + 50)});
    window.draw(banner);
}
