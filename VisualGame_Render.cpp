// ARCHITECTURE FIX M-2: Split from VisualGame.cpp — all rendering and drawing
// functions (board, pieces, highlights, eval bars, HUD, status, arrows, etc.)
//
#include "VisualGame.h"
#include <iostream>
#include <cmath>
#include <sstream>
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
    window.clear(sf::Color(40, 40, 40));
    drawBoard();
    drawHighlights();
    // Automate Chess setup overlay (valid placement squares)
    if (isAutomateChess_ && !board.automateSetupComplete)
        drawAutomateSetupOverlay();
    drawPieces();
    if (showArrows && (engineThinking || botHasPendingMove_))
        drawPVArrows();
    if (botVsNNUE_)
        drawDualEvalBars();
    else
        drawEvalBar();
    drawCoordinates();
    drawStatus();
    drawHUD();
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

#ifdef DUCK_CHESS
            // Duck piece uses special texture
            if (p.isDuck()) {
                sf::Sprite sprite(duckTexture_);
                sf::Vector2u ts = duckTexture_.getSize();
                float scale = float(SQ) / float(std::max(ts.x, ts.y));
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
            float scale = float(SQ) / float(std::max(ts.x, ts.y));
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
    float scale = float(SQ) / float(std::max(ts.x, ts.y));
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
    if (pi < 0 || pi >= 6) return;  // FIX 10.2: bounds check on piece type index

    sf::Sprite sprite(pieceTextures[ci][pi]);
    sf::Vector2u ts = pieceTextures[ci][pi].getSize();
    float scale = float(SQ) / float(std::max(ts.x, ts.y));
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
        float scale = float(SQ) / float(std::max(ts.x, ts.y));  // FIX 10.6: uniform scaling
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
    float barH = float(SQ * 8);
    float barW = float(EVAL_BAR_W);
    float barY = float(OY);

    // Left bar: White's engine eval
    float leftX = float(OX - EVAL_BAR_W - 28);
    // Right bar: Black's engine eval
    float rightX = float(OX + SQ * 8 + 28);

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
    // Side config indicator
    std::string sideTag;
    if (sideConfig_ == SideConfig::Swapped) sideTag = " [SWAPPED]";
    else if (sideConfig_ == SideConfig::Random) sideTag = " [RANDOM]";

    if (botVsNNUE_) {
        hudText = "BOT vs NNUE";
        if (isDuckChess_) hudText += "  [DUCK]";
        if (isAutomateChess_) hudText += "  [AUTOMATE]";
        if (botPaused) hudText += "  [PAUSED]";
        if (fastMode) hudText += "  [FAST]";
        hudText += sideTag;
        std::string w = nnueOnWhite_ ? "NNUE" : "Classical";
        std::string b = nnueOnWhite_ ? "Classical" : "NNUE";
        hudText += "  |  White=" + w + "  Black=" + b;
        hudText += "  |  [Space] Pause  [S] Sides  [A] Arrows  [F] Fast  [R] Reset  [V] Exit";
    } else if (botVsBot) {
        hudText = "BOT vs BOT";
        if (isDuckChess_) hudText += "  [DUCK]";
        if (isAutomateChess_) hudText += "  [AUTOMATE]";
        if (botPaused) hudText += "  [PAUSED]";
        if (fastMode) hudText += "  [FAST]";
        if (nnueEnabled_) hudText += "  [NNUE]";
        hudText += "  |  [Space] Pause  [A] Arrows  [F] Fast  [D] Duck  [M] Automate  [R] Reset  [B] Exit";
    } else {
        hudText = "PLAYER vs ENGINE";
        if (isDuckChess_) hudText += "  [DUCK]";
        if (isAutomateChess_) hudText += "  [AUTOMATE]";
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

    sf::Text hud(font, hudText, 13);
    hud.setFillColor(sf::Color(180, 180, 180));
    hud.setPosition({float(OX), float(OY - 25)});
    window.draw(hud);

    // Duck chess mode indicator
    if (isDuckChess_) {
        sf::Text duckLabel(font, "DUCK CHESS", 14);
        duckLabel.setFillColor(sf::Color(255, 220, 50));
        duckLabel.setPosition({float(OX + SQ * 8 - 90), float(OY - 25)});
        window.draw(duckLabel);
    }
    // Automate Chess mode indicator
    if (isAutomateChess_) {
        std::string automateLabel = board.automateSetupComplete ? "AUTOMATE CHESS" : "AUTOMATE SETUP";
        sf::Text autoLabel(font, automateLabel, 14);
        autoLabel.setFillColor(sf::Color(100, 220, 255));
        autoLabel.setPosition({float(OX + SQ * 8 - 130), float(OY - 25)});
        window.draw(autoLabel);
    }

    // NNUE status line (below board, above status)
    std::string displayStatus;
    {
        std::lock_guard<std::mutex> lk(nnueStatusMutex_);
        displayStatus = nnueStatus_;
    }
    if (!displayStatus.empty()) {
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
