// TR_Graph.cpp  --  Real-time training graph rendering
#include "TR_Types.h"
#include "TR_Globals.h"
#include "TR_Fwd.h"
#include <map>

// ── Graph drawing ─────────────────────────────────────────────────
LRESULT CALLBACK GraphProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp);

static void DrawGraph(HWND hw) {
    RECT rc; GetClientRect(hw, &rc);
    int W2 = rc.right, H2 = rc.bottom;
    PAINTSTRUCT ps; HDC hdc = BeginPaint(hw, &ps);
    HDC memDC = CreateCompatibleDC(hdc);
    HBITMAP bmp = CreateCompatibleBitmap(hdc, W2, H2);
    // AUDIT FIX M6: Save old bitmap to restore before cleanup (prevents GDI handle leak)
    HBITMAP oldBmp = (HBITMAP)SelectObject(memDC, bmp);

    Graphics g(memDC);
    g.SetSmoothingMode(SmoothingModeAntiAlias);
    SolidBrush bgBr(Color(255,16,16,24));
    g.FillRectangle(&bgBr, 0, 0, W2, H2);

    for (auto& b : g_graph.panelBounds) b.active = false;

    // legendReserve = width to keep clear on the right for the legend
    auto drawPanelDesc = [&](int panelIdx, float px, float py, float pw,
                              float legendReserve, const wchar_t* desc) {
        if (g_graph.hoverPanel != panelIdx) return;
        Font   df(L"Segoe UI", 7.5f);
        RectF dr;
        g.MeasureString(desc, -1, &df, PointF(0,0), &dr);
        // Right-align but stop before the legend area
        float maxRight = px + pw - legendReserve;
        float dx = maxRight - dr.Width - 6;
        // Don't overlap the panel title (~90 px from left edge)
        float minLeft = px + 90;
        if (dx < minLeft) dx = minLeft;
        // Clip so text never bleeds into the legend
        Region oldClip; g.GetClip(&oldClip);
        g.SetClip(RectF(minLeft, py, maxRight - minLeft, 18));
        // Subtle background so description is readable over chart elements
        float bgW = (std::min)(dr.Width + 8, maxRight - dx);
        SolidBrush bg(Color(180, 20, 20, 30));
        g.FillRectangle(&bg, dx - 2, py + 1, bgW, dr.Height + 2);
        SolidBrush db(Color(160, 160, 170, 190));
        g.DrawString(desc, -1, &df, PointF(dx, py + 2), &db);
        g.SetClip(&oldClip);
    };

    // FIX 7: Dirty-flag cache — catches both appends AND in-place mutations
    static std::vector<TrainPoint> cachedPts;
    {
        if (g_graph.dirty.exchange(false, std::memory_order_relaxed)) {
            std::lock_guard<std::mutex> lk(g_st.mtx);
            cachedPts = g_st.pts;
        }
    }
    auto& pts = cachedPts;

    if (pts.size() < 2) {
        Font fnt(L"Segoe UI", 10.0f);
        SolidBrush tb(Color(255,80,80,100));
        g.DrawString(L"No data yet", -1, &fnt, PointF((float)W2/2-40,(float)H2/2-8), &tb);
        BitBlt(hdc,0,0,W2,H2,memDC,0,0,SRCCOPY);
        SelectObject(memDC, oldBmp);  // AUDIT FIX M6: restore old bitmap
        DeleteObject(bmp); DeleteDC(memDC);
        EndPaint(hw, &ps); return;
    }

    // ── Compute generation boundaries ──
    std::vector<int> genBounds;  // indices where a new gen starts
    std::vector<int> genNumbers; // gen number for each boundary
    for (size_t i = 1; i < pts.size(); i++) {
        if (pts[i].gen != pts[i-1].gen) {
            genBounds.push_back((int)i);
            genNumbers.push_back(pts[i].gen);
        }
    }

    // ---- Layout: horizontal split if phase panel is visible ----
    float leftW  = (float)W2;    // width for stacked panels (Loss/Acc/LR)
    float rightX = 0, rightW = 0; // phase panel
    float hGap = 6.0f;
    if (g_graph.showPhase) {
        rightW = (float)W2 / 3.0f;
        leftW  = (float)W2 - rightW - hGap;
        rightX = leftW + hGap;
    }

    // ---- Stacked panels (Loss / Acc / LR / NPS) in left region ----
    float lossWeight = g_graph.showLoss ? 3.0f : 0.0f;
    float accWeight  = g_graph.showAcc  ? 1.0f : 0.0f;
    float lrWeight   = g_graph.showLR   ? 1.0f : 0.0f;
    float npsWeight  = g_graph.showNPS  ? 1.0f : 0.0f;
    float totalWeight = lossWeight + accWeight + lrWeight + npsWeight;
    bool anyLeft = totalWeight > 0.01f;
    if (!anyLeft && !g_graph.showPhase) { lossWeight = 1.0f; totalWeight = 1.0f; anyLeft = true; }

    int numPanels = (g_graph.showLoss?1:0) + (g_graph.showAcc?1:0) + (g_graph.showLR?1:0) + (g_graph.showNPS?1:0);
    if (!anyLeft) numPanels = 0;

    float ml = 52, mr = 16;
    float panelGap = 4.0f;
    float availH = (float)H2 - panelGap * ((numPanels > 1 ? numPanels : 1) - 1);
    float lossH = anyLeft ? availH * (lossWeight / totalWeight) : 0;
    float accH  = anyLeft ? availH * (accWeight  / totalWeight) : 0;
    float lrH   = anyLeft ? availH * (lrWeight   / totalWeight) : 0;
    float npsH  = anyLeft ? availH * (npsWeight  / totalWeight) : 0;
    float curY = 0;
    float gw = leftW - ml - mr;
    if (gw < 20) gw = 20;

    auto xfLeft = [&](int i) -> float {
        return ml + (float)i / (float)(pts.size()-1) * gw;
    };

    // Helper: draw vertical generation boundary lines in a panel
    auto drawGenBounds = [&](float panelTop, float panelH, std::function<float(int)> xfFunc) {
        Pen genPen(Color(80, 180, 180, 200), 1.0f);
        genPen.SetDashStyle(DashStyleDot);
        Font genFnt(L"Consolas", 6.5f);
        SolidBrush genBr(Color(140, 180, 180, 200));
        for (size_t b = 0; b < genBounds.size(); b++) {
            float gx = xfFunc(genBounds[b]);
            g.DrawLine(&genPen, gx, panelTop + 14, gx, panelTop + panelH - 2);
            std::wstring lbl = L"G" + std::to_wstring(genNumbers[b]);
            g.DrawString(lbl.c_str(), -1, &genFnt, PointF(gx + 2, panelTop + panelH - 13), &genBr);
        }
    };

    // ---- Loss panel ----
    if (g_graph.showLoss || (!anyLeft && !g_graph.showPhase)) {
        float pt_h = (lossWeight > 0) ? lossH : availH;
        float pt_top = curY;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2;
        if (gh2 < 10) gh2 = 10;

        g_graph.panelBounds[0] = { pt_top, pt_top + pt_h, true };

        SolidBrush panelBg(Color(255,20,20,30));
        g.FillRectangle(&panelBg, 0.0f, pt_top, leftW, pt_h);

        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Loss", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);
        drawPanelDesc(0, 0.0f, pt_top, leftW, 144, L"Training loss \u2014 lower = better. Train/Val curves with validation phase highlighted. Drops fast early, then plateaus.");

        double minV=1e9, maxV=-1e9;
        for (auto& p2 : pts) {
            if (!p2.hasLoss) continue;
            minV = (std::min)(minV, p2.train); maxV = (std::max)(maxV, p2.train);
            if (p2.hasVal) { minV = (std::min)(minV, p2.val); maxV = (std::max)(maxV, p2.val); }
        }
        if (maxV <= minV) maxV = minV + 0.1;
        double rng = maxV - minV;
        minV -= rng*0.05; maxV += rng*0.05; rng = maxV - minV;
        auto yf = [&](double v) -> float { return mt2 + (float)((maxV-v)/rng)*gh2; };

        Pen gridPen(Color(40,60,60,80), 1.0f);
        Font gridFnt(L"Consolas", 7.0f);
        SolidBrush gridBr(Color(255,80,80,100));
        for (int i=0; i<=4; i++) {
            float y2 = mt2 + gh2*i/4;
            g.DrawLine(&gridPen, ml, y2, ml+gw, y2);
            double val = maxV - rng*i/4;
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(4)<<val;
            g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
        }

        drawGenBounds(pt_top, pt_h, xfLeft);

        int bestTrainIdx=-1, bestValIdx=-1;
        double bestTrain=1e9, bestVal=1e9;
        for (size_t i=0; i<pts.size(); i++) {
            if (!pts[i].hasLoss) continue;
            if (pts[i].train < bestTrain) { bestTrain=pts[i].train; bestTrainIdx=(int)i; }
            if (pts[i].hasVal && pts[i].val < bestVal) { bestVal=pts[i].val; bestValIdx=(int)i; }
        }

        Pen trainPen(Color(255,65,125,245), 1.8f);
        for (size_t i=1; i<pts.size(); i++)
            if (!pts[i].hasLoss || !pts[i-1].hasLoss) continue;
            g.DrawLine(&trainPen, xfLeft((int)i-1), yf(pts[i-1].train), xfLeft((int)i), yf(pts[i].train));

        Pen valPen(Color(255,245,160,60), 1.8f);
        { bool st=false; float px2=0,py2=0;
          for (size_t i=0; i<pts.size(); i++) {
            if (!pts[i].hasVal) continue;
            float cx=xfLeft((int)i), cy=yf(pts[i].val);
            if (st) g.DrawLine(&valPen,px2,py2,cx,cy);
            px2=cx; py2=cy; st=true;
        }}

        if (bestTrainIdx >= 0) {
            float bx = xfLeft(bestTrainIdx), by = yf(bestTrain);
            SolidBrush mk(Color(255,65,125,245));
            PointF dm[4] = {{bx,by-5},{bx+5,by},{bx,by+5},{bx-5,by}};
            g.FillPolygon(&mk, dm, 4);
            Font mf(L"Consolas",6.5f);
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(5)<<bestTrain;
            std::wstring lbl = L"Best: " + ss.str();
            float lx2 = bx + 7; if (lx2 + 80 > leftW) lx2 = bx - 90;
            g.DrawString(lbl.c_str(),-1,&mf,PointF(lx2,by-5),&mk);
        }
        if (bestValIdx >= 0) {
            float bx = xfLeft(bestValIdx), by = yf(bestVal);
            SolidBrush mk(Color(255,245,160,60));
            PointF dm[4] = {{bx,by-5},{bx+5,by},{bx,by+5},{bx-5,by}};
            g.FillPolygon(&mk, dm, 4);
            Font mf(L"Consolas",6.5f);
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(5)<<bestVal;
            std::wstring lbl = L"Best: " + ss.str();
            float lx2 = bx + 7; if (lx2 + 80 > leftW) lx2 = bx - 90;
            g.DrawString(lbl.c_str(),-1,&mf,PointF(lx2,by+2),&mk);
        }

        { Font lf(L"Segoe UI",7.5f);
          float lx = leftW - mr - 120, ly = pt_top + 3;
          SolidBrush b1(Color(255,65,125,245)); Pen lp1(Color(255,65,125,245),2);
          g.DrawLine(&lp1,lx,ly+5,lx+12,ly+5);
          g.DrawString(L"train",-1,&lf,PointF(lx+14,ly-1),&b1);
          SolidBrush b2(Color(255,245,160,60)); Pen lp2(Color(255,245,160,60),2);
          g.DrawLine(&lp2,lx+52,ly+5,lx+64,ly+5);
          g.DrawString(L"val",-1,&lf,PointF(lx+66,ly-1),&b2);
        }
        curY += pt_h + panelGap;
    }

    // ---- Accuracy panel ----
    if (g_graph.showAcc) {
        float pt_top = curY, pt_h = accH;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        g_graph.panelBounds[1] = { pt_top, pt_top + pt_h, true };

        SolidBrush panelBg(Color(255,18,22,28));
        g.FillRectangle(&panelBg, 0.0f, pt_top, leftW, pt_h);
        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Accuracy", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);
        drawPanelDesc(1, 0.0f, pt_top, leftW, 0, L"Move-prediction accuracy on held-out positions. Rises as the network learns stronger patterns.");

        bool hasAnyAcc = false;
        for (auto& p2 : pts) if (p2.hasAcc) { hasAnyAcc = true; break; }

        if (!hasAnyAcc) {
            Font nf(L"Segoe UI", 9.0f); SolidBrush nb(Color(255,80,80,100));
            g.DrawString(L"No accuracy data", -1, &nf, PointF(ml+gw/2-50, mt2+gh2/2-6), &nb);
        } else {
            double minA=1e9, maxA=-1e9;
            for (auto& p2 : pts) {
                if (!p2.hasAcc) continue;
                minA=(std::min)(minA,p2.accuracy); maxA=(std::max)(maxA,p2.accuracy);
            }
            if (maxA<=minA) maxA=minA+0.1;
            double rng=maxA-minA; minA-=rng*0.05; maxA+=rng*0.05; rng=maxA-minA;
            auto yf=[&](double v)->float{return mt2+(float)((maxA-v)/rng)*gh2;};
            Pen gridPen(Color(40,60,60,80),1.0f); Font gridFnt(L"Consolas",7.0f);
            SolidBrush gridBr(Color(255,80,80,100));
            for (int i=0;i<=4;i++){
                float y2=mt2+gh2*i/4;
                g.DrawLine(&gridPen,ml,y2,ml+gw,y2);
                double val=maxA-rng*i/4;
                std::wostringstream ss; ss<<std::fixed<<std::setprecision(3)<<val;
                g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
            }
            drawGenBounds(pt_top, pt_h, xfLeft);
            int bestAccIdx=-1; double bestAcc=-1;
            Pen accPen(Color(255,0,200,120),1.8f);
            { bool st=false; float px3=0,py3=0;
              for (size_t i=0;i<pts.size();i++){
                if (!pts[i].hasAcc) continue;
                float cx=xfLeft((int)i),cy=yf(pts[i].accuracy);
                if (st) g.DrawLine(&accPen,px3,py3,cx,cy);
                px3=cx; py3=cy; st=true;
                if (pts[i].accuracy>bestAcc){bestAcc=pts[i].accuracy; bestAccIdx=(int)i;}
            }}
            if (bestAccIdx>=0){
                float bx=xfLeft(bestAccIdx),by=yf(bestAcc);
                SolidBrush mk(Color(255,0,200,120));
                PointF dm[4]={{bx,by-5},{bx+5,by},{bx,by+5},{bx-5,by}};
                g.FillPolygon(&mk,dm,4);
                Font mf(L"Consolas",6.5f);
                std::wostringstream ss; ss<<std::fixed<<std::setprecision(4)<<bestAcc;
                g.DrawString((L"Best: "+ss.str()).c_str(),-1,&mf,PointF(bx+7,by-5),&mk);
            }
        }
        curY += pt_h + panelGap;
    }

    // ---- Learning Rate panel ----
    if (g_graph.showLR) {
        float pt_top = curY, pt_h = lrH;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        g_graph.panelBounds[2] = { pt_top, pt_top + pt_h, true };

        SolidBrush panelBg(Color(255,22,18,28));
        g.FillRectangle(&panelBg, 0.0f, pt_top, leftW, pt_h);
        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Learning Rate", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);
        drawPanelDesc(2, 0.0f, pt_top, leftW, 0, L"LR schedule \u2014 drops at milestones or when loss plateaus. Lower = finer weight adjustments.");

        bool hasAnyLR = false;
        for (auto& p2 : pts) if (p2.hasLR) { hasAnyLR = true; break; }

        if (!hasAnyLR) {
            Font nf(L"Segoe UI", 9.0f); SolidBrush nb(Color(255,80,80,100));
            g.DrawString(L"No LR data", -1, &nf, PointF(ml+gw/2-30, mt2+gh2/2-6), &nb);
        } else {
            double minL=1e9, maxL=-1e9;
            for (auto& p2 : pts) {
                if (!p2.hasLR) continue;
                minL=(std::min)(minL,p2.lr); maxL=(std::max)(maxL,p2.lr);
            }
            if (maxL<=minL) maxL=minL+0.0001;
            double rng=maxL-minL; minL-=rng*0.05; maxL+=rng*0.05; rng=maxL-minL;
            auto yf=[&](double v)->float{return mt2+(float)((maxL-v)/rng)*gh2;};
            Pen gridPen(Color(40,60,60,80),1.0f); Font gridFnt(L"Consolas",7.0f);
            SolidBrush gridBr(Color(255,80,80,100));
            for (int i=0;i<=4;i++){
                float y2=mt2+gh2*i/4;
                g.DrawLine(&gridPen,ml,y2,ml+gw,y2);
                double val=maxL-rng*i/4;
                std::wostringstream ss; ss<<std::scientific<<std::setprecision(2)<<val;
                g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
            }
            drawGenBounds(pt_top, pt_h, xfLeft);
            Pen lrPen(Color(255,180,80,220),1.8f);
            { bool st=false; float px4=0,py4=0;
              for (size_t i=0;i<pts.size();i++){
                if (!pts[i].hasLR) continue;
                float cx=xfLeft((int)i),cy=yf(pts[i].lr);
                if (st) g.DrawLine(&lrPen,px4,py4,cx,cy);
                px4=cx; py4=cy; st=true;
            }}
        }
        curY += pt_h + panelGap;
    }

    // ---- NPS panel ----
    if (g_graph.showNPS) {
        float pt_top = curY, pt_h = npsH;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        g_graph.panelBounds[4] = { pt_top, pt_top + pt_h, true };

        SolidBrush panelBg(Color(255,18,22,28));
        g.FillRectangle(&panelBg, 0.0f, pt_top, leftW, pt_h);
        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"NPS (Self-Play)", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);
        drawPanelDesc(4, 0.0f, pt_top, leftW, 0, L"Nodes per second during self-play. Higher = faster generation. Reflects search + eval speed.");

        bool hasAnyNps = false;
        for (auto& p2 : pts) if (p2.hasNps) { hasAnyNps = true; break; }

        if (!hasAnyNps) {
            Font nf(L"Segoe UI", 9.0f); SolidBrush nb(Color(255,80,80,100));
            g.DrawString(L"No NPS data", -1, &nf, PointF(ml+gw/2-35, mt2+gh2/2-6), &nb);
        } else {
            double minN=1e9, maxN=-1e9;
            for (auto& p2 : pts) {
                if (!p2.hasNps) continue;
                minN=(std::min)(minN,p2.nps); maxN=(std::max)(maxN,p2.nps);
            }
            if (maxN<=minN) maxN=minN+1.0;
            double rng=maxN-minN; minN=std::max(0.0,minN-rng*0.1); maxN+=rng*0.1; rng=maxN-minN;
            auto yf=[&](double v)->float{return mt2+(float)((maxN-v)/rng)*gh2;};

            Pen gridPen(Color(40,60,60,80),1.0f); Font gridFnt(L"Consolas",7.0f);
            SolidBrush gridBr(Color(255,80,80,100));
            for (int i=0;i<=4;i++){
                float y2=mt2+gh2*i/4;
                g.DrawLine(&gridPen,ml,y2,ml+gw,y2);
                double val=maxN-rng*i/4;
                std::wostringstream ss;
                if (val>=1000000) ss<<std::fixed<<std::setprecision(2)<<val/1000000.0<<L"M";
                else if (val>=1000) ss<<std::fixed<<std::setprecision(1)<<val/1000.0<<L"K";
                else ss<<std::fixed<<std::setprecision(0)<<val;
                g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
            }
            drawGenBounds(pt_top, pt_h, xfLeft);

            // Build a (gen, step) -> x position map using training points (hasLoss=true).
            // NPS points have the same step numbers as training epochs, so we can
            // align them on the x-axis by matching step within each gen.
            std::map<std::pair<int,int>, float> stepToX;
            for (size_t i = 0; i < pts.size(); i++) {
                if (pts[i].hasLoss && pts[i].step > 0)
                    stepToX[{pts[i].gen, pts[i].step}] = xfLeft((int)i);
            }
            // For NPS points, find x by matching (gen, step) to training points.
            // If no match, fall back to array index.
            auto npsX = [&](size_t i) -> float {
                const auto& p = pts[i];
                auto it = stepToX.find({p.gen, p.step});
                if (it != stepToX.end()) return it->second;
                return xfLeft((int)i);
            };

            // Continuous line through NPS sample points (one per epoch slot).
            Pen npsPen(Color(255,80,220,180),1.8f);
            SolidBrush dotBr(Color(255,80,220,180));
            bool started=false; float px5=0,py5=0;
            int lastGen=-1;
            for (size_t i=0;i<pts.size();i++){
                if (!pts[i].hasNps) continue;
                float cx=npsX(i), cy=yf(pts[i].nps);
                if (started) {
                    if (pts[i].gen==lastGen) {
                        g.DrawLine(&npsPen,px5,py5,cx,cy);
                    } else {
                        g.FillEllipse(&dotBr,cx-3.0f,cy-3.0f,6.0f,6.0f);
                    }
                } else {
                    g.FillEllipse(&dotBr,cx-3.0f,cy-3.0f,6.0f,6.0f);
                }
                px5=cx; py5=cy; started=true; lastGen=pts[i].gen;
            }
        }
        curY += pt_h + panelGap;
    }

    // ==== Phase Loss panel (right 1/3) ====
    if (g_graph.showPhase) {
        float pml = rightX + 48, pmr = 12;
        float pgw = rightW - 48 - pmr;
        if (pgw < 20) pgw = 20;
        float pt_top = 0, pt_h = (float)H2;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        g_graph.panelBounds[3] = { pt_top, pt_top + pt_h, true };

        // Background with subtle separator
        SolidBrush panelBg(Color(255,18,20,32));
        g.FillRectangle(&panelBg, rightX, pt_top, rightW, pt_h);
        Pen sepPen(Color(60,80,80,120), 1.0f);
        g.DrawLine(&sepPen, rightX, 0.0f, rightX, (float)H2);

        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Phase Loss", -1, &titleFnt, PointF(pml, pt_top+2), &titleBr);
        drawPanelDesc(3, rightX, pt_top, rightW, 200, L"Loss by game phase (O=Opening, M=Midgame, E=Endgame). Diagnose which phase needs more data.");

        bool hasAnyPhase = false;
        for (auto& p2 : pts) if (p2.hasPhase) { hasAnyPhase = true; break; }

        auto xfRight = [&](int i) -> float {
            return pml + (float)i / (float)(pts.size()-1) * pgw;
        };

        if (!hasAnyPhase) {
            Font nf(L"Segoe UI", 9.0f); SolidBrush nb(Color(255,80,80,100));
            g.DrawString(L"No phase data yet", -1, &nf, PointF(pml+pgw/2-55, mt2+gh2/2-6), &nb);
        } else {
            // Auto-scale across all three phase curves
            double minP=1e9, maxP=-1e9;
            for (auto& p2 : pts) {
                if (!p2.hasPhase) continue;
                minP = (std::min)(minP, (std::min)(p2.openingLoss, (std::min)(p2.middlegameLoss, p2.endgameLoss)));
                maxP = (std::max)(maxP, (std::max)(p2.openingLoss, (std::max)(p2.middlegameLoss, p2.endgameLoss)));
            }
            if (maxP <= minP) maxP = minP + 0.1;
            double rng = maxP - minP;
            minP -= rng*0.05; maxP += rng*0.05; rng = maxP - minP;
            auto yf = [&](double v) -> float { return mt2 + (float)((maxP-v)/rng)*gh2; };

            // Grid
            Pen gridPen(Color(40,60,60,80), 1.0f);
            Font gridFnt(L"Consolas", 7.0f);
            SolidBrush gridBr(Color(255,80,80,100));
            for (int i=0; i<=4; i++) {
                float y2 = mt2 + gh2*i/4;
                g.DrawLine(&gridPen, pml, y2, pml+pgw, y2);
                double val = maxP - rng*i/4;
                std::wostringstream ss; ss<<std::fixed<<std::setprecision(4)<<val;
                g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(rightX+2,y2-6),&gridBr);
            }

            drawGenBounds(pt_top, pt_h, xfRight);

            // Opening curve - green
            Pen openPen(Color(255,76,175,80), 1.8f);
            { bool st=false; float px=0,py=0;
              for (size_t i=0;i<pts.size();i++){
                if (!pts[i].hasPhase) continue;
                float cx=xfRight((int)i), cy=yf(pts[i].openingLoss);
                if (st) g.DrawLine(&openPen,px,py,cx,cy);
                px=cx; py=cy; st=true;
            }}

            // Middlegame curve - blue
            Pen midPen(Color(255,66,165,245), 1.8f);
            { bool st=false; float px=0,py=0;
              for (size_t i=0;i<pts.size();i++){
                if (!pts[i].hasPhase) continue;
                float cx=xfRight((int)i), cy=yf(pts[i].middlegameLoss);
                if (st) g.DrawLine(&midPen,px,py,cx,cy);
                px=cx; py=cy; st=true;
            }}

            // Endgame curve - red/orange
            Pen endPen(Color(255,244,67,54), 1.8f);
            { bool st=false; float px=0,py=0;
              for (size_t i=0;i<pts.size();i++){
                if (!pts[i].hasPhase) continue;
                float cx=xfRight((int)i), cy=yf(pts[i].endgameLoss);
                if (st) g.DrawLine(&endPen,px,py,cx,cy);
                px=cx; py=cy; st=true;
            }}

            // Best markers for each phase
            auto drawBest = [&](const std::wstring& label, Color clr, auto getter) {
                int bestIdx = -1; double bestV = 1e9;
                for (size_t i=0; i<pts.size(); i++) {
                    if (!pts[i].hasPhase) continue;
                    double v = getter(pts[i]);
                    if (v < bestV) { bestV = v; bestIdx = (int)i; }
                }
                if (bestIdx >= 0) {
                    float bx = xfRight(bestIdx), by = yf(bestV);
                    SolidBrush mk(clr);
                    PointF dm[4] = {{bx,by-4},{bx+4,by},{bx,by+4},{bx-4,by}};
                    g.FillPolygon(&mk, dm, 4);
                }
            };
            drawBest(L"O", Color(255,76,175,80),  [](const TrainPoint& p){ return p.openingLoss; });
            drawBest(L"M", Color(255,66,165,245),  [](const TrainPoint& p){ return p.middlegameLoss; });
            drawBest(L"E", Color(255,244,67,54),   [](const TrainPoint& p){ return p.endgameLoss; });

            // Legend
            Font lf(L"Segoe UI", 7.0f);
            float lx = rightX + rightW - pmr - 180, ly = pt_top + 3;
            SolidBrush bO(Color(255,76,175,80));  Pen lpO(Color(255,76,175,80),2);
            g.DrawLine(&lpO, lx, ly+5, lx+10, ly+5);
            g.DrawString(L"Open", -1, &lf, PointF(lx+12, ly-1), &bO);
            SolidBrush bM(Color(255,66,165,245)); Pen lpM(Color(255,66,165,245),2);
            g.DrawLine(&lpM, lx+50, ly+5, lx+60, ly+5);
            g.DrawString(L"Mid", -1, &lf, PointF(lx+62, ly-1), &bM);
            SolidBrush bE(Color(255,244,67,54));   Pen lpE(Color(255,244,67,54),2);
            g.DrawLine(&lpE, lx+96, ly+5, lx+106, ly+5);
            g.DrawString(L"End", -1, &lf, PointF(lx+108, ly-1), &bE);
        }
    }

    // ---- Hover crosshair + tooltip (spans full width) ----
    if (g_graph.hoverIdx >= 0 && g_graph.hoverIdx < (int)pts.size()) {
        float hx = xfLeft(g_graph.hoverIdx);
        Pen crossPen(Color(100,200,200,220), 1.0f);
        crossPen.SetDashStyle(DashStyleDash);
        g.DrawLine(&crossPen, hx, 0.0f, hx, (float)H2);

        // Also draw crosshair in phase panel
        if (g_graph.showPhase) {
            float pml2 = rightX + 48;
            float pgw2 = rightW - 48 - 12;
            float hx2 = pml2 + (float)g_graph.hoverIdx / (float)(pts.size()-1) * pgw2;
            g.DrawLine(&crossPen, hx2, 0.0f, hx2, (float)H2);
        }

        auto& hp = pts[g_graph.hoverIdx];
        std::wostringstream ss;
        ss << L"Step: " << hp.step << L"  Gen: " << hp.gen;
        ss << L"\nTrain: " << std::fixed << std::setprecision(6) << hp.train;
        if (hp.hasVal) ss << L"\nVal: " << std::fixed << std::setprecision(6) << hp.val;
        if (hp.hasLR)  ss << L"\nLR: " << std::scientific << std::setprecision(4) << hp.lr;
        if (hp.hasAcc) ss << L"\nAcc: " << std::fixed << std::setprecision(4) << hp.accuracy;
        if (hp.hasPhase) {
            ss << L"\nOpen: " << std::fixed << std::setprecision(6) << hp.openingLoss;
            ss << L"  Mid: " << hp.middlegameLoss;
            ss << L"  End: " << hp.endgameLoss;
        }
        std::wstring info = ss.str();

        Font tipFnt(L"Consolas", 7.5f);
        RectF tipRc;
        g.MeasureString(info.c_str(), -1, &tipFnt, PointF(0,0), &tipRc);
        float tipW = tipRc.Width + 14, tipH = tipRc.Height + 10;
        float tipX = (float)g_graph.mousePt.x + 14;
        float tipY = (float)g_graph.mousePt.y + 4;
        if (tipX + tipW > W2) tipX = (float)g_graph.mousePt.x - tipW - 4;
        if (tipY + tipH > H2) tipY = (float)g_graph.mousePt.y - tipH - 4;

        SolidBrush tipBg(Color(230,30,30,40));
        Pen tipBorder(Color(200,100,100,120), 1.0f);
        g.FillRectangle(&tipBg, tipX, tipY, tipW, tipH);
        g.DrawRectangle(&tipBorder, tipX, tipY, tipW, tipH);
        SolidBrush tipText(Color(255,220,220,230));
        g.DrawString(info.c_str(), -1, &tipFnt, PointF(tipX+7, tipY+5), &tipText);
    }

    BitBlt(hdc,0,0,W2,H2,memDC,0,0,SRCCOPY);
    SelectObject(memDC, oldBmp);  // AUDIT FIX M6: restore old bitmap
    DeleteObject(bmp); DeleteDC(memDC);
    EndPaint(hw,&ps);
}

// ── PNG export ──────────────────────────────────────────────────
// Finds the GDI+ encoder CLSID for a given MIME type (e.g. "image/png").
static int GetEncoderClsid(const WCHAR* format, CLSID* pClsid) {
    UINT num = 0, size = 0;
    GetImageEncodersSize(&num, &size);
    if (size == 0) return -1;
    std::vector<BYTE> buf(size);
    ImageCodecInfo* pInfo = reinterpret_cast<ImageCodecInfo*>(buf.data());
    GetImageEncoders(num, size, pInfo);
    for (UINT i = 0; i < num; ++i) {
        if (wcscmp(pInfo[i].MimeType, format) == 0) {
            *pClsid = pInfo[i].Clsid;
            return (int)i;
        }
    }
    return -1;
}

// Render the current training graph to an off-screen bitmap and save as PNG.
// Output: <dataDir>/training_progress/training_graph.png
void SaveGraphPng(const std::string& dataDir) {
    // Snapshot the current graph data under lock
    std::vector<TrainPoint> pts;
    { std::lock_guard<std::mutex> lk(g_st.mtx); pts = g_st.pts; }
    if (pts.size() < 2) return;  // nothing useful to draw

    const int W2 = 1280, H2 = 720;

    // Create off-screen GDI+ bitmap and graphics context
    Bitmap offscreen(W2, H2, PixelFormat32bppARGB);
    Graphics g(&offscreen);
    g.SetSmoothingMode(SmoothingModeAntiAlias);

    SolidBrush bgBr(Color(255,16,16,24));
    g.FillRectangle(&bgBr, 0, 0, W2, H2);

    // ── Compute generation boundaries ──
    std::vector<int> genBounds;
    std::vector<int> genNumbers;
    for (size_t i = 1; i < pts.size(); i++) {
        if (pts[i].gen != pts[i-1].gen) {
            genBounds.push_back((int)i);
            genNumbers.push_back(pts[i].gen);
        }
    }

    // ---- Layout: horizontal split if phase data exists ----
    bool hasPhase = false;
    for (auto& p : pts) if (p.hasPhase) { hasPhase = true; break; }
    bool hasAcc = false;
    for (auto& p : pts) if (p.hasAcc) { hasAcc = true; break; }
    bool hasLR = false;
    for (auto& p : pts) if (p.hasLR) { hasLR = true; break; }

    float leftW  = (float)W2;
    float rightX = 0, rightW = 0;
    float hGap = 6.0f;
    if (hasPhase) {
        rightW = (float)W2 / 3.0f;
        leftW  = (float)W2 - rightW - hGap;
        rightX = leftW + hGap;
    }

    // ---- Stacked panels (Loss / Acc / LR) ----
    float lossWeight = 3.0f;
    float accWeight  = hasAcc ? 1.0f : 0.0f;
    float lrWeight   = hasLR  ? 1.0f : 0.0f;
    float totalWeight = lossWeight + accWeight + lrWeight;

    int numPanels = 1 + (hasAcc?1:0) + (hasLR?1:0);
    float ml = 52, mr = 16;
    float panelGap = 4.0f;
    float availH = (float)H2 - panelGap * (numPanels - 1);
    float lossH = availH * (lossWeight / totalWeight);
    float accH  = availH * (accWeight  / totalWeight);
    float lrH   = availH * (lrWeight   / totalWeight);
    float curY = 0;
    float gw = leftW - ml - mr;
    if (gw < 20) gw = 20;

    auto xfLeft = [&](int i) -> float {
        return ml + (float)i / (float)(pts.size()-1) * gw;
    };

    // Helper: draw generation boundary lines
    auto drawGenBounds = [&](float panelTop, float panelH, std::function<float(int)> xfFunc) {
        Pen genPen(Color(80, 180, 180, 200), 1.0f);
        genPen.SetDashStyle(DashStyleDot);
        Font genFnt(L"Consolas", 6.5f);
        SolidBrush genBr(Color(140, 180, 180, 200));
        for (size_t b = 0; b < genBounds.size(); b++) {
            float gx = xfFunc(genBounds[b]);
            g.DrawLine(&genPen, gx, panelTop + 14, gx, panelTop + panelH - 2);
            std::wstring lbl = L"G" + std::to_wstring(genNumbers[b]);
            g.DrawString(lbl.c_str(), -1, &genFnt, PointF(gx + 2, panelTop + panelH - 13), &genBr);
        }
    };

    // ---- Loss panel ----
    {
        float pt_h = lossH;
        float pt_top = curY;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        SolidBrush panelBg(Color(255,20,20,30));
        g.FillRectangle(&panelBg, 0.0f, pt_top, leftW, pt_h);

        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Loss", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);

        double minV=1e9, maxV=-1e9;
        for (auto& p2 : pts) {
            if (!p2.hasLoss) continue;
            minV = (std::min)(minV, p2.train); maxV = (std::max)(maxV, p2.train);
            if (p2.hasVal) { minV = (std::min)(minV, p2.val); maxV = (std::max)(maxV, p2.val); }
        }
        if (maxV <= minV) maxV = minV + 0.1;
        double rng = maxV - minV;
        minV -= rng*0.05; maxV += rng*0.05; rng = maxV - minV;
        auto yf = [&](double v) -> float { return mt2 + (float)((maxV-v)/rng)*gh2; };

        Pen gridPen(Color(40,60,60,80), 1.0f);
        Font gridFnt(L"Consolas", 7.0f);
        SolidBrush gridBr(Color(255,80,80,100));
        for (int i=0; i<=4; i++) {
            float y2 = mt2 + gh2*i/4;
            g.DrawLine(&gridPen, ml, y2, ml+gw, y2);
            double val = maxV - rng*i/4;
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(4)<<val;
            g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
        }
        drawGenBounds(pt_top, pt_h, xfLeft);

        int bestTrainIdx=-1, bestValIdx=-1;
        double bestTrain=1e9, bestVal=1e9;
        for (size_t i=0; i<pts.size(); i++) {
            if (!pts[i].hasLoss) continue;
            if (pts[i].train < bestTrain) { bestTrain=pts[i].train; bestTrainIdx=(int)i; }
            if (pts[i].hasVal && pts[i].val < bestVal) { bestVal=pts[i].val; bestValIdx=(int)i; }
        }

        Pen trainPen(Color(255,65,125,245), 1.8f);
        for (size_t i=1; i<pts.size(); i++)
            if (!pts[i].hasLoss || !pts[i-1].hasLoss) continue;
            g.DrawLine(&trainPen, xfLeft((int)i-1), yf(pts[i-1].train), xfLeft((int)i), yf(pts[i].train));

        Pen valPen(Color(255,245,160,60), 1.8f);
        { bool st=false; float px2=0,py2=0;
          for (size_t i=0; i<pts.size(); i++) {
            if (!pts[i].hasVal) continue;
            float cx=xfLeft((int)i), cy=yf(pts[i].val);
            if (st) g.DrawLine(&valPen,px2,py2,cx,cy);
            px2=cx; py2=cy; st=true;
        }}

        if (bestTrainIdx >= 0) {
            float bx = xfLeft(bestTrainIdx), by = yf(bestTrain);
            SolidBrush mk(Color(255,65,125,245));
            PointF dm[4] = {{bx,by-5},{bx+5,by},{bx,by+5},{bx-5,by}};
            g.FillPolygon(&mk, dm, 4);
            Font mf(L"Consolas",6.5f);
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(5)<<bestTrain;
            std::wstring lbl = L"Best: " + ss.str();
            float lx2 = bx + 7; if (lx2 + 80 > leftW) lx2 = bx - 90;
            g.DrawString(lbl.c_str(),-1,&mf,PointF(lx2,by-5),&mk);
        }
        if (bestValIdx >= 0) {
            float bx = xfLeft(bestValIdx), by = yf(bestVal);
            SolidBrush mk(Color(255,245,160,60));
            PointF dm[4] = {{bx,by-5},{bx+5,by},{bx,by+5},{bx-5,by}};
            g.FillPolygon(&mk, dm, 4);
            Font mf(L"Consolas",6.5f);
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(5)<<bestVal;
            std::wstring lbl = L"Best: " + ss.str();
            float lx2 = bx + 7; if (lx2 + 80 > leftW) lx2 = bx - 90;
            g.DrawString(lbl.c_str(),-1,&mf,PointF(lx2,by+2),&mk);
        }

        { Font lf(L"Segoe UI",7.5f);
          float lx = leftW - mr - 120, ly = pt_top + 3;
          SolidBrush b1(Color(255,65,125,245)); Pen lp1(Color(255,65,125,245),2);
          g.DrawLine(&lp1,lx,ly+5,lx+12,ly+5);
          g.DrawString(L"train",-1,&lf,PointF(lx+14,ly-1),&b1);
          SolidBrush b2(Color(255,245,160,60)); Pen lp2(Color(255,245,160,60),2);
          g.DrawLine(&lp2,lx+52,ly+5,lx+64,ly+5);
          g.DrawString(L"val",-1,&lf,PointF(lx+66,ly-1),&b2);
        }
        curY += pt_h + panelGap;
    }

    // ---- Accuracy panel ----
    if (hasAcc) {
        float pt_top = curY, pt_h = accH;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        SolidBrush panelBg(Color(255,18,22,28));
        g.FillRectangle(&panelBg, 0.0f, pt_top, leftW, pt_h);
        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Accuracy", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);

        double minA=1e9, maxA=-1e9;
        for (auto& p2 : pts) {
            if (!p2.hasAcc) continue;
            minA=(std::min)(minA,p2.accuracy); maxA=(std::max)(maxA,p2.accuracy);
        }
        if (maxA<=minA) maxA=minA+0.1;
        double rng=maxA-minA; minA-=rng*0.05; maxA+=rng*0.05; rng=maxA-minA;
        auto yf=[&](double v)->float{return mt2+(float)((maxA-v)/rng)*gh2;};
        Pen gridPen(Color(40,60,60,80),1.0f); Font gridFnt(L"Consolas",7.0f);
        SolidBrush gridBr(Color(255,80,80,100));
        for (int i=0;i<=4;i++){
            float y2=mt2+gh2*i/4;
            g.DrawLine(&gridPen,ml,y2,ml+gw,y2);
            double val=maxA-rng*i/4;
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(3)<<val;
            g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
        }
        drawGenBounds(pt_top, pt_h, xfLeft);
        Pen accPen(Color(255,0,200,120),1.8f);
        int bestAccIdx=-1; double bestAcc=-1;
        { bool st=false; float px3=0,py3=0;
          for (size_t i=0;i<pts.size();i++){
            if (!pts[i].hasAcc) continue;
            float cx=xfLeft((int)i),cy=yf(pts[i].accuracy);
            if (st) g.DrawLine(&accPen,px3,py3,cx,cy);
            px3=cx; py3=cy; st=true;
            if (pts[i].accuracy>bestAcc){bestAcc=pts[i].accuracy; bestAccIdx=(int)i;}
        }}
        if (bestAccIdx>=0){
            float bx=xfLeft(bestAccIdx),by=yf(bestAcc);
            SolidBrush mk(Color(255,0,200,120));
            PointF dm[4]={{bx,by-5},{bx+5,by},{bx,by+5},{bx-5,by}};
            g.FillPolygon(&mk,dm,4);
            Font mf(L"Consolas",6.5f);
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(4)<<bestAcc;
            g.DrawString((L"Best: "+ss.str()).c_str(),-1,&mf,PointF(bx+7,by-5),&mk);
        }
        curY += pt_h + panelGap;
    }

    // ---- Learning Rate panel ----
    if (hasLR) {
        float pt_top = curY, pt_h = lrH;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        SolidBrush panelBg(Color(255,22,18,28));
        g.FillRectangle(&panelBg, 0.0f, pt_top, leftW, pt_h);
        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Learning Rate", -1, &titleFnt, PointF(ml, pt_top+2), &titleBr);

        double minL=1e9, maxL=-1e9;
        for (auto& p2 : pts) {
            if (!p2.hasLR) continue;
            minL=(std::min)(minL,p2.lr); maxL=(std::max)(maxL,p2.lr);
        }
        if (maxL<=minL) maxL=minL+0.0001;
        double rng=maxL-minL; minL-=rng*0.05; maxL+=rng*0.05; rng=maxL-minL;
        auto yf=[&](double v)->float{return mt2+(float)((maxL-v)/rng)*gh2;};
        Pen gridPen(Color(40,60,60,80),1.0f); Font gridFnt(L"Consolas",7.0f);
        SolidBrush gridBr(Color(255,80,80,100));
        for (int i=0;i<=4;i++){
            float y2=mt2+gh2*i/4;
            g.DrawLine(&gridPen,ml,y2,ml+gw,y2);
            double val=maxL-rng*i/4;
            std::wostringstream ss; ss<<std::scientific<<std::setprecision(2)<<val;
            g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(2,y2-6),&gridBr);
        }
        drawGenBounds(pt_top, pt_h, xfLeft);
        Pen lrPen(Color(255,180,80,220),1.8f);
        { bool st=false; float px4=0,py4=0;
          for (size_t i=0;i<pts.size();i++){
            if (!pts[i].hasLR) continue;
            float cx=xfLeft((int)i),cy=yf(pts[i].lr);
            if (st) g.DrawLine(&lrPen,px4,py4,cx,cy);
            px4=cx; py4=cy; st=true;
        }}
        curY += pt_h + panelGap;
    }

    // ==== Phase Loss panel (right 1/3) ====
    if (hasPhase) {
        float pml = rightX + 48, pmr2 = 12;
        float pgw = rightW - 48 - pmr2;
        if (pgw < 20) pgw = 20;
        float pt_top = 0, pt_h = (float)H2;
        float mt2 = pt_top + 18, mb2 = pt_top + pt_h - 14;
        float gh2 = mb2 - mt2; if (gh2 < 10) gh2 = 10;

        SolidBrush panelBg(Color(255,18,20,32));
        g.FillRectangle(&panelBg, rightX, pt_top, rightW, pt_h);
        Pen sepPen(Color(60,80,80,120), 1.0f);
        g.DrawLine(&sepPen, rightX, 0.0f, rightX, (float)H2);

        Font titleFnt(L"Segoe UI", 8.0f, FontStyleBold);
        SolidBrush titleBr(Color(255,100,100,120));
        g.DrawString(L"Phase Loss", -1, &titleFnt, PointF(pml, pt_top+2), &titleBr);

        auto xfRight = [&](int i) -> float {
            return pml + (float)i / (float)(pts.size()-1) * pgw;
        };

        double minP=1e9, maxP=-1e9;
        for (auto& p2 : pts) {
            if (!p2.hasPhase) continue;
            minP = (std::min)(minP, (std::min)(p2.openingLoss, (std::min)(p2.middlegameLoss, p2.endgameLoss)));
            maxP = (std::max)(maxP, (std::max)(p2.openingLoss, (std::max)(p2.middlegameLoss, p2.endgameLoss)));
        }
        if (maxP <= minP) maxP = minP + 0.1;
        double rng = maxP - minP;
        minP -= rng*0.05; maxP += rng*0.05; rng = maxP - minP;
        auto yf = [&](double v) -> float { return mt2 + (float)((maxP-v)/rng)*gh2; };

        Pen gridPen(Color(40,60,60,80), 1.0f);
        Font gridFnt(L"Consolas", 7.0f);
        SolidBrush gridBr(Color(255,80,80,100));
        for (int i=0; i<=4; i++) {
            float y2 = mt2 + gh2*i/4;
            g.DrawLine(&gridPen, pml, y2, pml+pgw, y2);
            double val = maxP - rng*i/4;
            std::wostringstream ss; ss<<std::fixed<<std::setprecision(4)<<val;
            g.DrawString(ss.str().c_str(),-1,&gridFnt,PointF(rightX+2,y2-6),&gridBr);
        }
        drawGenBounds(pt_top, pt_h, xfRight);

        Pen openPen(Color(255,76,175,80), 1.8f);
        { bool st=false; float px=0,py=0;
          for (size_t i=0;i<pts.size();i++){
            if (!pts[i].hasPhase) continue;
            float cx=xfRight((int)i), cy=yf(pts[i].openingLoss);
            if (st) g.DrawLine(&openPen,px,py,cx,cy);
            px=cx; py=cy; st=true;
        }}
        Pen midPen(Color(255,66,165,245), 1.8f);
        { bool st=false; float px=0,py=0;
          for (size_t i=0;i<pts.size();i++){
            if (!pts[i].hasPhase) continue;
            float cx=xfRight((int)i), cy=yf(pts[i].middlegameLoss);
            if (st) g.DrawLine(&midPen,px,py,cx,cy);
            px=cx; py=cy; st=true;
        }}
        Pen endPen(Color(255,244,67,54), 1.8f);
        { bool st=false; float px=0,py=0;
          for (size_t i=0;i<pts.size();i++){
            if (!pts[i].hasPhase) continue;
            float cx=xfRight((int)i), cy=yf(pts[i].endgameLoss);
            if (st) g.DrawLine(&endPen,px,py,cx,cy);
            px=cx; py=cy; st=true;
        }}

        // Legend
        Font lf(L"Segoe UI", 7.0f);
        float lx = rightX + rightW - pmr2 - 180, ly = pt_top + 3;
        SolidBrush bO(Color(255,76,175,80));  Pen lpO(Color(255,76,175,80),2);
        g.DrawLine(&lpO, lx, ly+5, lx+10, ly+5);
        g.DrawString(L"Open", -1, &lf, PointF(lx+12, ly-1), &bO);
        SolidBrush bM(Color(255,66,165,245)); Pen lpM(Color(255,66,165,245),2);
        g.DrawLine(&lpM, lx+50, ly+5, lx+60, ly+5);
        g.DrawString(L"Mid", -1, &lf, PointF(lx+62, ly-1), &bM);
        SolidBrush bE(Color(255,244,67,54));   Pen lpE(Color(255,244,67,54),2);
        g.DrawLine(&lpE, lx+96, ly+5, lx+106, ly+5);
        g.DrawString(L"End", -1, &lf, PointF(lx+108, ly-1), &bE);
    }

    // ── Save to PNG ──
    fs::path outDir = fs::path(exeDir()) / "training progress";
    fs::create_directories(outDir);
    fs::path outPath = outDir / "training_graph.png";

    CLSID pngClsid;
    if (GetEncoderClsid(L"image/png", &pngClsid) >= 0) {
        offscreen.Save(outPath.wstring().c_str(), &pngClsid, nullptr);
    }
}

LRESULT CALLBACK GraphProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg==WM_PAINT) { DrawGraph(hw); return 0; }
    if (msg==WM_ERASEBKGND) return 1;
    if (msg==WM_MOUSEMOVE) {
        POINT pt = {GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
        g_graph.mousePt = pt;
        RECT rc2; GetClientRect(hw, &rc2);
        int W2 = rc2.right;
        std::vector<TrainPoint> pts2;
        { std::lock_guard<std::mutex> lk(g_st.mtx); pts2 = g_st.pts; }

        // Compute panel regions (mirrors DrawGraph logic)
        float leftW = (float)W2;
        float rightX2 = 0, rightW2 = 0;
        float hGap = 6;
        if (g_graph.showPhase) {
            rightW2 = (float)W2 / 3.0f;
            leftW   = (float)W2 - rightW2 - hGap;
            rightX2 = leftW + hGap;
        }

        float fx = -1.0f;
        if (pts2.size() >= 2) {
            if (g_graph.showPhase && pt.x >= (int)rightX2) {
                // Mouse is in the right (phase) panel
                float pml = rightX2 + 48;
                float pgw = rightW2 - 48 - 12;
                if (pgw > 0) fx = (float)(pt.x - pml) / pgw;
            } else {
                // Mouse is in the left (stacked) panels
                float ml2 = 52, mr2 = 16;
                float gw2 = leftW - ml2 - mr2;
                if (gw2 > 0) fx = (float)(pt.x - ml2) / gw2;
            }
        }

        if (fx >= 0 && fx <= 1.0f && pts2.size() >= 2) {
            int idx2 = (int)(fx * (pts2.size()-1) + 0.5f);
            if (idx2 < 0) idx2 = 0;
            if (idx2 >= (int)pts2.size()) idx2 = (int)pts2.size()-1;
            g_graph.hoverIdx = idx2;
        } else {
            g_graph.hoverIdx = -1;
        }
        // Detect which panel the mouse is over
        {
            int newPanel = -1;
            float fy = (float)pt.y;
            bool inRight = g_graph.showPhase && pt.x >= (int)rightX2;
            if (inRight) {
                if (g_graph.panelBounds[3].active && fy >= g_graph.panelBounds[3].top && fy < g_graph.panelBounds[3].bottom)
                    newPanel = 3;
            } else {
                for (int pi = 0; pi < 3; ++pi) {
                    if (g_graph.panelBounds[pi].active && fy >= g_graph.panelBounds[pi].top && fy < g_graph.panelBounds[pi].bottom) {
                        newPanel = pi;
                        break;
                    }
                }
            }
            g_graph.hoverPanel = newPanel;
        }
        InvalidateRect(hw, nullptr, FALSE);
        TRACKMOUSEEVENT tme{}; tme.cbSize=sizeof(tme);
        tme.dwFlags=TME_LEAVE; tme.hwndTrack=hw;
        TrackMouseEvent(&tme);
        return 0;
    }
    if (msg==WM_MOUSELEAVE) {
        g_graph.hoverIdx=-1; g_graph.mousePt={-1,-1}; g_graph.hoverPanel=-1;
        InvalidateRect(hw,nullptr,FALSE);
        return 0;
    }
    return DefWindowProcW(hw,msg,wp,lp);
}

