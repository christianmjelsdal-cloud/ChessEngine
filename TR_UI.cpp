// TR_UI.cpp  --  UI construction, tooltips, config panel, log
#include "TR_Types.h"
#include "TR_Globals.h"
#include "TR_Fwd.h"

// -- Log line classification --
LogColor ClassifyLogLine(const std::string& line) {
    // Errors / warnings (highest priority)
    if (line.find("[ERR]") != std::string::npos ||
        line.find("[ERROR]") != std::string::npos ||
        line.find("[WARN]") != std::string::npos ||
        line.find("WARNING") != std::string::npos ||
        line.find("failed") != std::string::npos ||
        line.find("Failed") != std::string::npos ||
        line.find("Error") != std::string::npos)
        return LC_ERROR;

    // Success / completion
    if (line.find("complete") != std::string::npos ||
        line.find("Complete") != std::string::npos ||
        line.find("done") != std::string::npos ||
        line.find("Done") != std::string::npos ||
        line.find("success") != std::string::npos ||
        line.find("Success") != std::string::npos ||
        line.find("saved") != std::string::npos ||
        line.find("=== Pipeline") != std::string::npos)
        return LC_SUCCESS;

    // Batch progress lines (dark cyan) — must check before ETA/Time to avoid LC_META
    if (line.find("Batch") != std::string::npos &&
        (line.find("Ep ") != std::string::npos || line.find("Loss:") != std::string::npos))
        return LC_SELFPLAY;

    // Self-play phase
    if (line.find("[SelfPlay]") != std::string::npos ||
        line.find("self-play") != std::string::npos ||
        line.find("Self-play") != std::string::npos ||
        line.find("--generate") != std::string::npos ||
        line.find("games/s") != std::string::npos)
        return LC_SELFPLAY;

    // Training phase (epoch summaries, phase loss, etc.)
    if (line.find("Train Loss") != std::string::npos ||
        line.find("Train:") != std::string::npos ||
        line.find("Val Loss") != std::string::npos ||
        line.find("Val:") != std::string::npos ||
        line.find("Epoch") != std::string::npos ||
        line.find("train_nnue") != std::string::npos ||
        line.find("LR:") != std::string::npos ||
        line.find("Phase loss") != std::string::npos ||
        line.find("Phase distribution") != std::string::npos ||
        line.find("Acc:") != std::string::npos)
        return LC_TRAINING;

    // Progress / milestones
    if (line.find("---") != std::string::npos ||
        line.find("===") != std::string::npos ||
        line.find("Generation") != std::string::npos ||
        line.find("Gen ") != std::string::npos ||
        line.find("ELO") != std::string::npos ||
        line.find("Elo") != std::string::npos ||
        line.find("Best loss") != std::string::npos)
        return LC_PROGRESS;

    // Metadata / timing
    if (line.find("[CMD]") != std::string::npos ||
        line.find("ETA") != std::string::npos ||
        line.find("elapsed") != std::string::npos ||
        line.find("Elapsed") != std::string::npos ||
        line.find("Time:") != std::string::npos ||
        line.find("pos/s") != std::string::npos ||
        line.find("Loaded") != std::string::npos ||
        line.find("Loading") != std::string::npos ||
        line.find("Using device") != std::string::npos ||
        line.find("CPU threads") != std::string::npos)
        return LC_META;

    return LC_GENERAL;
}


// ── Config panel helpers ──────────────────────────────────────────
static HWND mkLabel(HWND parent, const wchar_t* txt, int x, int y, int w, int h) {
    HWND hw = CreateWindowExW(0,L"STATIC",txt,WS_CHILD|WS_VISIBLE|SS_LEFT|SS_NOTIFY,
                              x,y,w,h,parent,nullptr,g_hInst,nullptr);
    SendMessageW(hw, WM_SETFONT, (WPARAM)g_ui.fUI, TRUE);
    SetWindowLongPtrW(hw, GWLP_ID, (LONG_PTR)hw); // use HWND as id for tooltip
    return hw;
}

// FIX: Child controls (edits, checkboxes, combos) eat WM_MOUSEWHEEL when
// focused, preventing the config panel from scrolling.  Forward to parent.
static LRESULT CALLBACK ChildScrollProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp,
                                         UINT_PTR, DWORD_PTR) {
    if (msg == WM_MOUSEWHEEL) {
        HWND parent = GetParent(hw);
        if (parent) return SendMessage(parent, msg, wp, lp);
    }
    return DefSubclassProc(hw, msg, wp, lp);
}

static HWND mkEdit(HWND parent, int id, const wchar_t* def, int x, int y, int w, int h) {
    HWND hw = CreateWindowExW(WS_EX_CLIENTEDGE,L"EDIT",def,
                              WS_CHILD|WS_VISIBLE|ES_AUTOHSCROLL,
                              x,y,w,h,parent,(HMENU)(LONG_PTR)id,g_hInst,nullptr);
    SendMessageW(hw, WM_SETFONT, (WPARAM)g_ui.fUI, TRUE);
    SetWindowSubclass(hw, ChildScrollProc, 3, 0);
    return hw;
}

static HWND mkCheck(HWND parent, int id, const wchar_t* txt, int x, int y, int w, int h, bool chk) {
    HWND hw = CreateWindowExW(0,L"BUTTON",txt,
                              WS_CHILD|WS_VISIBLE|BS_AUTOCHECKBOX,
                              x,y,w,h,parent,(HMENU)(LONG_PTR)id,g_hInst,nullptr);
    SendMessageW(hw, WM_SETFONT, (WPARAM)g_ui.fUI, TRUE);
    Button_SetCheck(hw, chk ? BST_CHECKED : BST_UNCHECKED);
    SetWindowSubclass(hw, ChildScrollProc, 3, 0);
    return hw;
}


// ── Custom tooltip system (avoids TOOLTIPS_CLASS white-box bugs) ──────────

LRESULT CALLBACK TipWndProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hw, &ps);
        RECT rc; GetClientRect(hw, &rc);
        // Dark background
        HBRUSH br = CreateSolidBrush(RGB(50, 50, 55));
        FillRect(hdc, &rc, br);
        DeleteObject(br);
        // Border
        HPEN pen = CreatePen(PS_SOLID, 1, RGB(100, 100, 110));
        HPEN oldPen = (HPEN)SelectObject(hdc, pen);
        HBRUSH oldBr = (HBRUSH)SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Rectangle(hdc, rc.left, rc.top, rc.right, rc.bottom);
        SelectObject(hdc, oldPen);
        SelectObject(hdc, oldBr);
        DeleteObject(pen);
        // Text
        const wchar_t* text = (const wchar_t*)GetWindowLongPtrW(hw, GWLP_USERDATA);
        if (text) {
            SetBkMode(hdc, TRANSPARENT);
            SetTextColor(hdc, RGB(220, 220, 220));
            HFONT oldFont = (HFONT)SelectObject(hdc, g_ui.fUI);
            RECT textRc = {6, 4, rc.right - 6, rc.bottom - 4};
            DrawTextW(hdc, text, -1, &textRc, DT_LEFT | DT_WORDBREAK | DT_NOPREFIX);
            SelectObject(hdc, oldFont);
        }
        EndPaint(hw, &ps);
        return 0;
    }
    if (msg == WM_NCHITTEST) return HTTRANSPARENT; // clicks pass through
    return DefWindowProcW(hw, msg, wp, lp);
}

static void ShowCustomTip(HWND hCtrl) {
    auto it = g_tip.tipMap.find(hCtrl);
    if (it == g_tip.tipMap.end()) return;
    if (g_tip.current == hCtrl) return;
    g_tip.current = hCtrl;

    const wchar_t* text = it->second;
    SetWindowLongPtrW(g_tip.hWnd, GWLP_USERDATA, (LONG_PTR)text);

    // Calculate size needed
    HDC hdc = GetDC(g_tip.hWnd);
    HFONT oldFont = (HFONT)SelectObject(hdc, g_ui.fUI);
    RECT calcRc = {0, 0, 280, 0};
    DrawTextW(hdc, text, -1, &calcRc, DT_CALCRECT | DT_WORDBREAK | DT_NOPREFIX);
    SelectObject(hdc, oldFont);
    ReleaseDC(g_tip.hWnd, hdc);

    int tipW = calcRc.right + 14;
    int tipH = calcRc.bottom + 10;
    if (tipW < 80) tipW = 80;

    // Position below the control
    RECT ctrlRc;
    GetWindowRect(hCtrl, &ctrlRc);
    int tipX = ctrlRc.left;
    int tipY = ctrlRc.bottom + 2;

    // Keep on screen
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    if (tipX + tipW > screenW) tipX = screenW - tipW;
    if (tipY + tipH > screenH) tipY = ctrlRc.top - tipH - 2;

    SetWindowPos(g_tip.hWnd, HWND_TOPMOST, tipX, tipY, tipW, tipH, SWP_NOACTIVATE);
    ShowWindow(g_tip.hWnd, SW_SHOWNOACTIVATE);
    InvalidateRect(g_tip.hWnd, nullptr, TRUE);
}

static void HideCustomTip() {
    ShowWindow(g_tip.hWnd, SW_HIDE);
    g_tip.current = nullptr;
}

static LRESULT CALLBACK TipSubclassProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp,
                                         UINT_PTR uIdSubclass, DWORD_PTR dwRefData) {
    if (msg == WM_MOUSEMOVE) {
        ShowCustomTip(hw);
        TRACKMOUSEEVENT tme{};
        tme.cbSize = sizeof(tme);
        tme.dwFlags = TME_LEAVE;
        tme.hwndTrack = hw;
        TrackMouseEvent(&tme);
    } else if (msg == WM_MOUSELEAVE) {
        HideCustomTip();
    }
    return DefSubclassProc(hw, msg, wp, lp);
}

static void AddTooltip(HWND hCtrl, const wchar_t* text) {
    g_tip.tipMap[hCtrl] = text;
    SetWindowSubclass(hCtrl, TipSubclassProc, 2, 0);
}

void BuildConfigPane(HWND pane, int PW) {
    int lw = 110, ew = PW - lw - 24, ex = lw + 12, lx = 8;
    int y = 8, dy = 24;


    // ── Section: Presets ──────────────────────────────────────────
    {
        HWND lbl = mkLabel(pane, L"Preset", lx, y+2, lw, 18);
        g_ui.hPreset = CreateWindowExW(0, L"COMBOBOX", nullptr,
                                    WS_CHILD|WS_VISIBLE|CBS_DROPDOWNLIST|WS_VSCROLL,
                                    ex, y, ew, 200, pane,
                                    (HMENU)(LONG_PTR)ID_COMBO_PRESET, g_hInst, nullptr);
        SendMessageW(g_ui.hPreset, WM_SETFONT, (WPARAM)g_ui.fUI, TRUE);
        SetWindowSubclass(g_ui.hPreset, ChildScrollProc, 3, 0);
        AddTooltip(lbl, L"WHAT: Selects the active configuration preset that populates all settings below. Built-in presets provide tuned starting points for common training scenarios.\n\nHOW TO USE: Pick a built-in preset to start, then adjust individual settings as needed. Use 'Save As...' to save your custom configuration. Custom presets appear in this list and can be deleted; built-in presets cannot.");
        y += dy;
    }
    // Save As / Delete buttons
    {
        int halfW = (ew - 4) / 2;
        g_ui.hBtnSave = CreateWindowExW(0, L"BUTTON", L"Save As...",
                                     WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                                     ex, y, halfW, 22, pane,
                                     (HMENU)(LONG_PTR)ID_BTN_SAVE_PRESET, g_hInst, nullptr);
        SendMessageW(g_ui.hBtnSave, WM_SETFONT, (WPARAM)g_ui.fUI, TRUE);
        SetWindowSubclass(g_ui.hBtnSave, ChildScrollProc, 3, 0);
        g_ui.hBtnDel = CreateWindowExW(0, L"BUTTON", L"Delete",
                                    WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                                    ex + halfW + 4, y, halfW, 22, pane,
                                    (HMENU)(LONG_PTR)ID_BTN_DEL_PRESET, g_hInst, nullptr);
        SendMessageW(g_ui.hBtnDel, WM_SETFONT, (WPARAM)g_ui.fUI, TRUE);
        SetWindowSubclass(g_ui.hBtnDel, ChildScrollProc, 3, 0);
        EnableWindow(g_ui.hBtnDel, FALSE);
        y += dy + 2;
    }

    // ── Section: Graph Toggles ───────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Graph Panels ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_ui.fUI,TRUE);
        y += 20;
        int chkW = (PW - 24) / 5;
        g_ui.hChkGLoss = mkCheck(pane, ID_CHK_GRAPH_LOSS, L"Loss", lx, y, chkW, 20, true);
        g_ui.hChkGAcc  = mkCheck(pane, ID_CHK_GRAPH_ACC,  L"Accuracy", lx+chkW, y, chkW, 20, true);
        g_ui.hChkGLR    = mkCheck(pane, ID_CHK_GRAPH_LR,    L"LR",     lx+chkW*2, y, chkW, 20, true);
        g_ui.hChkGPhase = mkCheck(pane, ID_CHK_GRAPH_PHASE, L"Phases", lx+chkW*3, y, chkW, 20, true);
        g_ui.hChkGNPS   = mkCheck(pane, ID_CHK_GRAPH_NPS,   L"NPS",   lx+chkW*4, y, chkW, 20, true);
        AddTooltip(g_ui.hChkGLoss,  L"WHAT: Shows or hides the Loss curve panel, which plots training loss and validation loss across epochs. Best-achieved values are highlighted with diamond markers.\n\nWHY: The loss curves are your primary diagnostic tool. A healthy run shows both curves declining together. If training loss drops but validation loss rises, the model is overfitting. If both plateau early, try increasing the learning rate or adding more data.");
        AddTooltip(g_ui.hChkGAcc,   L"WHAT: Shows or hides the Accuracy panel, which plots move prediction accuracy across epochs when reported by the training script.\n\nWHY: Accuracy measures how often the model's top move matches the best move from the training data. It provides a complementary view to loss -- a model can have low loss but poor accuracy if it spreads probability too evenly. Rising accuracy confirms the model is learning meaningful patterns.");
        AddTooltip(g_ui.hChkGLR,    L"WHAT: Shows or hides the Learning Rate schedule panel, which plots the effective learning rate at each epoch or step.\n\nWHY: Visualizing the LR schedule helps verify that cosine annealing, warm restarts, and warmup are working as expected. Unexpected LR behavior (flat when it should decay, or spikes) often explains sudden training instability.");
        AddTooltip(g_ui.hChkGPhase, L"WHAT: Shows or hides the Phase Loss panel, which breaks down training loss by game phase: Opening, Middlegame, and Endgame.\n\nWHY: Phase breakdown reveals where the engine struggles most. High endgame loss suggests the model needs more endgame training data or a higher mate boost. High opening loss may indicate insufficient opening book diversity. Balanced phase losses indicate a well-rounded model.");
        AddTooltip(g_ui.hChkGNPS,   L"WHAT: Shows or hides the NPS (Nodes Per Second) panel, which plots self-play search speed across generations.\n\nWHY: NPS directly reflects how fast the engine searches during self-play. Rising NPS across gens indicates search improvements are working. A sudden drop may indicate a regression or a harder position set being generated.");
        y += dy;
        g_ui.hChkMute = mkCheck(pane, ID_CHK_MUTE_SOUNDS, L"\xD83D\xDD07 Mute Sounds", lx, y, PW-16, 20, false);
        AddTooltip(g_ui.hChkMute, L"WHAT: Mutes the notification sounds that play when self-play and generation complete.\n\nWHY: Handy if you're running training overnight or just prefer silence.");
        y += dy;
    }

    // ── Section header: Self-Play ─────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Self-Play ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_ui.fUI,TRUE);
        y += 20;
    }

    // Generations
    {
        HWND lbl = mkLabel(pane, L"Generations", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_GENS, L"10", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_GENS] = ed;
        AddTooltip(lbl, L"WHAT: The total number of self-play \u2192 train cycles to execute. Each generation plays a batch of games using the current model, then trains a new model on the resulting data.\n\nWHY: Each generation produces a slightly stronger model that generates higher-quality training data for the next cycle. This feedback loop is the core of the reinforcement learning process -- more generations compound improvements, but with diminishing returns as the model approaches its architectural ceiling.\n\nWHEN TO ADJUST: For initial experiments, 5-10 generations is enough to see if training is working. For serious training runs, 30-100+ generations are typical. Very high values (200+) are safe -- early stopping or manual intervention can halt the run if progress stalls.\n\nDefault: 10 generations.");
        y += dy;
    }

    // Start Gen  (with "Latest" and "Best" buttons)
    {
        int btnW = 44;                       // width of each button
        int gap  = 3;
        int edW  = ew - btnW*2 - gap*2;     // shrink edit to make room for two buttons
        HWND lbl = mkLabel(pane, L"Start Gen", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_STARTGEN, L"0", ex, y, edW, 20);
        g_ui.edits[ID_EDIT_STARTGEN] = ed;
        int bx = ex + edW + gap;
        HWND btnLatest = CreateWindowExW(0, L"BUTTON", L"Latest",
                     WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                     bx, y, btnW, 20,
                     pane, (HMENU)(LONG_PTR)ID_BTN_LATEST_GEN, g_hInst, nullptr);
        SendMessageW(btnLatest, WM_SETFONT, (WPARAM)g_ui.fUI, TRUE);
        SetWindowSubclass(btnLatest, ChildScrollProc, 3, 0);
        bx += btnW + gap;
        HWND btnBest = CreateWindowExW(0, L"BUTTON", L"Best",
                     WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                     bx, y, btnW, 20,
                     pane, (HMENU)(LONG_PTR)ID_BTN_BEST_GEN, g_hInst, nullptr);
        SendMessageW(btnBest, WM_SETFONT, (WPARAM)g_ui.fUI, TRUE);
        SetWindowSubclass(btnBest, ChildScrollProc, 3, 0);
        AddTooltip(lbl, L"WHAT: The generation number to resume training from. The system will look for existing weights (nnue_weights_genN.bin) at this generation and continue the self-play \u2192 train loop from there.\n\nWHY: Training runs can be interrupted by crashes, power loss, or intentional stops. Resuming from a checkpoint avoids re-doing expensive self-play and training work. Starting from 0 begins a completely fresh run with random weights.\n\nWHEN TO ADJUST: Set to 0 for a brand-new training run. After an interruption, click 'Latest' to auto-detect the highest completed generation and resume seamlessly. Click 'Best' to resume from the generation with the lowest validation loss if recent generations regressed.\n\nDefault: 0 (fresh start).");
        AddTooltip(btnLatest, L"WHAT: Scans the assets folder for nnue_weights_genN.bin files and automatically fills in the highest generation number found.\n\nHOW TO USE: Click after an interrupted run to resume from the last completed generation without manually checking which files exist.");
        AddTooltip(btnBest, L"WHAT: Reads the training log and sets Start Gen to the generation that achieved the lowest validation loss.\n\nHOW TO USE: Click when you suspect recent generations have regressed (validation loss climbing). This lets you roll back to the strongest known checkpoint and try different settings from there.");
        y += dy;
    }

    // Games per Gen
    {
        HWND lbl = mkLabel(pane, L"Games per Gen", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_GAMES, L"5000", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_GAMES] = ed;
        AddTooltip(lbl, L"WHAT: The number of self-play games the engine plays against itself each generation to produce training data. Each game generates dozens to hundreds of labeled positions for the neural network to learn from.\n\nWHY: More games means more diverse positions and more robust gradient estimates during training, reducing overfitting and noise. However, each game costs CPU time proportional to the search depth. There is a sweet spot where additional games yield diminishing returns because the model can only absorb so much new information per generation.\n\nWHEN TO ADJUST: Start with 3000-5000 for fast iteration during development. Scale up to 10000-25000 for serious training runs. If training loss is noisy or the model oscillates between generations, increase games. If each generation takes too long and you want faster feedback, decrease games or use Mixed Depth to speed up generation.\n\nDefault: 5000 games per generation.");
        y += dy;
    }

    // Workers
    {
        HWND lbl = mkLabel(pane, L"Workers", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WORKERS, L"12", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_WORKERS] = ed;
        AddTooltip(lbl, L"WHAT: The number of parallel threads used to generate self-play games simultaneously. Each worker runs an independent game using the current engine weights.\n\nWHY: Self-play is CPU-bound and embarrassingly parallel -- each game is independent. Using more workers linearly reduces generation time up to your CPU's thread count. Beyond that, hyperthreading provides diminishing returns and can even slow down due to cache contention.\n\nWHEN TO ADJUST: Set to your CPU's physical thread count for maximum throughput (check Task Manager \u2192 Performance \u2192 Logical processors). If you want to keep your system responsive during training, set to 50-75%% of your thread count. Setting too high (beyond physical threads) wastes resources. Setting to 1 is useful for debugging but extremely slow for real training.\n\nDefault: 12 threads.");
        y += dy;
    }

    // Depth
    {
        HWND lbl = mkLabel(pane, L"Depth", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DEPTH, L"5", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DEPTH] = ed;
        AddTooltip(lbl, L"WHAT: The search depth (in plies) the engine uses when selecting moves during self-play. At each position, the engine searches this many half-moves ahead before choosing.\n\nWHY: Deeper searches produce stronger, more realistic games with better-quality position evaluations as training labels. Shallow games (depth 3-4) are fast but noisy -- the engine makes tactical blunders that teach bad habits. Deep games (depth 8+) produce expert-level data but are exponentially slower to generate.\n\nWHEN TO ADJUST: Depth 5-6 is the sweet spot for most NNUE training -- good move quality with reasonable speed. Use depth 7-8 for late-stage refinement when the model is already strong. Depth 3-4 is acceptable for very early training or when combined with Mixed Depth. Going beyond depth 10 is rarely worth the time cost unless you have very fast hardware.\n\nDefault: 5 plies.");
        y += dy;
    }

    // Mixed Depth %
    {
        HWND lbl = mkLabel(pane, L"Mixed Depth %", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MIXDEPTH_PCT, L"0", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_MIXDEPTH_PCT] = ed;
        AddTooltip(lbl, L"WHAT: The percentage of self-play games played at the reduced Low Depth instead of the full Depth. When set to 80, 80%% of games use Low Depth (fast) and 20%% use full Depth (strong).\n\nWHY: Playing all games at full depth is slow. Research shows that mixing a majority of fast, shallow games with a minority of deep games produces training data almost as good as all-deep data, at a fraction of the time cost. The shallow games provide volume and position diversity, while the deep games anchor the quality.\n\nWHEN TO ADJUST: Set to 0 to disable (all games at full depth). 70-80%% is optimal for most runs -- massive speed boost with minimal quality loss. Going above 90%% starts degrading data quality noticeably. If your full depth is already low (4-5), mixed depth provides less benefit.\n\nDefault: 0 (disabled -- all games at full depth).");
        y += dy;
    }

    // Mixed Low Depth
    {
        HWND lbl = mkLabel(pane, L"Low Depth", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MIXDEPTH_LOW, L"4", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_MIXDEPTH_LOW] = ed;
        AddTooltip(lbl, L"WHAT: The search depth used for the fast games when Mixed Depth is enabled. Only applies to the percentage of games specified by Mixed Depth %%.\n\nWHY: This controls the speed/quality trade-off for the fast portion of your data. A low value (3-4) maximizes throughput but produces noisier evaluations. A higher value (5-6) is slower but keeps data quality closer to full depth.\n\nWHEN TO ADJUST: Depth 4 is the recommended sweet spot -- fast enough to provide a 2-3x throughput boost while still producing reasonable move choices. Depth 3 is viable but expect more tactical blunders in the data. Depth 5+ narrows the gap with full depth, reducing the throughput benefit. This setting has no effect when Mixed Depth %% is 0.\n\nDefault: 4 plies.");
        y += dy;
    }

    // Depth Shuffle checkbox
    {
        g_ui.hChkDepthShuffle = CreateWindowExW(0, L"BUTTON", L"Depth Shuffle",
            WS_CHILD|WS_VISIBLE|BS_AUTOCHECKBOX, lx, y, lw+ew, 18, pane,
            (HMENU)(INT_PTR)ID_CHK_DEPTH_SHUFFLE, g_hInst, nullptr);
        SendMessageW(g_ui.hChkDepthShuffle, WM_SETFONT, (WPARAM)g_ui.fUI, TRUE);
        AddTooltip(g_ui.hChkDepthShuffle, L"WHAT: When enabled, games assigned to the Mixed Depth pool sample their search depth from a geometric distribution over [Low Depth, Depth) instead of all playing at Low Depth.\n\nWHY: A richer diversity of search depths in training data helps the neural network see positions evaluated at varying quality levels. Higher depths are weighted more heavily (controlled by Shuffle Bias), so most shuffled games still use strong searches, while a smaller fraction of shallow games adds throughput and diversity.\n\nWHEN TO ADJUST: Enable when your Depth minus Low Depth is at least 2 (e.g. Depth=9, Low Depth=4 gives 5 tiers). Requires Mixed Depth %% > 0 to have any effect. Leave disabled for the classic binary mixed depth behavior.\n\nDefault: off.");
        y += dy;
    }

    // Depth Shuffle Bias
    {
        HWND lbl = mkLabel(pane, L"Shuffle Bias", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DEPTH_SHUFFLE_BIAS, L"2.0", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DEPTH_SHUFFLE_BIAS] = ed;
        AddTooltip(lbl, L"WHAT: Controls the geometric weighting for Depth Shuffle. Each depth level d gets probability weight bias^(d - LowDepth). Higher bias means deeper searches are much more likely than shallow ones.\n\nWHY: Bias=2.0 means each depth tier is 2x more likely than the one below it. At bias=1.0 the distribution is uniform (equal chance for every depth). At bias=3.0+ almost all games cluster near the top depth with only rare shallow games.\n\nWHEN TO ADJUST: 2.0 is a good default -- it gives a natural exponential ramp favoring quality while still producing meaningful numbers of cheaper games. Increase to 3.0-4.0 if you want even stronger bias toward high-depth data. Decrease toward 1.0 for more even coverage across depths. Has no effect unless Depth Shuffle is enabled.\n\nDefault: 2.0");
        y += dy;
    }

    // ── Section header: Opening Diversity ──────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Opening Diversity ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_ui.fUI,TRUE);
        y += 20;
    }

    // Opening Temp
    {
        HWND lbl = mkLabel(pane, L"Opening Temp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_OPENING_TEMP, L"1.5", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_OPENING_TEMP] = ed;
        AddTooltip(lbl, L"WHAT: Softmax temperature for the first Opening Plies of each game. Higher values = wilder, more random opening moves. 0 = always pick the best move (no randomness).\n\nWHY: Without opening randomization, self-play games between identical engines produce nearly identical games. Temperature injects diversity by making the engine sometimes choose 2nd/3rd best moves, producing a wider variety of middlegame positions for training data.\n\nWHEN TO ADJUST: 1.5 is aggressive \u2014 produces very diverse openings. Decrease to 0.5-1.0 for more sensible openings that still have variety. Increase to 2.0+ if you want maximum opening chaos (useful for very early training). Set to 0 for deterministic openings.\n\nDefault: 1.5");
        y += dy;
    }

    // Opening Plies
    {
        HWND lbl = mkLabel(pane, L"Opening Plies", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_OPENING_PLIES, L"4", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_OPENING_PLIES] = ed;
        AddTooltip(lbl, L"WHAT: Number of half-moves at the start of each game that use softmax-random move selection with Opening Temp. After these plies, the engine transitions to the Softmax Plies phase (lower temperature) or best-move play.\n\nWHY: The first few moves determine the opening structure. Randomizing 4 plies (2 full moves) ensures diverse pawn structures and piece placements without making the openings too bizarre. More plies = more diverse but potentially more unrealistic game starts.\n\nWHEN TO ADJUST: 4 (2 full moves) is a strong default. Increase to 6-8 for maximum diversity. Decrease to 2 if openings are too random and producing nonsensical positions. Set to 0 to disable opening randomization entirely (all moves from best-move search).\n\nDefault: 4 plies.");
        y += dy;
    }

    // Softmax Plies
    {
        HWND lbl = mkLabel(pane, L"Softmax Plies", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SOFTMAX_PLIES, L"8", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_SOFTMAX_PLIES] = ed;
        AddTooltip(lbl, L"WHAT: Number of additional plies after the Opening Plies phase that use softmax move selection with the (lower) Softmax Temp. This creates a gradual transition from random opening play to best-move play.\n\nWHY: An abrupt switch from random to best-move creates an artificial boundary in game quality. The softmax phase provides a smooth transition \u2014 moves are mostly strong but with occasional variety. This extends diversity past the opening into the early middlegame.\n\nWHEN TO ADJUST: 8 plies (4 full moves) provides good diversity into the middlegame. Increase to 12-16 for broader coverage. Decrease to 4 or 0 if you want to minimize randomness outside the opening. The temperature during this phase is controlled by Softmax Temp.\n\nDefault: 8 plies.");
        y += dy;
    }

    // Softmax Temp
    {
        HWND lbl = mkLabel(pane, L"Softmax Temp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SOFTMAX_TEMP, L"0.5", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_SOFTMAX_TEMP] = ed;
        AddTooltip(lbl, L"WHAT: Softmax temperature for the post-opening phase (Softmax Plies). Lower than Opening Temp for more reasonable but still slightly varied move choices.\n\nWHY: After the opening, you want most moves to be strong but with occasional deviations. A temperature of 0.5 means the engine strongly prefers the best move but will sometimes play the 2nd/3rd best, keeping games varied without introducing obvious blunders.\n\nWHEN TO ADJUST: 0.5 is well-balanced. Decrease to 0.2-0.3 for nearly best-move play with very rare deviations. Increase to 0.8-1.0 for more aggressive randomization. Set to 0 to disable (only Opening Plies will be randomized). Only applies during the Softmax Plies phase.\n\nDefault: 0.5");
        y += dy;
    }

    // Root Noise
    {
        HWND lbl = mkLabel(pane, L"Root Noise \u03B5", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_ROOT_NOISE, L"0.0", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_ROOT_NOISE] = ed;
        AddTooltip(lbl, L"WHAT: Probability of replacing the best move with a randomly-weighted alternative at each position after the opening/softmax phases. Similar to Leela Chess Zero\u2019s epsilon-greedy exploration.\n\nWHY: Even after the opening, deterministic best-move play can produce repetitive game patterns. Root noise injects occasional \u201Cmistakes\u201D that keep games varied, forcing the engine to handle suboptimal positions. This produces training data for recovery play and defensive technique.\n\nWHEN TO ADJUST: 0.0 (off) is the default. Try 0.05-0.10 for subtle diversity throughout the game. 0.15-0.25 is aggressive and introduces frequent non-optimal moves. Values above 0.3 significantly degrade game quality. Use this as a complement to (not replacement for) Opening Temp.\n\nDefault: 0.0 (disabled).");
        y += dy;
    }

    // ── Section header: Recording Filters ─────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Recording Filters ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_ui.fUI,TRUE);
        y += 20;
    }

    // Record Min Ply
    {
        HWND lbl = mkLabel(pane, L"Record Min Ply", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_RECORD_MIN_PLY, L"10", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_RECORD_MIN_PLY] = ed;
        AddTooltip(lbl, L"WHAT: Positions before this ply number are not saved to the training data file. This filters out the very early opening moves.\n\nWHY: Early opening positions (first 5-10 plies) are dominated by randomized moves from the Opening Temp phase and don\u2019t represent real engine analysis. Training on these noisy positions teaches the model to value random moves, degrading evaluation quality. Filtering them out keeps the training set clean.\n\nWHEN TO ADJUST: 10 (5 full moves) filters the typical random opening. Increase to 16-20 if your Opening Plies + Softmax Plies extend further and you want to skip the entire random phase. Decrease to 4-6 if you want to train on opening positions (useful when Opening Temp is low). Set to 0 to record everything.\n\nDefault: 10 plies.");
        y += dy;
    }

    // Record Max Eval
    {
        HWND lbl = mkLabel(pane, L"Record Max Eval", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_RECORD_MAX_EVAL, L"2500", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_RECORD_MAX_EVAL] = ed;
        AddTooltip(lbl, L"WHAT: Positions where the absolute evaluation exceeds this centipawn threshold are not saved to the training data. These are already-decided positions.\n\nWHY: When one side has a crushing advantage (e.g. +25 pawns), the remaining moves provide little training value \u2014 the outcome is obvious and the positions are highly unusual. Filtering them reduces noise in the dataset and saves disk space for positions where evaluation accuracy actually matters.\n\nWHEN TO ADJUST: 2500cp (~25 pawns) is very permissive \u2014 only the most extreme positions are filtered. Decrease to 1500-2000 for tighter filtering that removes moderately one-sided positions. Decrease further to 1000 to focus training on competitive positions only. Increase to 5000+ to effectively disable the filter.\n\nDefault: 2500 centipawns.");
        y += dy;
    }

    // ── Section header: Self-Play Adjudication ─────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Self-Play Adjudication ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_ui.fUI,TRUE);
        y += 20;
    }

    // Resign Threshold (cp)
    {
        HWND lbl = mkLabel(pane, L"Resign Cp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_RESIGNCP, L"500", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_RESIGNCP] = ed;
        AddTooltip(lbl, L"WHAT: The centipawn threshold at which the engine concedes a lost game. When the engine's evaluation is worse than this value for 3 consecutive moves, it resigns instead of playing on.\n\nWHY: Without resignation, hopelessly lost games drag on for dozens of extra moves, wasting time generating low-quality training data from positions where the outcome is already decided. Resignation speeds up self-play and keeps training data focused on meaningful positions.\n\nWHEN TO ADJUST: Lower the value (e.g. 300-400) to end lost games sooner and speed up data generation. Raise it (e.g. 700-1000) if you want the engine to practice defending difficult endgames. Set to 0 to disable resignation entirely -- useful for testing but significantly slows training.\n\nDefault: 500 centipawns (roughly a rook disadvantage).");
        y += dy;
    }

    // Contempt (cp)
    {
        HWND lbl = mkLabel(pane, L"Contempt Cp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_CONTEMPT, L"25", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_CONTEMPT] = ed;
        AddTooltip(lbl, L"WHAT: A bias in centipawns that the engine adds to its evaluation to discourage accepting draws. A contempt of 25 means the engine treats a drawn position as if it were 25cp worse, making it prefer to keep playing rather than repeat moves or simplify into a dead-equal endgame.\n\nWHY: Self-play between identical engines naturally produces a high draw rate because both sides evaluate positions the same way and readily agree to repetitions. Contempt forces the engine to take risks and fight for decisive results, generating richer and more varied training data with a better balance of wins, losses, and draws.\n\nWHEN TO ADJUST: Increase (e.g. 40-75) if your draw rate is too high and you want more decisive games. Decrease (e.g. 5-15) if the engine is overextending and producing unnatural positions. Set to 0 for unbiased play -- useful for Elo testing but produces excessive draws in training.\n\nDefault: 25 centipawns.");
        y += dy;
    }

    // Max Plies
    {
        HWND lbl = mkLabel(pane, L"Max Plies", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MAXPLIES, L"250", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_MAXPLIES] = ed;
        AddTooltip(lbl, L"WHAT: The maximum number of half-moves (plies) allowed in a single game before it is automatically adjudicated as a draw. One ply equals one side's move, so 250 plies is roughly 125 full moves.\n\nWHY: Without a move limit, some games -- particularly in closed positions or repetitive endgames -- can spiral well past 300 moves, eventually triggering the 50-move draw rule anyway. These ultra-long games waste computation time generating repetitive, low-value training positions. A hard cap ensures no single game consumes a disproportionate amount of resources.\n\nWHEN TO ADJUST: Lower (e.g. 150-200) to speed up data generation at the cost of cutting off some endgames early. Raise (e.g. 300-400) if you want the engine to learn longer endgame technique and are willing to spend more time per game. Very low values (below 100) will prevent many games from reaching natural conclusions and degrade training quality.\n\nDefault: 250 plies (~125 full moves).");
        y += dy;
    }

    // Draw Adjudication (cp)
    {
        HWND lbl = mkLabel(pane, L"Draw Adj Cp", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAWCP, L"8", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DRAWCP] = ed;
        AddTooltip(lbl, L"WHAT: The centipawn threshold for automatic draw adjudication. If both sides' evaluations remain within this value of 0.00 for several consecutive moves after move 50, the game is declared a draw without playing further.\n\nWHY: Many self-play games reach dead-drawn positions (e.g. opposite-colored bishops, blocked pawns) long before a formal draw by repetition or 50-move rule occurs. Without adjudication, the engine plays dozens of meaningless shuffling moves that waste time and add noise to the training data. Draw adjudication detects these positions early and ends the game cleanly.\n\nWHEN TO ADJUST: Lower (e.g. 2-5) for stricter draw detection -- only truly dead positions are adjudicated, but more games drag on. Raise (e.g. 15-30) to be more aggressive about ending close games, speeding up generation but risking premature draws in positions where one side had a slight edge. Set to 0 to disable draw adjudication entirely -- games will only end by checkmate, resignation, repetition, stalemate, or the Max Plies limit.\n\nDefault: 8 centipawns (roughly equal to a small positional edge).");
        y += dy;
    }

    // Resign Count
    {
        HWND lbl = mkLabel(pane, L"Resign Count", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_RESIGN_COUNT, L"3", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_RESIGN_COUNT] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive moves where the engine\u2019s evaluation must exceed the Resign Cp threshold before the game is resigned. Prevents premature resignation from temporary evaluation spikes.\n\nWHY: A single move with a high eval can be a search artifact \u2014 the next move might refute the threat. Requiring multiple consecutive bad evaluations ensures the position is truly lost before resigning. Lower values speed up games but risk occasional premature resignations.\n\nWHEN TO ADJUST: 3 is the standard. Increase to 4-5 for more cautious resignation (fewer false quits). Decrease to 2 for faster game completion in clearly lost positions. Setting to 1 resigns immediately on any single evaluation above threshold \u2014 not recommended.\n\nDefault: 3 consecutive plies.");
        y += dy;
    }

    // Draw Count
    {
        HWND lbl = mkLabel(pane, L"Draw Count", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_COUNT, L"6", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DRAW_COUNT] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive moves where both sides\u2019 evaluations must stay within the Draw Adj Cp threshold before the game is adjudicated as a draw.\n\nWHY: Brief periods of equal evaluation can occur in positions with hidden tactics. Requiring 6 consecutive balanced evaluations ensures the position is genuinely drawn, not just temporarily quiet. Lower values catch draws faster but may prematurely end positions where one side has a hidden advantage.\n\nWHEN TO ADJUST: 6 is well-calibrated. Increase to 8-10 for stricter draw detection (fewer premature draws). Decrease to 4 for faster adjudication. Works in conjunction with Draw Min Ply \u2014 both conditions must be met.\n\nDefault: 6 consecutive plies.");
        y += dy;
    }

    // Draw Min Ply
    {
        HWND lbl = mkLabel(pane, L"Draw Min Ply", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_MIN_PLY, L"40", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DRAW_MIN_PLY] = ed;
        AddTooltip(lbl, L"WHAT: The earliest half-move at which the primary draw adjudication (Draw Adj Cp + Draw Count) can trigger. Before this ply, games continue regardless of evaluation.\n\nWHY: Early in the game, positions are still developing and many lines look temporarily equal. Allowing early draw adjudication would cut off games before meaningful play develops, producing training data dominated by opening positions with no middlegame content.\n\nWHEN TO ADJUST: 40 plies (20 full moves) ensures games reach the middlegame. Increase to 60-80 for longer games with more endgame data. Decrease to 20-30 for faster generation at the cost of shorter average game length.\n\nDefault: 40 plies.");
        y += dy;
    }

    // Draw Adj Moves
    {
        HWND lbl = mkLabel(pane, L"Draw Adj Moves", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_ADJ_MOVES, L"12", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DRAW_ADJ_MOVES] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive plies of near-zero evaluation (below Draw Adj Thresh) required for the secondary \u201Cdead position\u201D draw adjudication. This is a separate, more aggressive draw detector for completely lifeless positions.\n\nWHY: Some positions are obviously drawn (blocked pawns, insufficient material in practice) but may not trigger the primary draw check because the evaluations are slightly asymmetric. The secondary adjudicator catches these dead positions by looking for prolonged near-zero evaluation windows.\n\nWHEN TO ADJUST: 12 plies (6 full moves) of dead-equal play is a strong signal. Increase to 16-20 for more conservative detection. Decrease to 8-10 for faster detection of dead positions. This only triggers after Draw Adj Min Move.\n\nDefault: 12 plies.");
        y += dy;
    }

    // Draw Adj Threshold
    {
        HWND lbl = mkLabel(pane, L"Draw Adj Thresh", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_ADJ_THRESH, L"4", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DRAW_ADJ_THRESH] = ed;
        AddTooltip(lbl, L"WHAT: The centipawn threshold for the secondary \u201Cdead position\u201D draw adjudication. Evaluations with |eval| <= this value are considered dead equal.\n\nWHY: This is intentionally tighter than Draw Adj Cp. A 4cp threshold means only truly dead-equal positions (both sides agree the position is completely drawn) trigger the secondary adjudication. This catches fortress-type positions and opposite-colored bishop endgames that might not trigger the primary draw check.\n\nWHEN TO ADJUST: 4cp is very tight \u2014 nearly perfect equality. Increase to 8-12 for a wider dead-position band. Decrease to 2 for only mathematically dead positions. Works in conjunction with Draw Adj Moves and Draw Adj Min Move.\n\nDefault: 4 centipawns.");
        y += dy;
    }

    // Draw Adj Min Move
    {
        HWND lbl = mkLabel(pane, L"Draw Adj Min Move", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAW_ADJ_MIN_MOVE, L"50", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DRAW_ADJ_MIN_MOVE] = ed;
        AddTooltip(lbl, L"WHAT: The minimum move number (full moves, not plies) before the secondary dead-position draw adjudication can trigger. This ensures games play a substantial amount before being cut short.\n\nWHY: Even completely equal-looking positions in the early middlegame may have latent imbalances that surface later. This minimum ensures the secondary draw check only fires in the late middlegame or endgame, where dead positions are genuinely drawn.\n\nWHEN TO ADJUST: 50 (move 50) is conservative. Decrease to 30-40 for faster adjudication. Increase to 60-80 if you want very long games before any secondary draw detection. This is independent of Draw Min Ply (which controls the primary draw check).\n\nDefault: move 50.");
        y += dy;
    }

    // ── Section header: Training ──────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Training ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_ui.fUI,TRUE);
        y += 20;
    }

    // Epochs per Gen
    {
        HWND lbl = mkLabel(pane, L"Epochs per Gen", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_EPOCHS, L"10", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_EPOCHS] = ed;
        AddTooltip(lbl, L"WHAT: The number of complete passes through the training dataset each generation. One epoch means every position in the dataset is seen exactly once by the optimizer.\n\nWHY: Multiple epochs allow the model to refine its weights by seeing each position several times with different mini-batch compositions. Too few epochs and the model underfits -- it hasn't fully absorbed the data. Too many and it overfits -- it memorizes specific positions rather than learning general patterns, causing validation loss to rise.\n\nWHEN TO ADJUST: 10-15 epochs is a good starting range. If validation loss is still dropping when training ends, increase epochs. If validation loss starts rising well before the last epoch, reduce epochs or rely on Early Stop to catch it. Larger datasets need fewer epochs (the model sees enough variety in one pass), while smaller datasets benefit from more epochs.\n\nDefault: 10 epochs per generation.");
        y += dy;
    }

    // Batch Size
    {
        HWND lbl = mkLabel(pane, L"Batch Size", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_BATCHSZ, L"2048", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_BATCHSZ] = ed;
        AddTooltip(lbl, L"WHAT: The number of training positions processed together in a single forward/backward pass before updating the model weights. The gradient is averaged over all positions in the batch.\n\nWHY: Batch size controls the noise/stability trade-off in gradient estimation. Small batches (256-512) produce noisy gradients that can help escape local minima but cause erratic training. Large batches (4096-8192) produce smooth, stable gradients but may converge to sharper (less generalizable) minima and require more VRAM.\n\nWHEN TO ADJUST: 2048 works well for most NNUE training. Increase if you have VRAM to spare and want smoother loss curves. Decrease if you run out of VRAM (CUDA out of memory errors). You can also use Grad Accumulation to simulate a larger effective batch size without increasing VRAM usage. The effective batch size is Batch Size × Grad Accumulation.\n\nDefault: 2048 positions.");
        y += dy;
    }

    // Learning Rate
    {
        HWND lbl = mkLabel(pane, L"Learning Rate", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_LR, L"0.001", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_LR] = ed;
        AddTooltip(lbl, L"WHAT: The step size for each gradient update -- how much the model weights change in response to each batch of training data. This is the single most important hyperparameter in neural network training.\n\nWHY: The learning rate controls the speed and stability of convergence. Too high and the model overshoots good solutions, causing loss to spike or diverge. Too low and training crawls, potentially getting stuck in poor local minima. The right value depends on batch size, model architecture, and data quality.\n\nWHEN TO ADJUST: 0.001 is a strong default for Adam-family optimizers with batch size 2048. If loss oscillates wildly or spikes, halve the LR. If training is very slow or plateaus early, try doubling it. When increasing batch size, consider scaling LR proportionally (e.g. double batch \u2192 double LR). Enable Cosine LR to automatically decay from this starting value.\n\nDefault: 0.001.");
        y += dy;
    }

    // Weight Decay
    {
        HWND lbl = mkLabel(pane, L"Weight Decay", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WD, L"1e-5", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_WD] = ed;
        AddTooltip(lbl, L"WHAT: L2 regularization strength applied to all model weights during training. Each update, weights are multiplied by (1 - weight_decay), gently shrinking them toward zero.\n\nWHY: Without regularization, weights can grow arbitrarily large as the model memorizes training data. Weight decay acts as a soft constraint, keeping weights small and encouraging the model to find simpler, more generalizable solutions. This reduces overfitting -- the gap between training loss and validation loss.\n\nWHEN TO ADJUST: 0.00001 (1e-5) is a conservative default. Increase to 0.0001 or 0.001 if you see significant overfitting (training loss much lower than validation loss). Decrease or set to 0 if the model underfits (both losses plateau high). Larger models and smaller datasets benefit more from stronger decay.\n\nDefault: 0.00001 (1e-5).");
        y += dy;
    }

    // Dropout
    {
        HWND lbl = mkLabel(pane, L"Dropout", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DROPOUT, L"0.1", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DROPOUT] = ed;
        AddTooltip(lbl, L"WHAT: The probability that each neuron is randomly disabled (output set to zero) during each training forward pass. At inference time, all neurons are active but outputs are scaled accordingly.\n\nWHY: Dropout is a powerful regularization technique that prevents neurons from co-adapting -- relying too heavily on specific other neurons. By randomly disabling neurons, the network is forced to learn redundant representations, making it more robust and less prone to overfitting. It effectively trains an ensemble of sub-networks.\n\nWHEN TO ADJUST: 0.1 (10%%) is a good starting point for NNUE architectures which are relatively small. Increase to 0.2-0.3 if overfitting persists despite weight decay. Set to 0 to disable -- useful when you have abundant training data and overfitting is not a concern, or for final fine-tuning runs. Values above 0.5 are almost never beneficial and will cause underfitting.\n\nDefault: 0.1 (10%% dropout rate).");
        y += dy;
    }

    // Label Smoothing
    {
        HWND lbl = mkLabel(pane, L"Label Smoothing", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_LSMOOTH, L"0.05", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_LSMOOTH] = ed;
        AddTooltip(lbl, L"WHAT: Softens hard win/loss targets by blending them toward 0.5. With a value of 0.05, a target of 1.0 (win) becomes 0.95, and 0.0 (loss) becomes 0.05. Draw targets (0.5) are unaffected.\n\nWHY: Hard 0/1 targets encourage the model to produce extreme, overconfident predictions. In chess, even clearly winning positions have some probability of a draw or loss due to the opponent's counterplay. Label smoothing teaches the model that no outcome is 100%% certain, producing better-calibrated evaluations that correlate more closely with actual win probabilities.\n\nWHEN TO ADJUST: 0.05 is a safe default. Increase to 0.1-0.15 if the model's evaluations are too extreme (e.g. jumping to ±900cp in slightly favorable positions). Set to 0 if you want the sharpest possible evaluation distinctions, though this may reduce playing strength against varied opponents.\n\nDefault: 0.05.");
        y += dy;
    }

    // Grad Accumulation
    {
        HWND lbl = mkLabel(pane, L"Grad Accumulation", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_GRADACCUM, L"4", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_GRADACCUM] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive mini-batches whose gradients are accumulated (summed) before performing a single weight update. The effective batch size becomes Batch Size × Grad Accumulation.\n\nWHY: Some training benefits from large batch sizes (stable gradients, better convergence) but large batches require proportionally more VRAM. Gradient accumulation achieves the same mathematical result as a larger batch by spreading the computation across multiple smaller forward/backward passes, at the cost of slightly slower training due to more sequential steps.\n\nWHEN TO ADJUST: Set to 1 if your GPU can handle the desired batch size directly (most efficient). Increase to 2-8 if you want a larger effective batch but are VRAM-limited. For example, Batch Size 2048 × Grad Accumulation 4 = effective batch of 8192. Values above 8 are rarely needed. Higher values slow down training proportionally since more passes happen per weight update.\n\nDefault: 4 (effective batch size = 2048 × 4 = 8192).");
        y += dy;
    }

    // LR Warmup Steps
    {
        HWND lbl = mkLabel(pane, L"LR Warmup Steps", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WARMUP, L"500", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_WARMUP] = ed;
        AddTooltip(lbl, L"WHAT: The number of training steps over which the learning rate linearly increases from 0 to the target Learning Rate at the start of each generation's training. One step equals one batch (or one accumulated gradient update if Grad Accumulation > 1).\n\nWHY: When training begins, model weights are in a random or partially-trained state. Hitting them with the full learning rate immediately can cause destructive, oversized updates that push the model into a bad region of the loss landscape. Warmup lets the optimizer calibrate its internal momentum estimates (Adam's running averages) with small, safe steps before ramping to full speed.\n\nWHEN TO ADJUST: 500 steps is a solid default. Increase to 1000-2000 if training is unstable in the first epoch (loss spikes then recovers). Decrease to 100-200 if your dataset is small and 500 steps covers too large a fraction of the epoch. Set to 0 to disable warmup -- only recommended if you're fine-tuning an already well-trained model.\n\nDefault: 500 steps.");
        y += dy;
    }

    // Max Positions
    {
        HWND lbl = mkLabel(pane, L"Max Positions", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MAXPOS, L"300000", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_MAXPOS] = ed;
        AddTooltip(lbl, L"WHAT: The maximum number of positions loaded from each training data file per generation. When set, only the first N positions from each dataset (self-play, draws, replay) are used. 0 means no limit -- use all available positions.\n\nWHY: Self-play can generate millions of positions per generation, but training on all of them may not be necessary or practical. Capping the count speeds up each training epoch, reduces VRAM usage, and can actually improve model quality by preventing the optimizer from over-fitting to a single generation's data distribution.\n\nWHEN TO ADJUST: 300000 is a good balance for most runs. Increase to 500000-1000000 if you have fast hardware and want maximum data utilization. Decrease to 100000-200000 for faster iteration during early development. If training takes too long per generation, this is the first knob to turn. Set to 0 only if your games-per-gen is already low.\n\nDefault: 300000 positions.");
        y += dy;
    }

    // Early Stop
    {
        HWND lbl = mkLabel(pane, L"Early Stop", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_EARLYSTOP, L"10", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_EARLYSTOP] = ed;
        AddTooltip(lbl, L"WHAT: The number of consecutive epochs with no improvement in validation loss before training is automatically stopped for the current generation. The best model checkpoint (lowest validation loss) is kept.\n\nWHY: After a certain point, additional epochs only improve training loss while validation loss stagnates or worsens -- classic overfitting. Early stopping detects this plateau and saves time by ending training before all epochs are exhausted. The best weights are preserved, so no progress is lost.\n\nWHEN TO ADJUST: 10 epochs of patience is a safe default -- generous enough to ride out temporary plateaus. Decrease to 3-5 for faster iteration if you're confident the model converges quickly. Increase to 15-20 if your learning rate schedule has warm restarts (Cosine T0) that cause temporary loss increases before improving. Set very high (999) to effectively disable early stopping and always train for all epochs.\n\nDefault: 10 epochs patience.");
        y += dy;
    }

    // ── Section header: LR Schedule ──────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- LR Schedule ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_ui.fUI,TRUE);
        y += 20;
    }

    // Cosine LR checkbox
    {
        g_ui.hChkCosineLR = mkCheck(pane, ID_CHK_COSINELR, L"Cosine LR Schedule", lx, y, PW-16, 20, true);
        AddTooltip(g_ui.hChkCosineLR, L"WHAT: Enables cosine annealing, which smoothly decays the learning rate from the initial value down to near-zero following a cosine curve shape over the course of training.\n\nWHY: A constant learning rate is suboptimal -- early training benefits from large steps to explore the loss landscape, while later training needs small steps to fine-tune. Cosine annealing provides this naturally: the LR starts high, slowly decreases through the middle, and gently approaches zero at the end. This consistently outperforms constant LR in neural network training.\n\nWHEN TO ADJUST: Keep enabled for virtually all training runs. Disable only for debugging or if you're using a custom external LR schedule. When enabled, the initial Learning Rate value becomes the peak of the cosine curve.\n\nDefault: Enabled.");
        y += dy;
    }

    // Cosine T0
    {
        HWND lbl = mkLabel(pane, L"Cosine T0", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_COSINET0, L"50", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_COSINET0] = ed;
        AddTooltip(lbl, L"WHAT: The period (in epochs) for cosine warm restarts. Every T0 epochs, the learning rate jumps back to its initial value and begins decaying again. Set to 0 for a single smooth decay over all epochs with no restarts.\n\nWHY: Warm restarts periodically shake the model out of local minima by briefly increasing the learning rate. This can help the model discover better solutions it would miss with monotonic decay. The technique (SGDR) has been shown to improve generalization, especially when combined with SWA which averages checkpoints across restart cycles.\n\nWHEN TO ADJUST: Set to 0 for simple cosine decay without restarts (safest default). Try T0 = Epochs/2 or Epochs/3 for 2-3 restart cycles per generation. Shorter periods (5-10) create frequent restarts good for exploration but may prevent convergence. Longer periods (50+) behave more like plain cosine decay. Only effective when Cosine LR is enabled.\n\nDefault: 50 epochs (one restart if training runs 100 epochs).");
        y += dy;
    }

    // SWA checkbox
    {
        g_ui.hChkSWA = mkCheck(pane, ID_CHK_SWA, L"SWA (Stochastic Wt Avg)", lx, y, PW-16, 20, true);
        AddTooltip(g_ui.hChkSWA, L"WHAT: Enables Stochastic Weight Averaging, which maintains a running average of model weights collected at regular intervals during training. The averaged model is saved alongside the standard best-validation model.\n\nWHY: Individual model checkpoints can be noisy -- they happen to perform well on the validation set but may be in a sharp, fragile minimum. SWA averages multiple checkpoints, producing a model that sits in a wider, flatter minimum of the loss landscape. Flatter minima generalize better to unseen positions, often producing 10-30 Elo improvement over the single best checkpoint.\n\nWHEN TO ADJUST: Keep enabled for most training runs -- it's essentially free extra strength. Disable only if you want to simplify debugging or if SWA models are consistently weaker than best-val models (rare, but possible with very small datasets). SWA pairs especially well with cosine warm restarts.\n\nDefault: Enabled.");
        y += dy;
    }

    // SWA Start
    {
        HWND lbl = mkLabel(pane, L"SWA Start Epoch", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SWASTART, L"3", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_SWASTART] = ed;
        AddTooltip(lbl, L"WHAT: The epoch number after which SWA begins collecting model snapshots for averaging. Before this epoch, only standard training occurs.\n\nWHY: Early epochs produce rapidly-changing, unstable weights as the model makes large adjustments. Including these early snapshots in the SWA average would drag it toward poor solutions. By waiting until the model has partially converged, SWA only averages high-quality checkpoints from the refinement phase of training.\n\nWHEN TO ADJUST: 3 is a good default for typical 10-15 epoch runs. For longer runs (50+ epochs), set to 25-50%% of total epochs. If you're using warm restarts (Cosine T0), set SWA Start to at least T0 so the model completes one full cosine cycle before averaging begins. Setting too high leaves too few snapshots to average, reducing SWA's benefit.\n\nDefault: Epoch 3.");
        y += dy;
    }

    // ── Section header: Scoring ───────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Scoring ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_ui.fUI,TRUE);
        y += 20;
    }

    // Draw Weight
    {
        HWND lbl = mkLabel(pane, L"Draw Weight", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAWWT, L"0.5", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DRAWWT] = ed;
        AddTooltip(lbl, L"WHAT: A multiplier applied to the training loss for positions from drawn games. A value of 0.5 means drawn positions contribute half as much to the gradient as decisive (win/loss) positions.\n\nWHY: Draws are the most common outcome in strong chess. Without down-weighting, drawn positions dominate the training data and the model learns to predict 0.50 (draw) too eagerly. De-emphasizing draws forces the model to pay more attention to the distinguishing features of winning and losing positions, producing sharper, more decisive evaluations.\n\nWHEN TO ADJUST: 0.5 is the recommended value. Decrease to 0.2-0.3 if your draw rate is very high (60%%+) and you want even more emphasis on decisive games. Increase to 0.7-1.0 if draws are rare or if the engine misjudges drawn positions (e.g. calling drawn endgames as winning). Setting to 0 completely ignores draws in training -- extreme but sometimes useful for tactical-style training.\n\nDefault: 0.5 (draws at half weight).");
        y += dy;
    }

    // Mate Boost
    {
        HWND lbl = mkLabel(pane, L"Mate Boost", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_MATEBOOST, L"3.0", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_MATEBOOST] = ed;
        AddTooltip(lbl, L"WHAT: A loss multiplier applied to positions that are near checkmate (mate-in-N). Positions closer to mate receive a higher weight in the training loss, proportional to this value.\n\nWHY: Mating positions are rare in training data but critically important for playing strength. Without boosting, the model treats a position with mate-in-3 the same as a quiet middlegame advantage. The boost ensures the model develops strong pattern recognition for mating nets, back-rank threats, and king-hunt sequences -- skills that directly win games.\n\nWHEN TO ADJUST: 3.0 is a strong default. Increase to 5.0-10.0 if the engine frequently misses forced mates or mishandles winning endgames. Decrease to 1.0-2.0 if the engine becomes too tactics-focused and neglects positional play. Set to 1.0 to disable the boost entirely (all positions weighted equally regardless of proximity to mate).\n\nDefault: 3.0× weight for mating positions.");
        y += dy;
    }

    // Self-Play Ratio
    {
        HWND lbl = mkLabel(pane, L"Self-Play Ratio", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SPLRATIO, L"0.4", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_SPLRATIO] = ed;
        AddTooltip(lbl, L"WHAT: The fraction of the training dataset that comes from the current generation's self-play data versus replayed data from older generations. A value of 0.4 means 40%% current-gen self-play, with the remainder filled by replay data and other sources.\n\nWHY: Current-generation data reflects the model's latest strength level and is the most relevant for improvement. However, training only on current data causes catastrophic forgetting -- the model loses knowledge from earlier training. Mixing in older data (via Replay Window) provides stability and prevents the model from forgetting lessons learned in previous generations.\n\nWHEN TO ADJUST: 0.4 is balanced. Increase to 0.6-0.8 if you want faster adaptation to new patterns (good for early training). Decrease to 0.2-0.3 for more conservative training that preserves past knowledge (good for late-stage refinement). If Replay Window is 0 (no replay), this value has reduced effect since there's no older data to mix.\n\nDefault: 0.4 (40%% current generation data).");
        y += dy;
    }

    // Draw Ratio %
    {
        HWND lbl = mkLabel(pane, L"Draw Ratio %", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_DRAWPCT, L"10", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_DRAWPCT] = ed;
        AddTooltip(lbl, L"WHAT: The percentage of the Max Positions budget allocated to drawn game positions (from training_data_draws.bin). The remaining budget is split between decisive positions and self-play data.\n\nWHY: Draw data contains important positional knowledge (equal structures, fortresses, theoretical draws) but is extremely abundant -- often 50-60%% of all games are draws. Without capping, draws would flood the training set and dilute the decisive games that teach the model how to win and lose. This cap ensures a controlled proportion.\n\nWHEN TO ADJUST: 10%% is a good default. Increase to 15-25%% if the engine misjudges drawn endgames or fails to recognize fortress positions. Decrease to 5%% or less if you want maximum emphasis on decisive game positions and your draw rate is very high. Note: this interacts with Draw Weight -- both together control how much influence draws have on training.\n\nDefault: 10%%.");
        y += dy;
    }

    // FRC Mix %
    {
        HWND lbl = mkLabel(pane, L"FRC Mix %", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_FRCMIX, L"0", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_FRCMIX] = ed;
        AddTooltip(lbl, L"WHAT: The percentage of self-play games that begin from random Chess960 (Fischer Random) starting positions instead of standard chess openings. The remaining games use positions from the opening book.\n\nWHY: Standard chess openings produce recurring position types -- the same pawn structures and piece placements appear repeatedly. Chess960 forces the engine to evaluate unfamiliar piece configurations from move 1, teaching it general positional principles rather than memorized opening patterns. This produces a more robust, adaptable engine.\n\nWHEN TO ADJUST: 0%% for pure standard chess training. 10-20%% for a good diversity boost without straying too far from standard play. 30-50%% if you want a highly creative, unconventional engine. Above 50%% the engine becomes very strong in random positions but may underperform in standard openings where book knowledge matters. Ensure your engine supports FRC before enabling.\n\nDefault: 0%% (standard chess only).");
        y += dy;
    }

    // WDL Alpha
    {
        HWND lbl = mkLabel(pane, L"WDL Alpha", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WDLALPHA, L"0.5", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_WDLALPHA] = ed;
        AddTooltip(lbl, L"WHAT: The blending factor between two loss functions: Mean Squared Error on raw evaluation (alpha = 0.0) and Win/Draw/Loss cross-entropy (alpha = 1.0). At 0.5, both losses contribute equally to the training gradient.\n\nWHY: MSE loss teaches the model to predict accurate centipawn evaluations -- the raw numerical score. WDL loss teaches it to predict the probability of winning, drawing, or losing. Blending both produces an engine whose evaluation is both numerically accurate and well-calibrated to actual game outcomes. Pure MSE can produce evaluations that are precise but poorly correlated with win probability; pure WDL can produce good probability estimates but imprecise centipawn values.\n\nWHEN TO ADJUST: 0.5 (equal blend) is a strong default. Shift toward 0.3 if you want more precise centipawn evaluations (good for analysis). Shift toward 0.7 if you want better win-probability calibration (good for playing strength). Extreme values (0.0 or 1.0) use only one loss function -- useful for experimentation but generally weaker than a blend.\n\nDefault: 0.5 (equal MSE and WDL blend).");
        y += dy;
    }

    // WDL Draw Elo
    {
        HWND lbl = mkLabel(pane, L"WDL Draw Elo", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_WDLDRAWELO, L"100", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_WDLDRAWELO] = ed;
        AddTooltip(lbl, L"WHAT: Controls the width of the draw band when converting centipawn evaluations to Win/Draw/Loss targets. A higher value means a wider range of evaluations are mapped to high draw probability. Measured in Elo-scaled units.\n\nWHY: In real chess, positions evaluated at +0.30 (30 centipawns) are almost always draws between strong players, while +2.00 is usually a win. The Draw Elo parameter shapes this mapping curve -- it determines how much of an evaluation advantage is needed before the WDL target shifts from 'likely draw' to 'likely win'. Getting this right ensures training targets match realistic game outcomes.\n\nWHEN TO ADJUST: 100 is calibrated for typical engine play. Lower to 50-70 for a narrower draw band that pushes the model to be more decisive -- small advantages get labeled as potential wins, encouraging aggressive play. Raise to 130-180 for a wider draw band -- the model learns that even moderate advantages often result in draws, producing more conservative/positional play.\n\nDefault: 100.");
        y += dy;
    }

    // Replay Window
    {
        HWND lbl = mkLabel(pane, L"Replay Window", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_REPLAYWIN, L"3", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_REPLAYWIN] = ed;
        AddTooltip(lbl, L"WHAT: The number of past generations whose self-play data is mixed into the current generation's training set. A value of 3 means positions from the current generation plus the 3 most recent previous generations are used.\n\nWHY: Each generation's self-play data represents a snapshot of the model's strength at that point. Replaying older data provides training stability and prevents catastrophic forgetting -- where the model loses skills it learned earlier. It also increases the effective dataset size without additional self-play computation. Older positions are weighted less via Replay Decay so recent, higher-quality data dominates.\n\nWHEN TO ADJUST: 3-5 is the sweet spot. Increase to 7-10 if training is unstable between generations (large loss spikes). Set to 0 to use only current-gen data -- fast but prone to forgetting and overfitting. Very large windows (15+) dilute the training set with low-quality early-generation data, slowing improvement.\n\nDefault: 3 generations.");
        y += dy;
    }

    // Replay Decay
    {
        HWND lbl = mkLabel(pane, L"Replay Decay", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_REPLAYDECAY, L"0.7", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_REPLAYDECAY] = ed;
        AddTooltip(lbl, L"WHAT: The exponential decay factor applied to replayed positions based on their generation age. Each generation back, the sampling weight is multiplied by this value. With decay 0.7: gen-1 gets 70%%, gen-2 gets 49%% (0.7²), gen-3 gets 34%% (0.7³).\n\nWHY: Older self-play data was generated by weaker versions of the model and contains more mistakes and suboptimal evaluations. While this data still provides useful diversity, it should be down-weighted relative to recent, higher-quality data. The exponential decay naturally phases out stale data while keeping it available for stability.\n\nWHEN TO ADJUST: 0.7 is a strong default that significantly reduces old data influence while retaining it. Increase to 0.8-0.9 for more equal weighting across generations (useful if improvement between generations is small). Decrease to 0.4-0.6 to heavily favor recent data (useful when the model is improving rapidly and old data is outdated). Set to 1.0 for uniform weighting (all replayed generations treated equally).\n\nDefault: 0.7 (each generation back gets 70%% of the previous generation's weight).");
        y += dy;
    }

    // ── Section header: Validation ────────────────────────────────
    {
        HWND sep = CreateWindowExW(0,L"STATIC",L"--- Validation ---",
                                   WS_CHILD|WS_VISIBLE|SS_LEFT,
                                   lx,y,PW-16,16,pane,nullptr,g_hInst,nullptr);
        SendMessageW(sep,WM_SETFONT,(WPARAM)g_ui.fUI,TRUE);
        y += 20;
    }

    // ELO Validation checkbox
    {
        g_ui.hChkElo = mkCheck(pane, ID_CHK_ELO, L"ELO Validation", lx, y, PW-16, 20, false);
        AddTooltip(g_ui.hChkElo, L"WHAT: When enabled, an automated match is played between the newly trained model and the previous generation's model after each training cycle. The result is converted to an estimated Elo difference.\n\nWHY: Loss and accuracy metrics don't always correlate with playing strength -- a model can have lower loss but play weaker chess due to evaluation blind spots. Elo validation provides ground-truth measurement of whether the new model actually plays better. It catches regressions that loss metrics would miss.\n\nWHEN TO ADJUST: Enable for production training runs where you need confidence that each generation is improving. Disable for fast iteration or early experiments -- Elo matches add significant time to each generation (proportional to ELO Games setting). Also useful to disable temporarily if you're tuning hyperparameters and using validation loss as your primary metric.\n\nDefault: Disabled (loss-based evaluation only).");
        y += dy;
    }

    // ELO Games
    {
        HWND lbl = mkLabel(pane, L"ELO Games", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_ELOGAMES, L"100", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_ELOGAMES] = ed;
        AddTooltip(lbl, L"WHAT: The number of games played in the Elo validation match between the new model and the previous generation's model. Games are split evenly between playing white and black.\n\nWHY: Elo estimation from match results has statistical uncertainty that decreases with more games. A 10-game match can easily give misleading results (±100 Elo error). A 100-game match narrows uncertainty to roughly ±30 Elo. More games = more confidence that measured improvement is real and not noise.\n\nWHEN TO ADJUST: 100 is a reasonable default balancing accuracy with time cost. Increase to 200-500 for high-confidence measurements (important for late-stage training where improvements are small). Decrease to 50 for rough estimates during development. Below 30 games, Elo estimates are essentially meaningless noise. Only applies when ELO Validation is enabled.\n\nDefault: 100 games.");
        y += dy;
    }

    // SWA Games
    {
        HWND lbl = mkLabel(pane, L"SWA Games", lx, y+2, lw, 18);
        HWND ed  = mkEdit(pane, ID_EDIT_SWAGAMES, L"50", ex, y, ew, 20);
        g_ui.edits[ID_EDIT_SWAGAMES] = ed;
        AddTooltip(lbl, L"WHAT: The number of games played in a match between the SWA-averaged model and the best-validation-loss model to determine which is stronger. Games are split between white and black.\n\nWHY: SWA produces a separate model from a different optimization strategy (weight averaging vs best checkpoint). Sometimes SWA is stronger, sometimes best-val wins. This match determines which model to carry forward as the generation's final weights, ensuring you always keep the stronger one.\n\nWHEN TO ADJUST: 50 games provides a reasonable signal. Increase to 100-200 if SWA and best-val models are close in strength and you want a more reliable comparison. Decrease to 30 for speed during development. Only applies when both SWA and ELO Validation are enabled.\n\nDefault: 50 games.");
        y += dy;
    }

    // Overfitting Detection checkbox
    {
        g_ui.hChkOvfit = mkCheck(pane, ID_CHK_OVERFIT, L"Overfitting Detection", lx, y, PW-16, 20, true);
        AddTooltip(g_ui.hChkOvfit, L"WHAT: Monitors the gap between training loss and validation loss during each epoch. If the gap exceeds a threshold (training loss dropping while validation loss rises or stagnates), training is flagged as overfitting and may be stopped early.\n\nWHY: Overfitting means the model is memorizing training positions rather than learning generalizable patterns. An overfitting model performs worse on new positions despite appearing to improve on training data. Early detection saves time and prevents adopting a degraded model. This is complementary to Early Stop -- Early Stop watches validation loss alone, while Overfitting Detection watches the divergence between train and val.\n\nWHEN TO ADJUST: Keep enabled for most training runs -- it's a safety net with no downside. Disable only if you're intentionally training for many epochs on a small dataset and expect some train/val divergence as normal (e.g. fine-tuning on a curated position set).\n\nDefault: Enabled.");
        y += dy;
    }

    // Record total content height for scrolling
    g_cfgTotalH = y + 8;
    g_cfgScrollY = 0;
}


// ── Config panel subclass (dark background + scrolling + label colors) ────────
void UpdateCfgScroll(HWND hw) {
    RECT rc; GetClientRect(hw, &rc);
    int viewH = rc.bottom;
    int maxScroll = (std::max)(0, g_cfgTotalH - viewH);
    if (g_cfgScrollY > maxScroll) g_cfgScrollY = maxScroll;
    if (g_cfgScrollY < 0) g_cfgScrollY = 0;
    SCROLLINFO si = {}; si.cbSize = sizeof(si);
    si.fMask = SIF_RANGE | SIF_PAGE | SIF_POS;
    si.nMin = 0; si.nMax = g_cfgTotalH;
    si.nPage = viewH; si.nPos = g_cfgScrollY;
    SetScrollInfo(hw, SB_VERT, &si, TRUE);
    // Show/hide scrollbar based on whether content overflows
    ShowScrollBar(hw, SB_VERT, g_cfgTotalH > viewH);
}

void ScrollCfgTo(HWND hw, int newPos) {
    RECT rc; GetClientRect(hw, &rc);
    int viewH = rc.bottom;
    int maxScroll = (std::max)(0, g_cfgTotalH - viewH);
    newPos = (std::max)(0, (std::min)(newPos, maxScroll));
    int delta = g_cfgScrollY - newPos;
    if (delta == 0) return;
    g_cfgScrollY = newPos;
    ScrollWindowEx(hw, 0, delta, nullptr, nullptr, nullptr, nullptr,
                   SW_SCROLLCHILDREN | SW_INVALIDATE | SW_ERASE);
    UpdateCfgScroll(hw);
}

LRESULT CALLBACK PanelProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp,
                                   UINT_PTR, DWORD_PTR) {
    if (msg == WM_CTLCOLORSTATIC) {
        HDC hdc = (HDC)wp;
        SetBkColor(hdc, C_PANEL);
        SetTextColor(hdc, C_TEXT);
        return (LRESULT)g_ui.brPanel;
    }
    if (msg == WM_CTLCOLOREDIT) {
        HDC hdc = (HDC)wp;
        SetBkColor(hdc, RGB(30,30,46) /* INFO [6.29]: Panel uses (30,30,46) vs WndProc (20,20,32) — intentional visual distinction */);
        SetTextColor(hdc, C_TEXT);
        // INFO [6.14]: Static brush — created once, never freed. Acceptable for process-lifetime singletons.
        static HBRUSH editBr = CreateSolidBrush(RGB(30,30,46));
        return (LRESULT)editBr;
    }
    if (msg == WM_ERASEBKGND) {
        RECT rc; GetClientRect(hw, &rc);
        FillRect((HDC)wp, &rc, g_ui.brPanel);
        return 1;
    }
    if (msg == WM_SIZE) {
        UpdateCfgScroll(hw);
    }
    if (msg == WM_VSCROLL) {
        RECT rc; GetClientRect(hw, &rc);
        int viewH = rc.bottom;
        int newPos = g_cfgScrollY;
        switch (LOWORD(wp)) {
            case SB_LINEUP:        newPos -= 24; break;
            case SB_LINEDOWN:      newPos += 24; break;
            case SB_PAGEUP:        newPos -= viewH; break;
            case SB_PAGEDOWN:      newPos += viewH; break;
            case SB_THUMBTRACK:    // FIX M-10: Use SCROLLINFO for full 32-bit position
            case SB_THUMBPOSITION: {
                SCROLLINFO si{}; si.cbSize = sizeof(si); si.fMask = SIF_TRACKPOS;
                GetScrollInfo(hw, SB_VERT, &si);
                newPos = si.nTrackPos;
                break;
            }
        }
        ScrollCfgTo(hw, newPos);
        return 0;
    }
    if (msg == WM_MOUSEWHEEL) {
        int zDelta = GET_WHEEL_DELTA_WPARAM(wp);
        ScrollCfgTo(hw, g_cfgScrollY - zDelta / 2);
        return 0;
    }
    // Forward WM_COMMAND from child controls (buttons, combos, checkboxes)
    // to the main window so they can be handled in WndProc
    if (msg == WM_COMMAND) {
        PostMessage(g_ui.hWnd, WM_COMMAND, wp, lp);
        return 0;
    }
    return DefSubclassProc(hw, msg, wp, lp);
}

// ── Update log listbox ────────────────────────────────────────────
// INFO [6.17]: FlushLog reads g_st.log under mutex then operates on listbox without lock.
// This is safe because FlushLog is only called from the main UI thread (timer + WM_USER+1).
void FlushLog() {
    // FIX 6.17: copy log data under lock to avoid TOCTOU; g_logSent is UI-thread-only so safe to read here
    std::deque<std::string> snap;
    { std::lock_guard<std::mutex> lk(g_st.mtx); snap = g_st.log; }
    size_t sz = snap.size();

    // Check if the last line was updated in-place (same count, different text)
    if (sz == g_logSent && sz > 0 && snap.back() != g_lastLogText) {
        // Replace the last listbox entry
        int cnt = (int)SendMessageW(g_ui.hLog, LB_GETCOUNT, 0, 0);
        if (cnt > 0) {
            SendMessageW(g_ui.hLog, LB_DELETESTRING, (WPARAM)(cnt - 1), 0);
            SendMessageW(g_ui.hLog, LB_ADDSTRING, 0, (LPARAM)W(snap.back()).c_str());
            cnt = (int)SendMessageW(g_ui.hLog, LB_GETCOUNT, 0, 0);
            if (cnt > 0) SendMessageW(g_ui.hLog, LB_SETTOPINDEX, (WPARAM)(cnt - 1), 0);
        }
        g_lastLogText = snap.back();
        return;
    }

    if (sz <= g_logSent) return;
    for (size_t i = g_logSent; i < sz; i++) {
        SendMessageW(g_ui.hLog, LB_ADDSTRING, 0, (LPARAM)W(snap[i]).c_str());
    }
    g_logSent = sz;
    g_lastLogText = snap.back();
    int cnt = (int)SendMessageW(g_ui.hLog, LB_GETCOUNT, 0, 0);
    if (cnt > 0) SendMessageW(g_ui.hLog, LB_SETTOPINDEX, (WPARAM)(cnt - 1), 0);
}

