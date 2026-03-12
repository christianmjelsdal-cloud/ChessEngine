#include "VisualGame.h"
#include "UCI.h"
#include <memory>
#include <string>
#include <cstring>
#include <iostream>

int main(int argc, char* argv[]) {
    // Launch in UCI mode if:
    //   1. Command-line argument "--uci" is passed, OR
    //   2. First line of stdin is "uci" (auto-detection for GUIs)
    bool uciMode = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--uci") == 0 || std::strcmp(argv[i], "uci") == 0) {
            uciMode = true;
            break;
        }
    }

    if (uciMode) {
        auto uci = std::make_unique<UCI>();
        uci->loop();
        return 0;
    }

    // Default: visual/GUI mode (original behavior)
    auto game = std::make_unique<VisualGame>();
    game->run();
    return 0;
}
