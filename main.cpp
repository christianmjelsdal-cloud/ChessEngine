#include "VisualGame.h"
#include <memory>

int main() {
    // Heap-allocate VisualGame so the two Engine instances
    // (each ~500KB of fixed arrays) don't blow the main thread stack.
    auto game = std::make_unique<VisualGame>();
    game->run();
    return 0;
}
