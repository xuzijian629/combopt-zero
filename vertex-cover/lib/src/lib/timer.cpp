#include "timer.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <map>
#include <string>

namespace timer {
std::map<std::string, std::chrono::steady_clock::time_point> memo_cur;
std::map<std::string, int> memo_total;
void start(const std::string& s) {
    assert(memo_cur.count(s) == 0);
    memo_cur[s] = std::chrono::steady_clock::now();
}
int stop(const std::string& s) {
    assert(memo_cur.count(s));
    int elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - memo_cur[s]).count();
    memo_cur.erase(s);
    memo_total[s] += elapsed;
    return elapsed;
}
void print(const std::string& s) {
    assert(memo_cur.count(s) == 0);
    std::cout << s << ": " << memo_total[s] << "[ms]" << std::endl;
}
void print() {
    assert(memo_cur.empty());
    for (auto& p : memo_total) print(p.first);
}
}  // namespace timer
