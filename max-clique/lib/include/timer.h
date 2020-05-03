#ifndef TIMER_H
#define TIMER_H
#include <string>

namespace timer {
void start(const std::string& s);
int stop(const std::string& s);
void print(const std::string& s);
void print();
}  // namespace timer

#endif
