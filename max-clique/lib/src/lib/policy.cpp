#include "policy.h"

Policy::Policy() {}
Policy::~Policy() {}

std::shared_ptr<Policy> global_policy = nullptr;
