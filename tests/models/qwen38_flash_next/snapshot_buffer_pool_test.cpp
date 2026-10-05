// Snapshot buffer pool: released buffers must return to the pool and serve
// the next capture without a fresh mapping, indefinitely — the pooled-byte
// budget tracks retained buffers, so repeated reuse of one buffer must not
// exhaust the cap cumulatively (the accounting regression found in review:
// Acquire removed a buffer without releasing its allowance, so ~4 GiB of
// cumulative reuse silently disabled recycling).
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/models/qwen38_flash_next/snapshot_buffer_pool.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

void Run() {
  // A tight byte limit so six reuse cycles of one 1 GiB buffer exceed it
  // cumulatively: the regression disabled reuse after four cycles here.
  qfn::SnapshotBufferPool pool({.max_buffers = 8, .max_bytes = 4 * kGiB});

  std::uint8_t* address = nullptr;
  for (int cycle = 0; cycle < 6; ++cycle) {
    auto acquired = pool.Acquire(kGiB);
    Require(acquired.data != nullptr, "acquire returned no buffer");
    Require(acquired.capacity == kGiB, "acquire reported the wrong capacity");
    if (cycle == 0) {
      Require(!acquired.populated, "first acquire cannot be recycled");
    } else {
      Require(acquired.populated,
              "reuse stopped at cycle " + std::to_string(cycle) +
                  "; pooled budget exhausted cumulatively");
      Require(acquired.data.get() == address,
              "reuse served a different buffer");
    }
    address = acquired.data.get();
    pool.Release(std::move(acquired.data), acquired.capacity);
    Require(pool.pooled_buffers() == 1, "released buffer was not retained");
    Require(pool.pooled_bytes() == kGiB,
            "pooled bytes do not track retained buffers");
  }

  // Smallest-fit: the 1 GiB buffer serves the small ask; the 2 GiB pooled
  // buffer remains available for the larger one with its real capacity.
  auto large = pool.Acquire(2 * kGiB);
  Require(!large.populated, "no 2 GiB buffer is pooled yet");
  pool.Release(std::move(large.data), large.capacity);
  auto small = pool.Acquire(1 * kGiB);
  Require(small.populated, "1 GiB pooled buffer did not serve the small ask");
  Require(small.capacity == kGiB, "smallest-fit picked the wrong buffer");
  auto big = pool.Acquire(2 * kGiB);
  Require(big.populated, "2 GiB pooled buffer did not serve the larger ask");
  Require(big.capacity == 2 * kGiB, "smallest-fit lost the real capacity");

  // Byte-budget bound: two distinct buffers whose pooled bytes exceed the
  // limit — the second release is dropped instead of pooled (the caller's
  // unique_ptr frees the buffer).
  qfn::SnapshotBufferPool bounded({.max_buffers = 8, .max_bytes = 2 * kGiB});
  auto first = bounded.Acquire(kGiB);
  Require(!first.populated, "fresh bounded pool cannot recycle");
  bounded.Release(std::move(first.data), first.capacity);
  auto second = bounded.Acquire(2 * kGiB);
  Require(!second.populated, "1 GiB pooled buffer cannot serve a 2 GiB ask");
  bounded.Release(std::move(second.data), second.capacity);
  Require(bounded.pooled_buffers() == 1,
          "byte-bound pool retained more than its budget");
  Require(bounded.pooled_bytes() == kGiB,
          "byte-bound pool miscounted its allowance");
}

}  // namespace

int main() {
  try {
    Run();
  } catch (const std::exception& error) {
    std::cerr << "snapshot_buffer_pool_test: " << error.what() << "\n";
    return 1;
  }
  std::cout << "snapshot_buffer_pool_test: passed\n";
  return 0;
}
