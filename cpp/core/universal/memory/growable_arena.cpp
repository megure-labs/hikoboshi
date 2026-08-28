#include "growable_arena.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace hikoboshi::core::universal::memory {
namespace {

bool is_power_of_two(const std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

std::size_t normalize_alignment(const std::size_t alignment) {
  if (!is_power_of_two(alignment)) {
    throw std::bad_alloc();
  }
  return std::max(alignment, alignof(void*));
}

std::size_t checked_multiply(const std::size_t lhs,
                             const std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::bad_array_new_length();
  }
  return lhs * rhs;
}

std::size_t aligned_offset(const std::size_t offset,
                           const std::size_t alignment) {
  const std::size_t remainder = offset % alignment;
  if (remainder == 0) {
    return offset;
  }
  const std::size_t delta = alignment - remainder;
  if (offset > std::numeric_limits<std::size_t>::max() - delta) {
    throw std::bad_alloc();
  }
  return offset + delta;
}

std::size_t grow_block_size(const std::size_t current,
                            const std::size_t minimum) {
  std::size_t next = std::max(current, minimum);
  if (next < minimum) {
    return minimum;
  }
  return next;
}

}  // namespace

GrowableArena::Block::Block(const std::size_t capacity_bytes,
                            const std::size_t block_alignment)
    : data(::operator new(capacity_bytes,
                          std::align_val_t{block_alignment})),
      capacity(capacity_bytes),
      offset(0),
      alignment(block_alignment) {}

GrowableArena::Block::~Block() {
  if (data != nullptr) {
    ::operator delete(data, std::align_val_t{alignment});
  }
}

GrowableArena::Block::Block(Block&& other) noexcept
    : data(other.data),
      capacity(other.capacity),
      offset(other.offset),
      alignment(other.alignment) {
  other.data = nullptr;
  other.capacity = 0;
  other.offset = 0;
  other.alignment = kDefaultAlignment;
}

GrowableArena::Block& GrowableArena::Block::operator=(
    Block&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (data != nullptr) {
    ::operator delete(data, std::align_val_t{alignment});
  }
  data = other.data;
  capacity = other.capacity;
  offset = other.offset;
  alignment = other.alignment;
  other.data = nullptr;
  other.capacity = 0;
  other.offset = 0;
  other.alignment = kDefaultAlignment;
  return *this;
}

GrowableArena::GrowableArena(const std::size_t initial_block_bytes)
    : initial_block_bytes_(std::max(initial_block_bytes, kDefaultBlockBytes)),
      next_block_bytes_(initial_block_bytes_),
      bytes_used_(0),
      peak_bytes_used_(0) {}

GrowableArena::~GrowableArena() = default;

GrowableArena::GrowableArena(GrowableArena&& other) noexcept
    : blocks_(std::move(other.blocks_)),
      initial_block_bytes_(other.initial_block_bytes_),
      next_block_bytes_(other.next_block_bytes_),
      bytes_used_(other.bytes_used_),
      peak_bytes_used_(other.peak_bytes_used_) {
  other.initial_block_bytes_ = kDefaultBlockBytes;
  other.next_block_bytes_ = kDefaultBlockBytes;
  other.bytes_used_ = 0;
  other.peak_bytes_used_ = 0;
}

GrowableArena& GrowableArena::operator=(GrowableArena&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  blocks_ = std::move(other.blocks_);
  initial_block_bytes_ = other.initial_block_bytes_;
  next_block_bytes_ = other.next_block_bytes_;
  bytes_used_ = other.bytes_used_;
  peak_bytes_used_ = other.peak_bytes_used_;
  other.initial_block_bytes_ = kDefaultBlockBytes;
  other.next_block_bytes_ = kDefaultBlockBytes;
  other.bytes_used_ = 0;
  other.peak_bytes_used_ = 0;
  return *this;
}

void GrowableArena::add_block(const std::size_t minimum_bytes,
                              const std::size_t alignment) {
  const std::size_t block_alignment =
      std::max(normalize_alignment(alignment), kDefaultAlignment);
  const std::size_t capacity =
      grow_block_size(next_block_bytes_, minimum_bytes + block_alignment);
  blocks_.emplace_back(capacity, block_alignment);

  if (next_block_bytes_ <=
      std::numeric_limits<std::size_t>::max() / 2) {
    next_block_bytes_ = std::max(next_block_bytes_ * 2, capacity);
  } else {
    next_block_bytes_ = std::numeric_limits<std::size_t>::max();
  }
}

void* GrowableArena::allocate_bytes(const std::size_t element_size,
                                    const std::size_t alignment,
                                    const std::size_t count) {
  if (count == 0 || element_size == 0) {
    return nullptr;
  }

  const std::size_t normalized_alignment = normalize_alignment(alignment);
  const std::size_t bytes = checked_multiply(element_size, count);

  for (;;) {
    if (blocks_.empty() ||
        blocks_.back().alignment < normalized_alignment) {
      add_block(bytes, normalized_alignment);
    }

    Block& block = blocks_.back();
    const std::size_t offset = aligned_offset(block.offset,
                                              normalized_alignment);
    if (offset <= block.capacity && bytes <= block.capacity - offset) {
      unsigned char* const base = static_cast<unsigned char*>(block.data);
      block.offset = offset + bytes;
      bytes_used_ += block.offset - offset;
      peak_bytes_used_ = std::max(peak_bytes_used_, bytes_used_);
      return base + offset;
    }

    add_block(bytes, normalized_alignment);
  }
}

void GrowableArena::reset() noexcept {
  for (Block& block : blocks_) {
    block.offset = 0;
  }
  bytes_used_ = 0;
}

void GrowableArena::clear() noexcept {
  blocks_.clear();
  next_block_bytes_ = initial_block_bytes_;
  bytes_used_ = 0;
}

std::size_t GrowableArena::bytes_used() const noexcept {
  return bytes_used_;
}

std::size_t GrowableArena::peak_bytes_used() const noexcept {
  return peak_bytes_used_;
}

std::size_t GrowableArena::bytes_reserved() const noexcept {
  std::size_t total = 0;
  for (const Block& block : blocks_) {
    total += block.capacity;
  }
  return total;
}

std::size_t GrowableArena::block_count() const noexcept {
  return blocks_.size();
}

}  // namespace hikoboshi::core::universal::memory
