#ifndef HIKOBOSHI_CORE_UNIVERSAL_MEMORY_GROWABLE_ARENA_HPP
#define HIKOBOSHI_CORE_UNIVERSAL_MEMORY_GROWABLE_ARENA_HPP

#include <cstddef>
#include <vector>

namespace hikoboshi::core::universal::memory {

class GrowableArena {
 public:
  static constexpr std::size_t kDefaultBlockBytes = 64 * 1024;
  static constexpr std::size_t kDefaultAlignment = 64;

  explicit GrowableArena(
      std::size_t initial_block_bytes = kDefaultBlockBytes);
  ~GrowableArena();

  GrowableArena(const GrowableArena&) = delete;
  GrowableArena& operator=(const GrowableArena&) = delete;
  GrowableArena(GrowableArena&& other) noexcept;
  GrowableArena& operator=(GrowableArena&& other) noexcept;

  template <typename T>
  T* allocate(const std::size_t count) {
    return static_cast<T*>(allocate_bytes(sizeof(T), alignof(T), count));
  }

  void* allocate_bytes(std::size_t element_size,
                       std::size_t alignment,
                       std::size_t count);

  void reset() noexcept;
  void clear() noexcept;

  [[nodiscard]] std::size_t bytes_used() const noexcept;
  [[nodiscard]] std::size_t peak_bytes_used() const noexcept;
  [[nodiscard]] std::size_t bytes_reserved() const noexcept;
  [[nodiscard]] std::size_t block_count() const noexcept;

 private:
  struct Block {
    void* data;
    std::size_t capacity;
    std::size_t offset;
    std::size_t alignment;

    Block(std::size_t capacity_bytes, std::size_t block_alignment);
    ~Block();
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    Block(Block&& other) noexcept;
    Block& operator=(Block&& other) noexcept;
  };

  void add_block(std::size_t minimum_bytes, std::size_t alignment);

  std::vector<Block> blocks_;
  std::size_t initial_block_bytes_;
  std::size_t next_block_bytes_;
  std::size_t bytes_used_;
  std::size_t peak_bytes_used_;
};

}  // namespace hikoboshi::core::universal::memory

#endif  // HIKOBOSHI_CORE_UNIVERSAL_MEMORY_GROWABLE_ARENA_HPP
