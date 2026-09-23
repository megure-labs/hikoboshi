#include <hikoboshi/io/structure_embedding_cache.hpp>
#include <hikoboshi/universal/detail/sha256.hpp>
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>

using namespace hikoboshi;
void require(bool value, const char* label) {
  if (!value) throw std::runtime_error(label);
}
void require_ok(universal::Status status) {
  require(universal::is_ok(status), status.detail);
}
int main() {
  try {
    using universal::detail::Sha256;
    struct Vector { std::size_t length; const char* digest; };
    const Vector vectors[] = {
      {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {1, "ca358758f6d27e6cf45272937977a748fd88391db679ceda7dc7bf1f005ee879"},
      {55, "166cb1ce48dba01b7d55341ae1d847d73f24f538ecf4911f657fba255b83d2aa"},
      {56, "1bfb069cb406b899a507a8a00c5d721c207463d63e313b6dd994ce67cb46cc13"},
      {63, "f7191804fb70f054952e831548291633b7a1f300f0f4d877f38d995ed4ce274b"},
      {64, "2d91cac9246ea9f11939b47308360f0c8e8be87db686dc3c1d1dcbf91eb054cf"},
      {65, "d0c7d32839006da594139cfa302b1e478c238355573c80079aa4be0a94621c56"},
      {119, "fd06b902a80a7934de6c554139f009a618f21177609745903a149f81c80688a5"},
      {120, "b2b995b84fc277d63e13b96e5e7d94e4b0392761f43a48ba9d875e11a51b7cb3"},
      {127, "f6a779f356943adb60fffef004625272ae8e1421cce45d5a02d142a8f6229031"},
      {128, "477f9dbe9ced58a04330559b8e5a755753f3d72680062f2d72fe57e4b9370df1"},
      {129, "07780ed5a9c7be22ecc1ce1ba1e5432a7b8bfd61a2f43d67cd06622d96e0e90e"},
      {1000000, "e34a75ed76580fb5863eeb359217c231a9850babb72e9efd5c597b157003975f"},
    };
    for (const auto& sample : vectors) {
      std::vector<unsigned char> bytes(sample.length);
      for (std::size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<unsigned char>(i * 13 + 7);
      Sha256 whole, split;
      whole.update(bytes.data(), bytes.size());
      for (const auto byte : bytes) split.update(&byte, 1);
      require(Sha256::hex(whole.finish()) == sample.digest, "SHA256 known answer");
      require(split.finish() == whole.finish(), "SHA256 incremental boundary");
      require(split.finish() == whole.finish(), "SHA256 finish is nondestructive");
    }
    std::string pattern = (std::filesystem::temp_directory_path() / "hikoboshi-cache-test-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());name.push_back('\0');
    require(mkdtemp(name.data()) != nullptr, "temp directory");
    const std::filesystem::path directory(name.data());
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code e; std::filesystem::remove_all(path, e); } } cleanup{directory};
    io::DiskStructureEmbeddingCache cache;
    require_ok(cache.prepare(directory, "model-A/binary-A/fast"));
    std::array<float, 30> coordinates{};
    std::array<universal::AtomSource, 10> atoms{};
    atoms.fill(universal::AtomSource::Observed);
    universal::StructureView structure{};
    structure.residue_count = 2;
    structure.coordinates = {coordinates.data(), coordinates.size()};
    structure.atom_sources = {atoms.data(), atoms.size()};
    std::array<float, 128> expected{}, actual{};
    for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<float>(i) / 3;
    bool hit = true;
    require_ok(cache.load(structure, {actual.data(), actual.size()}, hit));
    require(!hit, "cold miss");
    require_ok(cache.store(structure, {expected.data(), expected.size()}));
    require_ok(cache.load(structure, {actual.data(), actual.size()}, hit));
    require(hit && actual == expected, "warm exact values");
    // Larger borrowed backing spans are valid; unused tail storage is not input.
    std::vector<float> padded_coordinates(coordinates.begin(), coordinates.end());
    padded_coordinates.push_back(123.0F);
    std::vector<universal::AtomSource> padded_atoms(atoms.begin(), atoms.end());
    padded_atoms.push_back(universal::AtomSource::Missing);
    auto padded = structure;
    padded.coordinates = {padded_coordinates.data(), padded_coordinates.size()};
    padded.atom_sources = {padded_atoms.data(), padded_atoms.size()};
    require_ok(cache.load(padded, {actual.data(), actual.size()}, hit));
    require(hit && actual == expected, "only consumed span contents affect identity");
    io::DiskStructureEmbeddingCache reopened;
    require_ok(reopened.prepare(directory, "model-A/binary-A/fast"));
    require_ok(reopened.load(structure, {actual.data(), actual.size()}, hit));
    require(hit, "independent instance reuse");
    for (const char* identity : {"model-B/binary-A/fast", "model-A/binary-B/fast", "model-A/binary-A/strict"}) {
      io::DiskStructureEmbeddingCache changed;
      require_ok(changed.prepare(directory, identity));
      require_ok(changed.load(structure, {actual.data(), actual.size()}, hit));
      require(!hit, "encoder identity invalidation");
    }
    coordinates[3] = 1.0F;
    require_ok(cache.load(structure, {actual.data(), actual.size()}, hit));
    require(!hit, "coordinate content invalidation"); coordinates[3] = 0;
    atoms[0] = universal::AtomSource::Inferred;
    require_ok(cache.load(structure, {actual.data(), actual.size()}, hit));
    require(!hit, "atom provenance invalidation"); atoms[0] = universal::AtomSource::Observed;
    require_ok(cache.load(structure, {actual.data(), 64}, hit));
    require(!hit, "embedding dimension invalidation");
    std::filesystem::path entry;
    for (const auto& file : std::filesystem::recursive_directory_iterator(directory))
      if (file.path().extension() == ".hke") entry = file.path();
    require(!entry.empty(), "cache entry exists");
    for (const std::size_t offset : {std::size_t{0}, std::size_t{20}, std::size_t{45}, std::size_t{60}, std::size_t{90}}) {
      std::fstream corrupt(entry, std::ios::in | std::ios::out | std::ios::binary);
      corrupt.seekg(offset);char value = 0;corrupt.read(&value, 1); value ^= 1;
      corrupt.seekp(offset);corrupt.write(&value, 1);corrupt.close();
      require_ok(cache.load(structure, {actual.data(), actual.size()}, hit));
      require(!hit, "corrupt entry rejected");
      require_ok(cache.store(structure, {expected.data(), expected.size()}));
    }
    std::filesystem::resize_file(entry, 90);
    require_ok(cache.load(structure, {actual.data(), actual.size()}, hit));
    require(!hit, "truncated payload rejected");
    require_ok(cache.store(structure, {expected.data(), expected.size()}));
    { std::ofstream append(entry, std::ios::app | std::ios::binary); append << 'x'; }
    require_ok(cache.load(structure, {actual.data(), actual.size()}, hit));
    require(!hit, "extra bytes rejected");
    require_ok(cache.store(structure, {expected.data(), expected.size()}));
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 4; ++worker) workers.emplace_back([&] {
      io::DiskStructureEmbeddingCache concurrent;
      require_ok(concurrent.prepare(directory, "model-A/binary-A/fast"));
      for (int i = 0; i < 40; ++i) {
        std::array<float, 128> read{}; bool found = false;
        require_ok(concurrent.store(structure, {expected.data(), expected.size()}));
        require_ok(concurrent.load(structure, {read.data(), read.size()}, found));
        require(found && read == expected, "atomic concurrent publication");
      }
    });
    for (auto& worker : workers) worker.join();
    require(cache.statistics().rejected == 7, "rejected count");
    for (const auto& file : std::filesystem::recursive_directory_iterator(directory))
      require(file.path().filename().string().find(".pending-") != 0, "no pending files after success");
    io::DiskStructureEmbeddingCache invalid;
    require(!universal::is_ok(invalid.prepare(entry, "identity")), "file directory rejected");
    require(!universal::is_ok(cache.load(structure, {nullptr, 128}, hit)), "invalid output rejected");
    const auto executable = io::current_executable_sha256();require_ok(executable.status);
    require(executable.value.size() == 64, "executable hash");
    std::cout << "embedding cache identity, integrity, concurrency and SHA256 checks passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
