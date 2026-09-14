#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "belief_update/detail/probability.hpp"

namespace belief_update {

enum class SourceType : std::uint8_t {
  direct_input,
  sensor,
  human,
  model,
  file,
  system,
};

enum class OverrideStatus : std::uint8_t {
  not_requested,
  requested,
  applied,
  rejected,
};

class SourceId final {
 public:
  explicit SourceId(std::string value) : value_(std::move(value)) {
    if (value_.empty()) {
      throw std::invalid_argument("source id must not be empty");
    }
  }

  SourceId(const SourceId&) = default;
  SourceId(SourceId&&) noexcept = default;
  SourceId& operator=(const SourceId&) = delete;
  SourceId& operator=(SourceId&&) = delete;

  [[nodiscard]] const std::string& value() const noexcept { return value_; }

 private:
  std::string value_;
};

class SourceVersion final {
 public:
  explicit SourceVersion(std::string value) : value_(std::move(value)) {
    if (value_.empty()) {
      throw std::invalid_argument("source version must not be empty");
    }
  }

  SourceVersion(const SourceVersion&) = default;
  SourceVersion(SourceVersion&&) noexcept = default;
  SourceVersion& operator=(const SourceVersion&) = delete;
  SourceVersion& operator=(SourceVersion&&) = delete;

  [[nodiscard]] const std::string& value() const noexcept { return value_; }

 private:
  std::string value_;
};

class ProvenanceTimestamp final {
 public:
  explicit ProvenanceTimestamp(std::int64_t unix_nanoseconds) noexcept
      : unix_nanoseconds_(unix_nanoseconds) {}

  ProvenanceTimestamp(const ProvenanceTimestamp&) = default;
  ProvenanceTimestamp(ProvenanceTimestamp&&) noexcept = default;
  ProvenanceTimestamp& operator=(const ProvenanceTimestamp&) = delete;
  ProvenanceTimestamp& operator=(ProvenanceTimestamp&&) = delete;

  [[nodiscard]] std::int64_t unix_nanoseconds() const noexcept {
    return unix_nanoseconds_;
  }

 private:
  std::int64_t unix_nanoseconds_;
};

class Provenance final {
 public:
  explicit Provenance(SourceType source_type, SourceId source_id,
                      SourceVersion version, ProvenanceTimestamp timestamp,
                      double confidence, OverrideStatus override_status)
      : source_type_(source_type),
        source_id_(std::move(source_id)),
        version_(std::move(version)),
        timestamp_(std::move(timestamp)),
        confidence_(detail::checked_probability(confidence)),
        override_status_(override_status) {}

  Provenance(const Provenance&) = default;
  Provenance(Provenance&&) noexcept = default;
  Provenance& operator=(const Provenance&) = delete;
  Provenance& operator=(Provenance&&) = delete;

  [[nodiscard]] SourceType source_type() const noexcept { return source_type_; }
  [[nodiscard]] const SourceId& source_id() const noexcept { return source_id_; }
  [[nodiscard]] const SourceVersion& version() const noexcept {
    return version_;
  }
  [[nodiscard]] const ProvenanceTimestamp& timestamp() const noexcept {
    return timestamp_;
  }
  [[nodiscard]] double confidence() const noexcept { return confidence_; }
  [[nodiscard]] OverrideStatus override_status() const noexcept {
    return override_status_;
  }

 private:
  SourceType source_type_;
  SourceId source_id_;
  SourceVersion version_;
  ProvenanceTimestamp timestamp_;
  double confidence_;
  OverrideStatus override_status_;
};

}  // namespace belief_update
