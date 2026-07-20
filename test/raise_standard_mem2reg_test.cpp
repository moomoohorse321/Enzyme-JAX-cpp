#include "src/enzyme_ad/jax/raise.h"

#include <iostream>
#include <string>

int main() {
  constexpr char input[] = R"llvm(
%Pair = type { i32, i32 }

define void @cpp_kernel(ptr %out) {
  %slot = alloca %Pair, align 4
  %field0 = getelementptr inbounds %Pair, ptr %slot, i64 0, i32 0
  %field1 = getelementptr inbounds %Pair, ptr %slot, i64 0, i32 1
  store i32 1, ptr %field0, align 4
  %value = load i32, ptr %field0, align 4
  store i32 %value, ptr %field1, align 4
  store i32 %value, ptr %out, align 4
  ret void
}
)llvm";

  std::string output = enzyme_jax::runLLVMToMLIRRoundTripWithTypes(
      input, "cpp_kernel", {{"int32_t", {1}}});
  if (output.empty()) {
    std::cerr << "LLVM-to-MLIR lowering failed\n";
    return 1;
  }
  if (output.find("alloca") != std::string::npos) {
    std::cerr << "aggregate alloca survived the RaiseLib pipeline:\n"
              << output;
    return 1;
  }
  return 0;
}
