#pragma once

// Force-included by MACA AOT/JIT builds so kernels can retain CUDA declarations.
// cu-bridge currently lacks the grid-constant parameter qualifier. Leave it
// intact once the compiler supports the attribute or provides its own macro.
#ifdef __MACACC__
#if !defined(__grid_constant__) && !__has_attribute(grid_constant)
#define __grid_constant__
#endif
#endif
