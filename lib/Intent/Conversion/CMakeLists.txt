add_mlir_library(MLIRIntentConversionCommon
  ConvertIntentToLinalg.cpp
  LowerScanRowPackets.cpp
  ExpandRsqrtForLibm.cpp
  TileLinalgByIteratorKind.cpp
  VectorizeLinalgRowReduction.cpp
  VectorizeLinalgMixedPrecisionDotReduction.cpp
  FuseLinalgSoftmax.cpp
  LowerFlashAttentionToLoops.cpp
  LowerRowMaskToLoops.cpp
  LowerPairedRotationToLoops.cpp
  LowerMatmulRowsToVectors.cpp
  LowerGatherChainToLoops.cpp
  FuseLinalgSiblingOutputs.cpp
  FuseLinalgPreservingBroadcastMath.cpp
  ForwardBufferResultsToOutParams.cpp
  LowerDisjointMemrefCopyToLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Intent/Conversion

  DEPENDS
  IntentConversionCommonPassIncGen

  LINK_LIBS PUBLIC
  MLIRIntent
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTransformUtils
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMathTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRTilingInterface
)

add_subdirectory(CUDA)
