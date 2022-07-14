set(PHY_SRCS
  PhyDialect.cpp
)

add_phy_dialect_library(Phy
  ${PHY_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR

  DEPENDS
  MLIRPhyEnumsIncGen
)
