######### This package contains functions
#            for automatically binding the
#            compiled object symbols for
#
##########      Sparse LOGREG             ##########
##author:       Michael Mader, MIPS/IBI
##initial vs:   25.03.2004
##last vs:      25.03.2004
###########################################################

.First.lib <- function(lib, pkg){
#  dyn.load(paste(.path.package("SparseLogReg"), "libs/SparseLogReg.so",
#                 sep="/"))
  library.dynam("SparseLogReg", pkg, lib)

}

