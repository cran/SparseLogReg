######### This package contains functions
#            wrapping the C code for 
#
##########      Sparse LOGREG             ##########
##author:       Michael Mader, MIPS/IBI
##initial vs:   25.03.2004
##last vs:      25.03.2004
###########################################################

runSparseLogreg <- function(numTrains=62, numGenes=2000, numExperiments=100,
                gammaMin=0.01, gammaMax=4.0, numGamma=5,
                intKfold=3, tol=1e-6, maxFeatures=20, 
                inData, inClass, ...){
  
  m <- .C("findcounts",
          as.integer(numTrains),
          as.integer(numGenes),
          as.integer(numExperiments),
          as.double(gammaMin),
          as.double(gammaMax),
          as.integer(numGamma),
          as.integer(intKfold),
          as.double(tol),
          as.matrix(inData),
          as.matrix(inClass),
          out = matrix(nrow=numGenes, ncol=2, 0),
          PACKAGE = "SparseLogReg")
#  m$out is input for 
  m <- .C("findgenes",
          as.integer(numTrains),
          as.integer(numGenes),
          as.integer(numExperiments),
          as.double(gammaMin),
          as.double(gammaMax),
          as.integer(numGamma),
          as.integer(intKfold),
          as.double(tol),
          as.integer(maxFeatures),
          as.matrix(inData),
          as.matrix(inClass),
          as.matrix(m$out),
          out = matrix(nrow=, ncol= , 0),
          PACKAGE = "SparseLogReg")
  return(m$out)
}
