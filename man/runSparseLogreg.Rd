% file SparseLogReg/man/runSparseLogreg.Rd
% copyright (C) 2004 M. T. Mader
%
\name{runSparseLogreg}
\alias{runSparseLogreg}
\title{R interface for SparseLOGREG}
\description{
  This is a simplistic interface to SparseLOGREG
}
\usage{
runSparseLogreg(numTrains=62, numGenes=2000, numExperiments=100,
                gammaMin=0.01, gammaMax=4.0, numGamma=5,
                intKfold=3, tol=1e-6, maxFeatures=20, 
                inData, inClass, ...)
}
\arguments{
  \item{numTrains}{Number of training cases}
  \item{numGenes}{Number of variables/genes}
  \item{numExperiments}{Number of measurements/experiments}
  \item{gammaMin}{}
  \item{gammaMax}{}
  \item{numGamma}{number of Gamma}
  \item{intKfold}{number of internal k-folds}
  \item{tol}{tolerance}
  \item{maxFeatures}{}
  \item{inData}{Input data matrix}
  \item{inClass}{Classification vector (consisting of \code{c(0,1)}}
  %  \code{\link[CatAna]{categorialAnalysis}}}
  \item{...}{additional arguments are piped through to subfunctions}    
}
\value{
  \item{out}{result matrix of SparseLOGREG}
}
\author{M. T. Mader (interface), }
%\note{This function is an \emph{internal} subfunction of
%  \code{\link[CatAna]{categorial.wrapper}} and
%  \code{\link[CatAna]{localcat}}.
%}
\seealso{

}
\references{
Shevade, S. K. and Keerthi, S. S. (2003): A simple and efficient
algorithm for gene selection using sparse logistic regression.-
Bioinformatics 19(17): 2246-2253
}
\examples{
  %results <- runSparseLogreg()
}
\keyword{misc}


